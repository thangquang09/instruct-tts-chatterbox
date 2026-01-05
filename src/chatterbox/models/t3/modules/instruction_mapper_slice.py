
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
    """
    Mã hóa timestep t thành vector embedding
    Ref: DiTTo-TTS
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLNBlock(nn.Module):
    """
    Adaptive Layer Normalization Block.
    Đây là cơ chế cốt lõi để bơm thông tin Style (Condition) vào quá trình sinh.
    Thay vì cộng/nối đơn thuần, ta dùng style để scale & shift đặc trưng.
    Ref: NaturalSpeech 3, DiT.
    """
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # Dự đoán scale (gamma) và shift (beta) từ condition
        self.ada_proj = nn.Linear(cond_dim, hidden_dim * 2)
        
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.act = nn.SiLU()  # SiLU (Swish) hoạt động tốt hơn ReLU trong Diffusion
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)

        # Zero-initialization cho lớp cuối để quá trình train ổn định ban đầu
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x, condition):
        # x: [Batch, Hidden] (Noisy Vector)
        # condition: [Batch, Cond_Dim] (Combined Text + Time)
        
        # 1. Điều phối (Modulate)
        scale, shift = self.ada_proj(condition).chunk(2, dim=1)
        x_norm = self.norm(x) * (1 + scale) + shift
        
        # 2. MLP Processing
        h = self.linear1(x_norm)
        h = self.act(h)
        h = self.linear2(h)
        
        # 3. Residual Connection
        return x + h


class InstructionMapper(nn.Module):
    """
    Flow Matching Backbone.
    Nhiệm vụ: Dự đoán vector vận tốc (v) để biến Noise -> Speaker Latent.
    """
    def __init__(self, input_dim=1024, internal_dim=448, spk_emb_dim=256, xvec_dim=192, depth=6):
        super().__init__()
        
        self.internal_dim = internal_dim
        self.spk_emb_dim = spk_emb_dim
        self.xvec_dim = xvec_dim
        
        # 1. Condition Processing
        # Timestep embedding cho Flow Matching
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(internal_dim),
            nn.Linear(internal_dim, internal_dim),
            nn.SiLU(),
            nn.Linear(internal_dim, internal_dim),
        )
        
        # Projection cho Instruction Text (từ T5)
        self.cond_proj = nn.Linear(input_dim, internal_dim)
        
        # 2. Backbone (ResNet-MLP with AdaLN)
        self.input_proj = nn.Linear(internal_dim, internal_dim)  # Chiếu noise x_t về hidden
        
        self.blocks = nn.ModuleList([
            AdaLNBlock(internal_dim, internal_dim) for _ in range(depth)
        ])
        
        self.final_norm = nn.LayerNorm(internal_dim)
        
        # 3. Output Heads (Dự đoán velocity v)
        # Output của backbone là velocity field trong không gian Latent chung (512 dim)
        self.output_proj = nn.Linear(internal_dim, internal_dim)
        
        # 4. Normalizers (Registered as buffers/modules to be saved with state_dict)
        self.spk_normalizer = StreamingNormalizer(spk_emb_dim)
        self.xvec_normalizer = StreamingNormalizer(xvec_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Zero-init cho output cuối cùng của Flow Matching
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, t, style_emb):
        """
        x: [Batch, 448] - Noisy Vector ở bước t
        t: [Batch] - Timestep (0 đến 1)
        style_emb: [Batch, 1024] - Instruction Embedding từ T5
        
        Returns:
            v_pred: [Batch, 448] - Dự đoán hướng di chuyển (velocity)
        """
        # 1. Chuẩn bị Condition
        t_emb = self.time_mlp(t)                    # [B, 448]
        c_emb = self.cond_proj(style_emb)           # [B, 448]
        
        # Cộng gộp Time và Text Condition (hoặc có thể concat)
        global_cond = t_emb + c_emb                 # [B, 448]
        
        # 2. Backbone Processing
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, global_cond)  # AdaLN Injection
            
        h = self.final_norm(h)
        
        # 3. Output Velocity
        v_pred = self.output_proj(h)
        
        return v_pred

        v_pred = self.output_proj(h)
        
        return v_pred

    def denormalize(self, z_norm):
        """
        De-normalize vector z về miền giá trị gốc.
        z_norm: [Batch, 448] (Concatenated: SpkNorm | XVecNorm)
        """
        spk_norm = z_norm[:, :self.spk_emb_dim]
        xvec_norm = z_norm[:, self.spk_emb_dim:self.spk_emb_dim+self.xvec_dim]
        
        # De-normalize từng phần: y = x * sigma + mu
        # BatchNorm tracking: running_mean, running_var
        # Inverse: y = x * sqrt(var + eps) + mean
        
        # Speaker Embedding
        if hasattr(self, 'spk_normalizer'):
             # eps default of BatchNorm1d is 1e-5
            std = torch.sqrt(self.spk_normalizer.running_var + 1e-5)
            spk_emb = spk_norm * std + self.spk_normalizer.running_mean
        else:
            spk_emb = spk_norm
            
        # X-Vector
        if hasattr(self, 'xvec_normalizer'):
            std = torch.sqrt(self.xvec_normalizer.running_var + 1e-5)
            x_vector = xvec_norm * std + self.xvec_normalizer.running_mean
        else:
            x_vector = xvec_norm
            
        return spk_emb, x_vector

    def project_to_targets(self, latent_z):
        """
        Hàm này chỉ gọi khi Inference xong (đã có z_0 sạch từ ODE Solver).
        latent_z: [Batch, 448] (Kết quả của Flow Matching)
        """
        # De-normalize trước khi tách
        spk_emb, x_vector = self.denormalize(latent_z)
        return spk_emb, x_vector

    @torch.no_grad()
    def inference(self, style_emb, num_steps=10):
        """
        Inference: Sample từ noise về latent sạch bằng Euler ODE Solver.
        style_emb: [Batch, 1024]
        """
        batch_size = style_emb.shape[0]
        device = style_emb.device
        
        # Bắt đầu từ nhiễu Gaussian (Standard Normal Distribution)
        # Vì ta train trên Normalized Data (mean=0, std=1), nên prior chính là N(0, I)
        x = torch.randn(batch_size, self.internal_dim, device=device)
        
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            v = self.forward(x, t, style_emb)
            x = x + v * dt
        
        return self.project_to_targets(x)


class StreamingNormalizer(nn.Module):
    """
    Wrapper quanh BatchNorm1d để dùng làm Normalizer (Affine=False).
    Chúng ta dùng BatchNorm để track running_mean và running_var.
    """
    def __init__(self, dim):
        super().__init__()
        # affine=False: Không học gamma/beta, chỉ track stats
        # track_running_stats=True: Track global mean/var
        self.bn = nn.BatchNorm1d(dim, affine=False, track_running_stats=True)

    def forward(self, x):
        # Khi train: Update stats và normalize batch hiện tại
        # Khi eval: Dùng stored stats để normalize
        
        # BatchNorm expect [N, C] hoặc [N, C, L]
        # Input x: [Batch, Dim] -> OK
        return self.bn(x)
        
    @property
    def running_mean(self):
        return self.bn.running_mean
        
    @property
    def running_var(self):
        return self.bn.running_var
