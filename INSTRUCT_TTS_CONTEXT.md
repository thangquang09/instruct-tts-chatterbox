# Instruct-TTS Chatterbox - Implementation Context

> **Version:** 2.0 (Dec 2025) - Updated with bug fixes and lessons learned

---

## 1. Tổng quan Kiến trúc

Dự án mở rộng **Chatterbox T3** để điều khiển giọng nói bằng **Instruction Text** thay vì chỉ dùng Reference Audio.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INSTRUCT-TTS PIPELINE                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Instruction Text ──► [T5 Encoder] ──► [Attn Pooling] ──► (1024)     │
│                         (frozen)        (trainable)                   │
│                                              │                        │
│                                              ▼                        │
│  Text + Speech ──► [CustomLlamaModel] ◄── AdaRMSNorm Modulation      │
│     Tokens              (trainable)                                   │
│                              │                                        │
│                              ▼                                        │
│                      [Text + Speech Heads] ──► Logits                │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Thành phần chính

| Component | File | Trainable | Mô tả |
|-----------|------|-----------|-------|
| **InstructionEncoder** | `modules/instruction_encoder.py` | Partial | T5 frozen, Attn+Proj trainable |
| **CustomLlamaModel** | `modeling_llama_adapter.py` | ✅ Yes | LlamaModel + AdaRMSNorm adapters |
| **AdaRMSNormAdapter** | `modeling_llama_adapter.py` | ✅ Yes | Modulates hidden states với instruction |
| **T3** | `t3.py` | ✅ Yes | Main model, tích hợp tất cả |

---

## 2. Implementation (Verified & Fixed Code)

### 2.1. InstructionEncoder

**File:** `src/chatterbox/models/t3/modules/instruction_encoder.py`

**⚠️ CRITICAL:** Module này phải chạy trong **FP32** để tránh NaN với FP16 training.

```python
class InstructionEncoder(nn.Module):
    def __init__(self, model_name="google/flan-t5-large", output_dim=1024):
        super().__init__()
        self.t5 = T5EncoderModel.from_pretrained(model_name)
        
        # Clone tied weights để fix Safetensors saving issue
        if hasattr(self.t5, "shared") and hasattr(self.t5.encoder, "embed_tokens"):
            if self.t5.shared.weight.data_ptr() == self.t5.encoder.embed_tokens.weight.data_ptr():
                self.t5.encoder.embed_tokens.weight.data = self.t5.encoder.embed_tokens.weight.data.clone()
        
        # Freeze T5
        for param in self.t5.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.t5.config.d_model
        
        # Trainable components với proper initialization
        self.query = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)  # Scaled init!
        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8, batch_first=True)
        self.proj = nn.Linear(self.hidden_size, output_dim)
        
        # Xavier init for stability
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, input_ids, attention_mask):
        # ⚠️ CRITICAL: Disable autocast to avoid FP16 NaN issues
        with torch.amp.autocast('cuda', enabled=False):
            with torch.no_grad():
                outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask)
                encoder_hidden_states = outputs.last_hidden_state.detach().float()
            
            batch_size = input_ids.shape[0]
            query = self.query.float().expand(batch_size, -1, -1)
            key_padding_mask = (attention_mask == 0).to(device=encoder_hidden_states.device)
            
            # Handle all-masked case
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask[all_masked] = False
            
            style_emb, _ = self.attn(query, encoder_hidden_states, encoder_hidden_states, 
                                     key_padding_mask=key_padding_mask)
            style_vector = F.linear(style_emb, self.proj.weight.float(), 
                                   self.proj.bias.float()).squeeze(1)
        
        return style_vector
```

### 2.2. AdaRMSNormAdapter

**File:** `src/chatterbox/models/t3/modeling_llama_adapter.py`

```python
class AdaRMSNormAdapter(nn.Module):
    """Modulates RMSNorm output based on instruction embedding."""
    
    def __init__(self, hidden_size: int, instruction_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(instruction_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2)
        )
        # Zero-init for residual learning
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        
    def forward(self, x, instruction_emb, original_norm_layer):
        normed_x = original_norm_layer(x)
        
        # ⚠️ Handle None instruction_emb (important for inference without instruction)
        if instruction_emb is None:
            return normed_x
        
        style_params = self.adapter(instruction_emb).unsqueeze(1)
        gamma, beta = style_params.chunk(2, dim=-1)
        return normed_x * (1 + gamma) + beta
```

### 2.3. T3 Integration (Key Parts)

**File:** `src/chatterbox/models/t3/t3.py`

```python
class T3(nn.Module):
    def __init__(self, hp=T3Config()):
        # ...
        adapter_config = {"instruction_dim": 1024}
        self.tfmr = CustomLlamaModel(self.cfg, adapter_config)
        
        # InstructionEncoder - T5 frozen inside, adapters trainable
        self.instr_encoder = InstructionEncoder("google/flan-t5-large", 1024)
        # ⚠️ Do NOT freeze entire instr_encoder here - let finetune script control it

    def forward(self, ..., instruction_input_ids=None, instruction_attention_mask=None):
        # ...
        instruction_emb = None
        if hasattr(self, "instr_encoder") and instruction_input_ids is not None:
            # ⚠️ Ensure correct device
            instruction_input_ids = instruction_input_ids.to(self.device)
            if instruction_attention_mask is not None:
                instruction_attention_mask = instruction_attention_mask.to(self.device)
            
            instruction_emb = self.instr_encoder(instruction_input_ids, instruction_attention_mask)
            # ⚠️ Align dtype with embeddings
            instruction_emb = instruction_emb.to(dtype=embeds.dtype)
        
        tfmr_out = self.tfmr.forward(..., instruction_emb=instruction_emb)
```

---

## 3. Training Script Key Points

**File:** `src/finetune_t3.py`

### 3.1. Freezing Strategy

```python
# 1. Freeze Voice Encoder và S3Gen
for param in model.ve.parameters():
    param.requires_grad = False
for param in model.s3gen.parameters():
    param.requires_grad = False

# 2. Unfreeze T3 (main model)
for param in model.t3.parameters():
    param.requires_grad = True

# 3. ⚠️ Re-freeze T5 inside InstructionEncoder (chỉ adapter trainable)
if hasattr(model.t3, 'instr_encoder') and hasattr(model.t3.instr_encoder, 't5'):
    for param in model.t3.instr_encoder.t5.parameters():
        param.requires_grad = False
```

### 3.2. Loss Handling Edge Case

```python
def loss(self, logits_for_speech, labels_speech, ...):
    # ⚠️ Handle all-masked labels (would cause NaN)
    valid_speech_tokens = (labels_speech != IGNORE_ID).sum()
    if valid_speech_tokens == 0:
        loss_speech = torch.tensor(0.0, device=device, requires_grad=self.training)
    else:
        loss_speech = F.cross_entropy(logits_for_speech.transpose(1, 2), 
                                      labels_speech, ignore_index=IGNORE_ID)
```

### 3.3. Saving với Safetensors

```python
# ⚠️ Clone tensors to avoid shared memory issue (T5 tied weights)
finetuned_t3_state_dict = {k: v.clone().contiguous() 
                           for k, v in model.t3.state_dict().items()}
save_file(finetuned_t3_state_dict, output_path)
```

---

## 4. Training Command

```bash
# ⚠️ KHÔNG dùng --fp16 (gây NaN trong InstructionEncoder)
CUDA_VISIBLE_DEVICES=0 uv run accelerate launch src/finetune_t3.py \
    --do_train \
    --output_dir "./chkpt/instruct_tts_v1" \
    --model_name_or_path "ResembleAI/Chatterbox" \
    --metadata_file "captts_sft_expresso.txt" \
    --instruction_column_name "caption" \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --save_steps 500 \
    --logging_steps 10 \
    --freeze_voice_encoder True \
    --freeze_s3gen True \
    --dataloader_num_workers 8 \
    --save_safetensors False
```

---

## 5. Data Format

**Metadata file** (pipe-separated):
```
path/to/audio1.wav|Nội dung văn bản 1|Hãy nói với giọng vui vẻ và hào hứng
path/to/audio2.wav|Nội dung văn bản 2|Nói giọng buồn, chậm rãi
```

---

## 6. ⚠️ Known Issues & Fixes

### Issue 1: NaN Loss với FP16

**Nguyên nhân:** T5 + MultiheadAttention không stable với FP16

**Fix:** Wrap `InstructionEncoder.forward()` trong `torch.amp.autocast('cuda', enabled=False)`

### Issue 2: Safetensors Shared Memory Error

**Nguyên nhân:** T5 có tied weights (`shared.weight` == `embed_tokens.weight`)

**Fix:** Clone tensors trước khi save, hoặc dùng `--save_safetensors False`

### Issue 3: All-Masked Labels → NaN Loss

**Nguyên nhân:** `F.cross_entropy` với tất cả labels = `ignore_index` → NaN

**Fix:** Check `valid_tokens == 0` và return `0.0` thay vì compute loss

### Issue 4: Device Mismatch

**Nguyên nhân:** Instruction tensors không được move sang GPU

**Fix:** Explicit `.to(self.device)` trong `T3.forward()`

---

## 7. Debugging Checklist

Khi gặp NaN loss:

```
□ 1. Tắt --fp16, chạy lại → Nếu hết NaN → FP16 là nguyên nhân
□ 2. Thêm debug logging: torch.isnan(tensor).any()
□ 3. Check trainable params: sum(p.requires_grad for p in model.parameters())
□ 4. Verify devices: tensor.device cho mọi inputs
□ 5. Check edge cases: all labels masked?
```

Chi tiết: Xem `Troubleshooting_Tips.md`

---

*Last Updated: December 2025*

---

## 8. Instruction Mapper (Stage 1 Pre-training)

### 8.1. Tổng quan 2-Stage Training Strategy

Để giảm sự phụ thuộc vào Reference Audio, ta cần train một **Mapper** để ánh xạ từ **Instruction Text** sang **Speaker Embedding** và **X-Vector**.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    2-STAGE TRAINING STRATEGY                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  STAGE 1: Train Instruction Mapper (Riêng biệt)                          │
│  ═══════════════════════════════════════════════                          │
│                                                                           │
│  Instruction ──► [T5] ──► [Query+Attn] ──► style_emb ──► [Mapper]        │
│     Text       FREEZE      TRAIN              │           TRAIN          │
│                                               │              │           │
│                                               │         ┌────┴────┐      │
│                                               │         ▼         ▼      │
│                                               │     SpkEmb     X-Vec     │
│                                               │      (256)     (192)     │
│                                               │         │         │      │
│                Audio (GT) ──► [VoiceEnc] ────►├─────────┴─────────┤      │
│                            ──► [CAM++]  ─────►│   Flow Matching   │      │
│                                               │   Loss (MSE)      │      │
│                                               └───────────────────┘      │
│                                                                           │
│  STAGE 2: Finetune T3 (Sử dụng Mapper đã train)                          │
│  ══════════════════════════════════════════════                          │
│                                                                           │
│  Instruction ──► [T5+Query+Attn] ──► style_emb ──► [proj] ──► T3         │
│                       FREEZE              │        TRAIN      TRAIN      │
│                                           ▼                              │
│                                    [Mapper Heads] (FREEZE)               │
│                                     → SpkEmb, X-Vec → S3Gen              │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 8.2. InstructionMapper Architecture (Flow Matching)

**File:** `src/chatterbox/models/t3/modules/instruction_mapper.py`

Sử dụng kiến trúc **Conditional Flow Matching** với **AdaLN (Adaptive Layer Normalization)**.

```python
class InstructionMapper(nn.Module):
    """
    Flow Matching Backbone.
    Input: style_emb (1024) từ InstructionEncoder
    Output: velocity prediction (512) cho ODE sampling
    Final: SpkEmb (256), X-Vec (192)
    """
    def __init__(self, input_dim=1024, internal_dim=512, 
                 spk_emb_dim=256, xvec_dim=192, depth=6):
        # Time embedding (Sinusoidal)
        self.time_mlp = SinusoidalPosEmb + MLP
        
        # Condition projection
        self.cond_proj = Linear(1024 → 512)
        
        # ResNet-MLP với AdaLN blocks
        self.blocks = [AdaLNBlock(512, 512) for _ in range(6)]
        
        # Output heads
        self.head_spk = Linear(512 → 256)
        self.head_xvec = Linear(512 → 192)
    
    def forward(self, x, t, style_emb):
        """Train mode: Predict velocity v."""
        return v_pred  # [B, 512]
    
    def inference(self, style_emb, num_steps=10):
        """Inference mode: ODE sampling → (spk_emb, x_vector)"""
        return spk_emb, x_vector
```

### 8.3. InstructionEncoder (Refactored)

**File:** `src/chatterbox/models/t3/modules/instruction_encoder.py`

Đã refactor để loại bỏ projection layer thừa. Chỉ output `style_emb` size 1024.

```python
def forward(self, input_ids, attention_mask):
    # ... T5 + Attention Pooling ...
    style_emb, _ = self.attn(query, encoder_hidden_states, ...)  # [B, 1, 1024]
    return style_emb.squeeze(1)  # [B, 1024] - Trunk output
```

### 8.4. Two-Phase Training Scripts (Stage 1)

Training được chia thành 2 script riêng biệt để đảm bảo ổn định tối đa:

**Phase 1: `run_phase1.sh` (GT Reconstruction)**
- **Mục tiêu:** Train stable heads & backbone projection.
- **Input Reconstruction:** Dùng Ground Truth `x_1` (concat GT SpkEmb + XVec).
- **Flag:** Default (không dùng `--use_predicted_recon`).
- **Early Stopping:** Dựa trên Avg Cosine Similarity.

**Phase 2: `run_phase2.sh` (Predicted Reconstruction)**
- **Mục tiêu:** End-to-end optimization, giảm gap giữa train/inference.
- **Input Reconstruction:** Dùng `x_1_pred = x_t + v_pred * (1-t)`.
- **Resume:** Load checkpoint từ Phase 1.
- **Flag:** `--use_predicted_recon --reset_best_cos`.
- **LR:** Thấp hơn (5e-5) để fine-tune.

### 8.5. Training Components & Fixes

| Component | Trainable | Ghi chú |`
|-----------|-----------|---------|
| T5 Encoder | ❄️ Frozen | Pretrained knowledge |
| InstructionEncoder | ✅ Train | Query + Attn (No Proj) |
| InstructionMapper | ✅ Train | AdaLN blocks, Heads |
| **Target Construction** | ❄️ **FIXED** | `x_1` = Concat[GT_Spk, GT_XVec, Pad] (No LatentProjector!) |

> ⚠️ **CRITICAL FIX:** Target của Flow Matching phải là **FIXED** (concat GT embeddings). Không dùng trainable projector để tạo target, tránh hiện tượng collapse loss.

### 8.6. Training Results (300k Data - Dec 2025)

Kết quả training Phase 1 trên dataset 300k samples:

| Epoch | Flow Loss | Recon Loss | SpkCos | XVecCos | Avg Cos |
|-------|-----------|------------|--------|---------|---------|
| 1 | 0.496 | 0.0005 | **0.237** | **0.111** | 0.174 |
| 2 | 0.058 | 0.0000 | **0.789** | **0.515** | **0.652** |

**Nhận xét:** Model hội tụ cực nhanh với dữ liệu lớn. SpkCos đạt ~0.79 chỉ sau 2 epochs.

### 8.7. Loading cho Stage 2

Sau khi xong Phase 2, load weights để fine-tune T3:

```python
# 1. Load Mapper weights
checkpoint = torch.load("./checkpoints/mapper_phase2/best_model.pt")

# 2. Load InstructionEncoder (Query + Attn)
model.t3.instr_encoder.load_state_dict(checkpoint['encoder'], strict=False)

# 3. Load InstructionMapper (cho S3Gen inference)
model.instruction_mapper.load_state_dict(checkpoint['mapper'])
model.instruction_mapper.eval()  # Freeze mapper in Stage 2
```

### 8.8. Tensor Shapes

| Tensor | Shape | Mô tả |
|--------|-------|-------|
| `style_emb` | [B, 1024] | Trunk output (T5 + Attn Pooling) |
| `x_1` | [B, 512] | Fixed target = [SpkEmb, XVec, Pad] |
| `v_pred` | [B, 512] | Predicted velocity |
| `spk_emb` | [B, 256] | Speaker Embedding (cho T3 T3Cond) |
| `x_vector` | [B, 192] | CAM++ X-Vector (cho S3Gen) |
