# Instruct-TTS Chatterbox Implementation Context

## 1. Tổng quan Kiến trúc (Architecture Overview)

Dự án này mở rộng mô hình **Chatterbox (T3)** để hỗ trợ điều khiển giọng nói bằng văn bản (Instruction/Style Prompt) thay vì chỉ dùng vector tham chiếu (Reference Vector).

### Các thành phần chính:

1. **Instruction Encoder:**
* Sử dụng **T5-Large Encoder** (frozen) để hiểu ngữ nghĩa của instruction text.
* Áp dụng cơ chế **Attention Pooling** để nén chuỗi output của T5 thành một vector cố định `(Batch, 1024)`.


2. **Adapter Backbone (CustomLlamaModel):**
* Thay thế `LlamaModel` gốc bằng `CustomLlamaModel`.
* Sử dụng cơ chế **AdaRMSNorm (Adaptive RMS Norm)**: Điều biến (modulate) các features của Llama dựa trên instruction vector theo công thức: .
* Adapter được chèn vào trước Self-Attention và trước MLP block của mỗi lớp Decoder.


3. **Data Pipeline:**
* Cập nhật `Dataset` và `Collator` để đọc, tokenize và padding instruction text.
* Hỗ trợ đọc instruction từ cột metadata thứ 3 (Audio | Text | Instruction).



---

## 2. Implementation Details (Verified Code)

### 2.1. Encoder Module

**File:** `src/chatterbox/models/t3/modules/instruction_encoder.py`

* **Chức năng:** Chuyển text instruction thành vector embedding 1024 chiều.
* **Trạng thái:** ✅ Verified (AutoTokenizer, Frozen Parameters, Attention Pooling logic).

```python
import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer

class InstructionEncoder(nn.Module):
    def __init__(self, model_name="google/flan-t5-large", output_dim=1024):
        super().__init__()
        print(f"Loading T5 Encoder: {model_name}...")
        self.t5 = T5EncoderModel.from_pretrained(model_name)
        for param in self.t5.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.t5.config.d_model
        
        # Attention Pooling
        self.query = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8, batch_first=True)
        self.proj = nn.Linear(self.hidden_size, output_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask)
            encoder_hidden_states = outputs.last_hidden_state
            
        batch_size = input_ids.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        key_padding_mask = (attention_mask == 0) 
        
        style_emb, _ = self.attn(query, encoder_hidden_states, encoder_hidden_states, key_padding_mask=key_padding_mask)
        style_vector = self.proj(style_emb).squeeze(1) 
        return style_vector

```

### 2.2. Backbone Adapter

**File:** `src/chatterbox/models/t3/modeling_llama_adapter.py`

* **Chức năng:** Llama Model tùy chỉnh có khả năng nhận `instruction_emb`.
* **Trạng thái:** ✅ Verified (Injection Points, Zero-Init, Input Arguments).

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaModel as HFLlamaModel,
    LlamaDecoderLayer as HFLlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)

class AdaRMSNormAdapter(nn.Module):
    def __init__(self, hidden_size: int, instruction_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(instruction_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2)
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        
    def forward(self, x, instruction_emb, original_norm_layer):
        normed_x = original_norm_layer(x)
        if instruction_emb is None: return normed_x
        
        style_params = self.adapter(instruction_emb).unsqueeze(1)
        gamma, beta = style_params.chunk(2, dim=-1)
        return normed_x * (1 + gamma) + beta

class CustomLlamaDecoderLayer(HFLlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, adapter_config: dict):
        super().__init__(config, layer_idx)
        instr_dim = adapter_config.get("instruction_dim", 1024)
        self.input_adapter = AdaRMSNormAdapter(config.hidden_size, instr_dim)
        self.post_attention_adapter = AdaRMSNormAdapter(config.hidden_size, instr_dim)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, 
                output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, 
                instruction_emb=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_adapter(hidden_states, instruction_emb, self.input_layernorm)
        
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
            cache_position=cache_position, position_embeddings=position_embeddings, **kwargs
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_adapter(hidden_states, instruction_emb, self.post_attention_layernorm)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions: outputs += (self_attn_weights,)
        if use_cache: outputs += (present_key_value,)
        return outputs

class CustomLlamaModel(HFLlamaModel):
    def __init__(self, config: LlamaConfig, adapter_config: dict):
        super().__init__(config)
        self.layers = nn.ModuleList([
            CustomLlamaDecoderLayer(config, layer_idx, adapter_config) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, 
                inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, 
                return_dict=None, cache_position=None, instruction_emb=None):
        # ... (Standard Pre-processing Logic) ...
        # (Giữ nguyên logic xử lý inputs_embeds, cache, mask từ HFLlamaModel gốc)
        # Chỉ thay đổi đoạn loop qua layers để truyền instruction_emb:
        
        # ... [Code omitted for brevity, ensure exact copy from previous full version] ...
        
        for decoder_layer in self.layers:
            # ...
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                # ...
                instruction_emb=instruction_emb # <--- Critical Pass
            )
            # ...
        
        # ... (Return Standard Output) ...
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, ...)

```

### 2.3. Inference Backend Wrapper

**File:** `src/chatterbox/models/t3/inference/t3_hf_backend.py`

* **Chức năng:** Đảm bảo `instruction_emb` đi xuyên qua hàm `generate` của HF.
* **Trạng thái:** ✅ Verified.

```python
class T3HuggingfaceBackend(LlamaPreTrainedModel, GenerationMixin):
    # ... (__init__ giữ nguyên) ...

    @torch.inference_mode()
    def forward(self, inputs_embeds, past_key_values=None, use_cache=True, 
                output_attentions=False, output_hidden_states=True, return_dict=True, 
                instruction_emb=None, **kwargs): # <--- Capture kwargs
        
        tfmr_out = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            # ...
            instruction_emb=instruction_emb # <--- Pass down
        )
        # ... (Logits processing) ...
        return CausalLMOutputWithCrossAttentions(...)

```

### 2.4. Main Model (T3)

**File:** `src/chatterbox/models/t3/t3.py`

* **Chức năng:** Tích hợp Encoder, Backbone, xử lý Forward cho Train và Inference (có CFG).
* **Trạng thái:** ✅ Verified (Optional Instruction Handling, CFG Batch Expansion).

```python
# ... (Imports) ...
from .modeling_llama_adapter import CustomLlamaModel
from .modules.instruction_encoder import InstructionEncoder

class T3(nn.Module):
    def __init__(self, hp=T3Config()):
        super().__init__()
        # ...
        adapter_config = {"instruction_dim": 1024}
        self.tfmr = CustomLlamaModel(self.cfg, adapter_config)
        
        if getattr(hp, "use_instruction", True):
            self.instr_encoder = InstructionEncoder("google/flan-t5-large", 1024)
            self.instr_encoder.eval()
            for p in self.instr_encoder.parameters(): p.requires_grad = False
        # ...

    def forward(self, ..., instruction_input_ids=None, instruction_attention_mask=None, training=False):
        # ... (Embeddings prep) ...
        
        instruction_emb = None
        if hasattr(self, "instr_encoder") and instruction_input_ids is not None:
            instruction_emb = self.instr_encoder(instruction_input_ids, instruction_attention_mask)
            
        tfmr_out = self.tfmr.forward(..., instruction_emb=instruction_emb)
        # ...

    @torch.inference_mode()
    def inference(self, ..., instruction_input_ids=None, instruction_attention_mask=None, cfg_weight=0):
        # ...
        instruction_emb = None
        if hasattr(self, "instr_encoder") and instruction_input_ids is not None:
            instruction_emb = self.instr_encoder(instruction_input_ids.to(self.device), instruction_attention_mask.to(self.device))
            if cfg_weight > 0:
                instruction_emb = torch.cat([instruction_emb, instruction_emb], dim=0)
        
        # ... (In loop: call patched_model with instruction_emb) ...

```

### 2.5. Training Script & Data

**File:** `src/finetune_t3.py`

* **Chức năng:** Tokenize instruction, Collate padding, Metadata parsing (hỗ trợ 3 cột).
* **Trạng thái:** ✅ Verified (Bug fix: `instruction_text` init, Metadata parsing logic).

```python
# ... (Imports & Args) ...

class SpeechFineTuningDataset(Dataset):
    def __init__(self, ...):
        # ...
        self.instruction_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", use_fast=True)

    def _load_audio_text_from_item(self, idx):
        instruction_text = "" # <--- Quan trọng: Init default
        # ... (Logic load instruction từ HF dataset hoặc local file) ...
        return wav_16k, text, instruction_text

    def __getitem__(self, idx):
        # ...
        instruction_input_ids = self.instruction_tokenizer(
            instruction_text, return_tensors="pt", truncation=True, max_length=512, add_special_tokens=True
        ).input_ids.squeeze(0)
        
        return { ..., "instruction_input_ids": instruction_input_ids.long() }

@dataclass
class SpeechDataCollator:
    instruction_pad_token_id: int = 0
    def __call__(self, features):
        # ...
        instr_list = [f["instruction_input_ids"] for f in features]
        max_len = max(len(t) for t in instr_list) if instr_list else 0
        if max_len == 0: max_len = 1
        
        padded_instr = torch.stack([F.pad(t, (0, max_len-len(t)), value=0) for t in instr_list])
        mask_instr = (padded_instr != 0).long()
        
        return { ..., "instruction_input_ids": padded_instr, "instruction_attention_mask": mask_instr }

# ... (Main function: Updated metadata parsing logic) ...
# if len(parts) >= 2:
#     audio = parts[0]; text = parts[1]
#     instruction = parts[2] if len(parts) > 2 else ""

```

---

## 3. Hướng dẫn sử dụng (Usage)

### Định dạng dữ liệu (Metadata file)

File metadata (vd: `train.txt`) nên có định dạng pipe-separated (`|`) hoặc tab-separated:

```text
path/to/audio1.wav|Nội dung văn bản 1|Hãy nói với giọng vui vẻ và hào hứng
path/to/audio2.wav|Nội dung văn bản 2|Nói giọng buồn, chậm rãi

```

### Lệnh Training

```bash
CUDA_VISIBLE_DEVICES=0 uv run src/finetune_t3.py \
    --do_train \
    --output_dir "./chkpt/instruct_tts_v1" \
    --model_name_or_path "ResembleAI/Chatterbox" \
    --metadata_file "captts_sft_expresso.txt" \
    --instruction_column_name "caption" \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --save_steps 500 \
    --eval_steps 500 \
    --logging_steps 10 \
    --freeze_voice_encoder True \
    --freeze_s3gen True

```