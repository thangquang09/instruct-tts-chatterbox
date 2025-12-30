

## 🆕 Tóm tắt các thay đổi chính (Proposed vs Original)

### 1. **Instruction Encoder** (Hoàn toàn mới)
| Component | Chi tiết |
|-----------|----------|
| **T5 Tokenizer** | `google/flan-t5-large` tokenizer |
| **Flan-T5-Large Encoder** | FROZEN, không train |
| **Attention Pooling** | Learnable query `[1, 1, 1024]` + MultiheadAttention (8 heads) |
| **Linear Projection** | `1024 → 1024` với Xavier init |
| **Output** | `instruction_emb [Batch, 1024]` |

### 2. **CustomLlamaModel** (Thay thế LlamaModel gốc)
| Component | Chi tiết |
|-----------|----------|
| **CustomLlamaDecoderLayer** | 30 layers, mỗi layer có 2 adapters |
| **AdaRMSNormAdapter** | Adaptive Layer Normalization |
| **Adapter Formula** | `output = RMSNorm(x) × (1 + γ_ada) + β_ada` |
| **Zero-Init** | Last layer của adapter được init zeros để training ổn định |

### 3. **AdaRMSNormAdapter** (Adaptive Layer Norm)
```
instruction_emb [B, 1024]
       ↓
Linear(1024 → hidden) → SiLU → Linear(hidden → 2048)
       ↓
Split → γ_ada [B, 1, 1024], β_ada [B, 1, 1024]
       ↓
output = RMSNorm(x) × (1 + γ_ada) + β_ada
```

### 4. **Speaker Embedding Dropout** (Training trick)
- Trong training: **20% chance** speaker_emb bị zero-out
- Mục đích: Ép model học từ instruction text thay vì chỉ dựa vào speaker embedding
- Không áp dụng khi inference

### 5. **Trainable vs Frozen Parameters**
| Module | Status |
|--------|--------|
| T5 Encoder | ❄️ FROZEN |
| Attention Pooling Query | 🔥 Trainable |
| Attention Pooling MHA | 🔥 Trainable |
| Linear Projection | 🔥 Trainable |
| Original LLaMA weights | ❄️ FROZEN (backbone) |
| AdaRMSNormAdapter (×60) | 🔥 Trainable |
| Voice Encoder | ❄️ FROZEN |
| S3Gen | ❄️ FROZEN |

