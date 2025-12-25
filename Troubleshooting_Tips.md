# Troubleshooting Tips: Deep Learning Training Issues

> **Tài liệu này được tạo từ kinh nghiệm debug issue NaN loss trong dự án Instruct-TTS-Chatterbox (Dec 2025)**

---

## 📋 Table of Contents

1. [Quick Diagnosis Checklist](#quick-diagnosis-checklist)
2. [NaN/Inf Loss Debugging Playbook](#naninf-loss-debugging-playbook)
3. [FP16/Mixed Precision Issues](#fp16mixed-precision-issues)
4. [Common Root Causes](#common-root-causes)
5. [Debug Logging Templates](#debug-logging-templates)
6. [Prevention Strategies](#prevention-strategies)

---

## 🎯 Quick Diagnosis Checklist

**Khi gặp NaN/Inf loss, hãy kiểm tra theo thứ tự ưu tiên sau:**

```
□ 1. [5 min] Tắt FP16, chạy lại với FP32 → NaN hết? → FP16 là nguyên nhân
□ 2. [2 min] Kiểm tra learning rate (thử giảm 10x)
□ 3. [5 min] In ra input tensors: có NaN không? Device đúng chưa?
□ 4. [10 min] Thêm debug logging tại TỪNG module → xác định NaN xuất hiện ở đâu
□ 5. [5 min] Kiểm tra loss function với edge cases (empty labels, all masked)
□ 6. [5 min] Verify trainable parameters (param.requires_grad == True?)
```

---

## 🔥 NaN/Inf Loss Debugging Playbook

### Step 1: Binary Search để xác định nguồn NaN

**Nguyên tắc vàng: Đừng đoán, hãy đo!**

```python
# Thêm vào forward() của từng module
def debug_tensor(name, tensor, step=None):
    """In debug info cho tensor"""
    if tensor is None:
        print(f"[DEBUG] {name}: None")
        return
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    status = "✓" if not (has_nan or has_inf) else "⚠️ NaN!" if has_nan else "⚠️ Inf!"
    
    print(f"[DEBUG] {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, {status}")
    
    if has_nan or has_inf:
        print(f"  → min={tensor[~torch.isnan(tensor)].min().item():.4f}, "
              f"max={tensor[~torch.isnan(tensor)].max().item():.4f}")
```

### Step 2: Cô lập module có vấn đề

```python
# Trong main forward pass, thêm checkpoints
def forward(self, ...):
    # Checkpoint 1: Input
    debug_tensor("input_embeds", embeds)
    
    # Checkpoint 2: Sau mỗi sub-module
    instruction_emb = self.instr_encoder(...)
    debug_tensor("instruction_emb", instruction_emb)  # ← NaN xuất hiện ở đây!
    
    # Checkpoint 3: Hidden states
    hidden_states = self.backbone(...)
    debug_tensor("hidden_states", hidden_states)
    
    # Checkpoint 4: Logits
    logits = self.output_head(hidden_states)
    debug_tensor("logits", logits)
```

### Step 3: Drill down vào module có vấn đề

```python
# Khi đã biết module nào có NaN, thêm debug chi tiết hơn
class InstructionEncoder(nn.Module):
    def forward(self, input_ids, attention_mask=None):
        # Debug từng bước trong module
        debug_tensor("1. input_ids", input_ids)
        
        outputs = self.t5(input_ids, attention_mask)
        debug_tensor("2. t5_output", outputs.last_hidden_state)  # ← NaN ở đây!
        
        style_emb, _ = self.attn(query, outputs.last_hidden_state, ...)
        debug_tensor("3. after_attention", style_emb)
        
        result = self.proj(style_emb)
        debug_tensor("4. final_output", result)
        
        return result
```

---

## ⚡ FP16/Mixed Precision Issues

### Các module HAY GÂY NaN với FP16

| Module | Nguyên nhân | Giải pháp |
|--------|-------------|-----------|
| `nn.MultiheadAttention` | Softmax overflow với large values | Force FP32 |
| `LayerNorm` / `RMSNorm` | Division by small variance | Force FP32 hoặc eps lớn hơn |
| Large pretrained models (T5, BERT) | Internal ops không stable với FP16 | Wrap trong `autocast(enabled=False)` |
| Cross-entropy loss | Log của values gần 0 | Clamp logits hoặc FP32 loss |

### Pattern: Force FP32 cho module không stable

```python
def forward(self, input_ids, attention_mask=None):
    # Force FP32 cho toàn bộ module
    with torch.amp.autocast('cuda', enabled=False):
        # Cast inputs về FP32
        query = self.query.float()
        
        # Forward pass trong FP32
        outputs = self.encoder(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state.float()
        
        # Attention trong FP32
        attn_output, _ = self.attn(query, hidden_states, hidden_states)
        
        result = self.proj(attn_output)
    
    return result  # Có thể cast lại FP16 nếu cần
```

### Quick Test: FP16 có phải nguyên nhân không?

```bash
# Test 1: Chạy với FP32
python train.py --fp16 false

# Test 2: Nếu FP32 work → FP16 là vấn đề
# Tìm module nào cần force FP32
```

---

## 🎯 Common Root Causes

### 1. Incorrect Parameter Freezing

**Symptom**: `grad_norm=nan` hoặc loss không giảm

**Check**:
```python
# Kiểm tra trainable parameters
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
frozen = [n for n, p in model.named_parameters() if not p.requires_grad]

print(f"Trainable: {len(trainable)}")
print(f"Frozen: {len(frozen)}")

# In chi tiết nếu nghi ngờ
for name in trainable[:10]:
    print(f"  ✓ {name}")
```

**Common Mistake**:
```python
# ❌ SAI: Freeze cả module, bao gồm adapter trainable
for param in model.encoder.parameters():
    param.requires_grad = False

# ✓ ĐÚNG: Chỉ freeze phần cần thiết
for param in model.encoder.pretrained_part.parameters():
    param.requires_grad = False
# Giữ adapter trainable
for param in model.encoder.adapter.parameters():
    param.requires_grad = True
```

### 2. Device/Dtype Mismatch

**Symptom**: RuntimeError hoặc silent NaN

**Check**:
```python
def check_model_devices(model):
    devices = set()
    dtypes = set()
    for name, param in model.named_parameters():
        devices.add(str(param.device))
        dtypes.add(str(param.dtype))
    print(f"Devices: {devices}")
    print(f"Dtypes: {dtypes}")
```

**Fix Pattern**:
```python
def forward(self, input_ids, ...):
    # Explicit device alignment
    input_ids = input_ids.to(self.device)
    
    # Explicit dtype alignment
    embeds = self.embed(input_ids)
    condition = condition.to(dtype=embeds.dtype, device=embeds.device)
```

### 3. Edge Cases trong Loss Computation

**Symptom**: NaN loss với một số batches

**Common Cases**:
```python
# ❌ All labels masked → NaN
loss = F.cross_entropy(logits, labels, ignore_index=-100)
# Nếu tất cả labels == -100 → loss = nan

# ✓ Handle edge case
valid_tokens = (labels != -100).sum()
if valid_tokens == 0:
    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
else:
    loss = F.cross_entropy(logits, labels, ignore_index=-100)
```

### 4. Uninitialized/Bad Weight Initialization

**Symptom**: NaN ngay từ step đầu tiên

**Fix**:
```python
# Proper initialization cho attention
nn.init.xavier_uniform_(self.attn.in_proj_weight)
nn.init.xavier_uniform_(self.attn.out_proj.weight)
nn.init.zeros_(self.attn.out_proj.bias)

# Scaled initialization cho learnable parameters
self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
```

---

## 📝 Debug Logging Templates

### Template 1: Training Loop Debug

```python
# Trong custom training step
def training_step(self, model, inputs):
    # Log inputs
    logger.info(f"[Step {self.state.global_step}] Input shapes:")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            has_nan = torch.isnan(v).any().item()
            logger.info(f"  {k}: {v.shape}, nan={has_nan}")
    
    # Forward
    outputs = model(**inputs)
    
    # Log outputs
    loss = outputs.loss
    logger.info(f"[Step {self.state.global_step}] Loss: {loss.item():.4f}")
    
    if torch.isnan(loss):
        logger.error("NaN loss detected! Dumping debug info...")
        self._dump_debug_info(inputs, outputs)
        raise ValueError("NaN loss")
    
    return loss
```

### Template 2: Module-level Debug

```python
class MyModule(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
    
    def forward(self, x):
        if self.debug:
            self._check_tensor("input", x)
        
        # ... operations ...
        
        if self.debug:
            self._check_tensor("output", output)
        
        return output
    
    def _check_tensor(self, name, tensor):
        if tensor is None:
            print(f"[{self.__class__.__name__}] {name}: None")
            return
        
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        
        if nan_count > 0 or inf_count > 0:
            print(f"[{self.__class__.__name__}] ⚠️ {name}: "
                  f"NaN={nan_count}, Inf={inf_count}, shape={tensor.shape}")
```

---

## 🛡️ Prevention Strategies

### 1. Pre-training Sanity Checks

```python
def sanity_check_before_training(model, dataloader, device):
    """Chạy trước khi training để phát hiện issues sớm"""
    
    print("=" * 50)
    print("PRE-TRAINING SANITY CHECK")
    print("=" * 50)
    
    # Check 1: Model devices
    print("\n1. Checking model devices...")
    devices = {p.device for p in model.parameters()}
    print(f"   Model on devices: {devices}")
    
    # Check 2: Trainable params
    print("\n2. Checking trainable parameters...")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Check 3: Forward pass
    print("\n3. Testing forward pass (1 batch, no grad)...")
    model.eval()
    batch = next(iter(dataloader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in batch.items()}
    
    with torch.no_grad():
        try:
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            print(f"   Forward pass OK, loss = {loss.item():.4f}")
            
            if torch.isnan(loss):
                print("   ⚠️ WARNING: Loss is NaN!")
        except Exception as e:
            print(f"   ❌ Forward pass FAILED: {e}")
            return False
    
    # Check 4: Backward pass
    print("\n4. Testing backward pass...")
    model.train()
    try:
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        loss.backward()
        
        # Check gradients
        nan_grads = sum(1 for p in model.parameters() 
                       if p.grad is not None and torch.isnan(p.grad).any())
        print(f"   Backward pass OK, NaN gradients: {nan_grads}")
        
        model.zero_grad()
    except Exception as e:
        print(f"   ❌ Backward pass FAILED: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("SANITY CHECK PASSED ✓")
    print("=" * 50)
    return True
```

### 2. Gradient Monitoring

```python
# Thêm vào training loop
def check_gradients(model, step):
    """Monitor gradients for anomalies"""
    total_norm = 0
    nan_params = []
    inf_params = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            
            if torch.isnan(param.grad).any():
                nan_params.append(name)
            if torch.isinf(param.grad).any():
                inf_params.append(name)
    
    total_norm = total_norm ** 0.5
    
    if nan_params:
        print(f"[Step {step}] ⚠️ NaN gradients in: {nan_params[:5]}...")
    if inf_params:
        print(f"[Step {step}] ⚠️ Inf gradients in: {inf_params[:5]}...")
    if total_norm > 100:
        print(f"[Step {step}] ⚠️ Large grad norm: {total_norm:.2f}")
    
    return total_norm
```

### 3. FP16 Compatibility Test

```python
def test_fp16_compatibility(model, sample_input, device):
    """Test if model works with FP16"""
    
    print("Testing FP16 compatibility...")
    model = model.to(device)
    
    # Test FP32
    model.float()
    with torch.no_grad():
        out_fp32 = model(**sample_input)
        loss_fp32 = out_fp32.loss.item()
    print(f"  FP32 loss: {loss_fp32:.4f}")
    
    # Test FP16
    model.half()
    sample_input_fp16 = {k: v.half() if v.dtype == torch.float32 else v 
                         for k, v in sample_input.items()}
    
    with torch.no_grad():
        try:
            out_fp16 = model(**sample_input_fp16)
            loss_fp16 = out_fp16.loss.item()
            print(f"  FP16 loss: {loss_fp16:.4f}")
            
            if torch.isnan(out_fp16.loss):
                print("  ⚠️ FP16 produces NaN! Use FP32 or fix unstable modules.")
                return False
        except Exception as e:
            print(f"  ❌ FP16 failed: {e}")
            return False
    
    # Test autocast
    model.float()
    with torch.amp.autocast('cuda'):
        out_autocast = model(**sample_input)
        loss_autocast = out_autocast.loss.item()
    print(f"  Autocast loss: {loss_autocast:.4f}")
    
    if torch.isnan(out_autocast.loss):
        print("  ⚠️ Autocast produces NaN!")
        return False
    
    print("  ✓ FP16 compatibility OK")
    return True
```

---

## 📚 Lessons Learned (Case Study: Instruct-TTS)

### Vấn đề gặp phải
- **Triệu chứng**: `loss: nan`, `grad_norm: nan` khi fine-tune model với instruction encoder
- **Thời gian debug**: ~2 giờ (quá lâu!)

### Root Causes tìm được
1. **FP16 + T5 + MultiheadAttention** = Numerical instability
2. **Incorrect freezing**: Freeze cả trainable adapter components
3. **Device mismatch**: Instruction tensors không được move sang GPU
4. **Edge case**: All-masked labels gây NaN trong cross-entropy

### Sai lầm trong quá trình debug
1. ❌ Không test FP32 ngay từ đầu (mất 30 phút)
2. ❌ Sửa nhiều thứ cùng lúc thay vì từng cái một
3. ❌ Không có debug logging từ đầu
4. ❌ Không verify trainable parameters

### Nên làm gì
1. ✓ **Bước đầu tiên**: Tắt FP16, test với FP32
2. ✓ **Bước hai**: Thêm debug logging ở các checkpoints quan trọng
3. ✓ **Bước ba**: Binary search để tìm module có vấn đề
4. ✓ **Bước bốn**: Kiểm tra edge cases trong loss computation

---

## 🔧 Quick Reference Commands

```bash
# Debug mode: giảm batch size, tăng logging
python train.py --per_device_train_batch_size 1 \
                --logging_steps 1 \
                --max_steps 10 \
                --fp16 false

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1

# Monitor training loss real-time
tail -f logs/train.log | grep -E "(loss|nan|inf)"
```

---

*Last updated: December 2025*
*Generated from debugging session: Instruct-TTS-Chatterbox NaN loss issue*

