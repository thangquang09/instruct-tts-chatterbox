#!/usr/bin/env python3
"""
Pytest test suite to verify InstructionChatterBox model loading.

Tests:
1. Checkpoint files exist
2. Mapper checkpoint structure is correct
3. InstructionEncoder weights load correctly (query adapter changes)
4. InstructionMapper weights load correctly
5. T3 + InstructionEncoder integration works
6. Full inference pipeline produces correct shapes
7. Weights are not random (specific value checks)

Usage:
    pytest test_instruction_loading.py -v
    pytest test_instruction_loading.py -v -s  # with print output
"""

import pytest
import torch
from pathlib import Path

# =================== Configuration ===================
# Update these paths to match your setup
T3_CKPT_DIR = Path("checkpoints/t3_instruct_ddp")
MAPPER_CKPT = Path("checkpoints/mapper_slice_v4/best_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =================== Fixtures ===================
@pytest.fixture(scope="module")
def mapper_checkpoint():
    """Load mapper checkpoint once for all tests."""
    if not MAPPER_CKPT.exists():
        pytest.skip(f"Mapper checkpoint not found: {MAPPER_CKPT}")
    return torch.load(MAPPER_CKPT, map_location="cpu")


@pytest.fixture(scope="module")
def instruction_encoder():
    """Create and load InstructionEncoderT5."""
    from chatterbox.models.t3.modules.instruction_encoder import InstructionEncoderT5

    encoder = InstructionEncoderT5(model_name="google/flan-t5-large")
    return encoder


@pytest.fixture(scope="module")
def instruction_mapper():
    """Create InstructionMapper."""
    from chatterbox.models.t3.modules.instruction_mapper_slice import InstructionMapper

    mapper = InstructionMapper()
    return mapper


@pytest.fixture(scope="module")
def tokenizer():
    """Load T5 tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("google/flan-t5-large")


# =================== Test: Checkpoint Exists ===================
class TestCheckpointExists:
    """Test that required checkpoint files exist."""

    def test_mapper_checkpoint_exists(self):
        """Mapper checkpoint file exists."""
        assert MAPPER_CKPT.exists(), f"Mapper checkpoint not found: {MAPPER_CKPT}"

    def test_t3_checkpoint_dir_exists(self):
        """T3 checkpoint directory exists (optional - may not be trained yet)."""
        # This is a soft check - T3 may not be trained yet
        if not T3_CKPT_DIR.exists():
            pytest.skip("T3 checkpoint not trained yet (this is optional)")

    def test_t3_has_required_files(self):
        """T3 checkpoint has required files (if exists)."""
        if not T3_CKPT_DIR.exists():
            pytest.skip("T3 checkpoint directory not found")

        required_files = ["t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json"]
        for filename in required_files:
            filepath = T3_CKPT_DIR / filename
            assert filepath.exists(), f"Missing file: {filepath}"


# =================== Test: Checkpoint Structure ===================
class TestCheckpointStructure:
    """Test checkpoint internal structure."""

    def test_mapper_has_encoder_key(self, mapper_checkpoint):
        """Mapper checkpoint contains 'encoder' key."""
        assert "encoder" in mapper_checkpoint, (
            "Missing 'encoder' key in mapper checkpoint"
        )

    def test_mapper_has_mapper_key(self, mapper_checkpoint):
        """Mapper checkpoint contains 'mapper' key."""
        assert "mapper" in mapper_checkpoint, (
            "Missing 'mapper' key in mapper checkpoint"
        )

    def test_encoder_has_query_adapter(self, mapper_checkpoint):
        """Encoder weights include query adapter."""
        encoder_keys = list(mapper_checkpoint["encoder"].keys())
        has_query = any("query" in k for k in encoder_keys)
        assert has_query, (
            f"Encoder missing 'query' adapter. Keys: {encoder_keys[:5]}..."
        )

    def test_encoder_key_count(self, mapper_checkpoint):
        """Encoder has expected number of keys (~225 for T5-large adapters)."""
        encoder_keys = mapper_checkpoint["encoder"]
        # InstructionEncoderT5 with adapters should have around 225 keys
        assert len(encoder_keys) > 200, f"Encoder has too few keys: {len(encoder_keys)}"

    def test_mapper_key_count(self, mapper_checkpoint):
        """Mapper has expected number of keys."""
        mapper_keys = mapper_checkpoint["mapper"]
        # InstructionMapper should have multiple keys for flow network
        assert len(mapper_keys) > 10, f"Mapper has too few keys: {len(mapper_keys)}"


# =================== Test: Weight Loading ===================
class TestWeightLoading:
    """Test that weights load correctly and change model parameters."""

    def test_encoder_query_weight_changes(self, instruction_encoder, mapper_checkpoint):
        """Loading encoder weights changes the query adapter."""
        # Clone original query before loading
        original_query = instruction_encoder.query.clone()

        # Load weights
        instruction_encoder.load_state_dict(mapper_checkpoint["encoder"], strict=False)

        # Check query changed
        query_changed = not torch.equal(original_query, instruction_encoder.query)
        assert query_changed, "Query adapter not updated after loading weights!"

    def test_mapper_weights_load_without_error(
        self, instruction_mapper, mapper_checkpoint
    ):
        """Mapper weights load without missing/unexpected keys."""
        missing, unexpected = instruction_mapper.load_state_dict(
            mapper_checkpoint["mapper"], strict=False
        )

        assert len(missing) == 0, f"Missing keys in mapper: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys in mapper: {unexpected}"

    def test_encoder_missing_keys_are_t5_only(
        self, instruction_encoder, mapper_checkpoint
    ):
        """Missing keys when loading encoder are only T5 base weights (expected)."""
        missing, unexpected = instruction_encoder.load_state_dict(
            mapper_checkpoint["encoder"], strict=False
        )

        # All missing keys should be from frozen T5 encoder (t5.encoder.*)
        for key in missing:
            assert "t5" in key, f"Unexpected missing key (not T5): {key}"


# =================== Test: T3 Integration ===================
class TestT3Integration:
    """Test T3 model integration with InstructionEncoder."""

    def test_t3_has_instruction_encoder(self):
        """T3 model has instr_encoder attribute."""
        from chatterbox.models.t3 import T3

        t3 = T3()
        assert hasattr(t3, "instr_encoder"), "T3 missing 'instr_encoder' attribute"

    def test_t3_encoder_weight_update(self, mapper_checkpoint):
        """T3's instr_encoder updates when loading mapper weights."""
        from chatterbox.models.t3 import T3

        t3 = T3()
        original_query = t3.instr_encoder.query.clone()

        t3.instr_encoder.load_state_dict(mapper_checkpoint["encoder"], strict=False)

        query_changed = not torch.equal(original_query, t3.instr_encoder.query)
        assert query_changed, "T3's instr_encoder.query not updated!"


# =================== Test: Inference Shapes ===================
class TestInferenceShapes:
    """Test that inference produces correct output shapes."""

    def test_encoder_output_shape(
        self, instruction_encoder, mapper_checkpoint, tokenizer
    ):
        """Encoder produces [B, 1024] style embedding."""
        instruction_encoder.load_state_dict(mapper_checkpoint["encoder"], strict=False)
        instruction_encoder.to(DEVICE).eval()

        instruction = "Speak in a warm and friendly voice."
        tokens = tokenizer(
            instruction, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = tokens.input_ids.to(DEVICE)
        attention_mask = tokens.attention_mask.to(DEVICE)

        with torch.no_grad():
            style_emb = instruction_encoder(input_ids, attention_mask)

        assert style_emb.shape == (1, 1024), f"Wrong style_emb shape: {style_emb.shape}"

    def test_mapper_output_shapes(
        self, instruction_encoder, instruction_mapper, mapper_checkpoint, tokenizer
    ):
        """Mapper produces [B, 256] spk_emb and [B, 192] x_vector."""
        # Load weights
        instruction_encoder.load_state_dict(mapper_checkpoint["encoder"], strict=False)
        instruction_mapper.load_state_dict(mapper_checkpoint["mapper"])

        instruction_encoder.to(DEVICE).eval()
        instruction_mapper.to(DEVICE).eval()

        instruction = "An elderly man with a raspy voice."
        tokens = tokenizer(
            instruction, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = tokens.input_ids.to(DEVICE)
        attention_mask = tokens.attention_mask.to(DEVICE)

        with torch.no_grad():
            style_emb = instruction_encoder(input_ids, attention_mask)
            spk_emb, x_vector = instruction_mapper.inference(style_emb, num_steps=10)

        assert spk_emb.shape == (1, 256), f"Wrong spk_emb shape: {spk_emb.shape}"
        assert x_vector.shape == (1, 192), f"Wrong x_vector shape: {x_vector.shape}"


# =================== Test: Weight Values (Not Random) ===================
# =================== Test: Weight Values (Exact Match) ===================
class TestWeightValues:
    """Test that loaded weights match checkpoint exactly and are not random."""

    def test_encoder_query_matches_checkpoint(
        self, instruction_encoder, mapper_checkpoint
    ):
        """Encoder weights in model match exactly with checkpoint values."""
        # Load weights
        instruction_encoder.load_state_dict(mapper_checkpoint["encoder"], strict=False)

        # Check 'query' weight
        model_weight = instruction_encoder.query
        ckpt_weight = mapper_checkpoint["encoder"]["query"]

        # Move to same device for comparison (ckpt is usually CPU)
        model_weight_cpu = model_weight.cpu()
        ckpt_weight_cpu = ckpt_weight.cpu()

        # Exact match check
        assert torch.equal(model_weight_cpu, ckpt_weight_cpu), (
            "Encoder query weight does not match checkpoint!"
        )

        # Stats check (trained weights usually have smaller std than random init)
        weight_std = model_weight.std().item()
        print(f"\n  Encoder query std: {weight_std:.6f}")
        # Random init (e.g. Xavier) typically has higher std for this dim
        assert weight_std < 0.5, (
            f"Encoder query std {weight_std} too high (looks random)"
        )

    def test_mapper_weights_match_checkpoint(
        self, instruction_mapper, mapper_checkpoint
    ):
        """Mapper weights in model match exactly with checkpoint values."""
        # Load weights
        instruction_mapper.load_state_dict(mapper_checkpoint["mapper"])

        # Pick a random key to verify, e.g., first layer weight
        key = list(mapper_checkpoint["mapper"].keys())[0]

        # Handle state_dict key mapping (sometimes keys have prefixes)
        # Here we assume direct mapping as per load_state_dict logic
        model_dict = instruction_mapper.state_dict()
        assert key in model_dict, f"Key {key} not found in model state_dict"

        model_weight = model_dict[key]
        ckpt_weight = mapper_checkpoint["mapper"][key]

        # Move to same device for comparison (ckpt is usually CPU)
        model_weight_cpu = model_weight.cpu()
        ckpt_weight_cpu = ckpt_weight.cpu()

        # Exact match check
        assert torch.equal(model_weight_cpu, ckpt_weight_cpu), (
            f"Mapper weight '{key}' does not match checkpoint!"
        )

    def test_embeddings_are_normalized(
        self, instruction_encoder, instruction_mapper, mapper_checkpoint, tokenizer
    ):
        """Mapper produces unit-normalized embeddings (L2 norm ≈ 1)."""
        instruction_encoder.load_state_dict(mapper_checkpoint["encoder"], strict=False)
        instruction_mapper.load_state_dict(mapper_checkpoint["mapper"])

        instruction_encoder.to(DEVICE).eval()
        instruction_mapper.to(DEVICE).eval()

        instruction = "A young girl with a cheerful voice."
        tokens = tokenizer(
            instruction, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = tokens.input_ids.to(DEVICE)
        attention_mask = tokens.attention_mask.to(DEVICE)

        with torch.no_grad():
            style_emb = instruction_encoder(input_ids, attention_mask)
            spk_emb, x_vector = instruction_mapper.inference(style_emb, num_steps=20)

        # Check L2 norms are close to 1 (normalized)
        spk_norm = spk_emb.norm(dim=-1).item()
        xvec_norm = x_vector.norm(dim=-1).item()

        assert 0.9 < spk_norm < 1.1, f"SpkEmb not normalized: L2={spk_norm}"
        assert 0.9 < xvec_norm < 1.1, f"X-Vector not normalized: L2={xvec_norm}"


# =================== Test: Diversity Check ===================
class TestDiversity:
    """Test that different instructions produce different embeddings."""

    def test_different_instructions_give_different_embeddings(
        self, instruction_encoder, instruction_mapper, mapper_checkpoint, tokenizer
    ):
        """Different instructions produce different speaker embeddings."""
        instruction_encoder.load_state_dict(mapper_checkpoint["encoder"], strict=False)
        instruction_mapper.load_state_dict(mapper_checkpoint["mapper"])

        instruction_encoder.to(DEVICE).eval()
        instruction_mapper.to(DEVICE).eval()

        instructions = [
            "A deep male voice.",
            "A high-pitched female voice.",
        ]

        embeddings = []
        for instr in instructions:
            tokens = tokenizer(
                instr, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = tokens.input_ids.to(DEVICE)
            attention_mask = tokens.attention_mask.to(DEVICE)

            with torch.no_grad():
                style_emb = instruction_encoder(input_ids, attention_mask)
                spk_emb, _ = instruction_mapper.inference(style_emb, num_steps=20)
                embeddings.append(spk_emb)

        # Check embeddings are different
        emb1, emb2 = embeddings
        cosine_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()

        # Different voices should have cosine < 0.9
        assert cosine_sim < 0.95, f"Embeddings too similar: cosine={cosine_sim}"


# =================== Test: T3 Adapters (Zero-Init) ===================
class TestT3Adapters:
    """Test T3 internal adapters (AdaRMSNorm) are initialized correctly."""

    def test_adapters_exist_and_zero_init(self):
        """T3 adapters exist and are zero-initialized (ready for finetuning)."""
        from chatterbox.models.t3 import T3

        # Initialize T3 (checks initialization logic)
        print("\n[Initializing T3 to check adapters...]")
        t3 = T3()

        # Check a few layers (start, middle, end) to ensure consistency
        num_layers = len(t3.tfmr.layers)
        indices_to_check = [0, num_layers // 2, num_layers - 1]

        for i in indices_to_check:
            layer = t3.tfmr.layers[i]

            # Check existence
            # Based on modeling_llama_adapter.py: CustomLlamaDecoderLayer has input_adapter and post_attention_adapter
            assert hasattr(layer, "input_adapter"), f"Layer {i} missing input_adapter"
            assert hasattr(layer, "post_attention_adapter"), (
                f"Layer {i} missing post_attention_adapter"
            )

            # Check Zero-Init
            # AdaRMSNormAdapter.adapter is nn.Sequential(Linear, SiLU, Linear)
            # The last Linear layer (index -1) determines the output scale (gamma, beta)

            # 1. Input Adapter (replaces input_layernorm behavior)
            input_last_linear = layer.input_adapter.adapter[-1]
            assert isinstance(input_last_linear, torch.nn.Linear)

            weight_max = input_last_linear.weight.abs().max().item()
            bias_max = input_last_linear.bias.abs().max().item()

            assert weight_max == 0.0, (
                f"Layer {i} Input Adapter weight not zero! Max: {weight_max}"
            )
            assert bias_max == 0.0, (
                f"Layer {i} Input Adapter bias not zero! Max: {bias_max}"
            )

            # 2. Post-Attention Adapter
            post_attn_last_linear = layer.post_attention_adapter.adapter[-1]
            assert isinstance(post_attn_last_linear, torch.nn.Linear)

            weight_max = post_attn_last_linear.weight.abs().max().item()
            bias_max = post_attn_last_linear.bias.abs().max().item()

            assert weight_max == 0.0, (
                f"Layer {i} Post-Attn Adapter weight not zero! Max: {weight_max}"
            )
            assert bias_max == 0.0, (
                f"Layer {i} Post-Attn Adapter bias not zero! Max: {bias_max}"
            )

        print("  [✓] All T3 adapters verified as Zero-Initialized")


# =================== Test: T3 Finetuned Adapter Loading ===================
class TestT3AdapterLoading:
    """Test loading finetuned T3 adapter weights from checkpoint."""

    T3_CKPT_FILE = T3_CKPT_DIR / "t3_cfg.safetensors"

    @pytest.fixture(scope="class")
    def t3_checkpoint(self):
        """Load T3 checkpoint."""
        if not self.T3_CKPT_FILE.exists():
            pytest.skip(f"T3 checkpoint not found: {self.T3_CKPT_FILE}")

        from safetensors.torch import load_file

        return load_file(self.T3_CKPT_FILE)

    def test_adapter_keys_exist_in_checkpoint(self, t3_checkpoint):
        """Checkpoint contains adapter keys."""
        adapter_keys = [k for k in t3_checkpoint.keys() if "adapter" in k]

        assert len(adapter_keys) > 0, "No adapter keys found in checkpoint!"
        print(f"\n  Found {len(adapter_keys)} adapter keys in checkpoint")

        # Check both input_adapter and post_attention_adapter exist
        has_input = any("input_adapter" in k for k in adapter_keys)
        has_post_attn = any("post_attention_adapter" in k for k in adapter_keys)

        assert has_input, "Missing input_adapter keys in checkpoint"
        assert has_post_attn, "Missing post_attention_adapter keys in checkpoint"

    def test_adapter_weights_are_trained(self, t3_checkpoint):
        """Adapter weights are not zero (have been trained)."""
        # Get first layer's input_adapter last linear weight
        adapter_key = None
        for k in t3_checkpoint.keys():
            if "layers.0.input_adapter.adapter.2.weight" in k:
                adapter_key = k
                break

        if adapter_key is None:
            pytest.skip("Could not find layer 0 input_adapter weight key")

        weight = t3_checkpoint[adapter_key]
        weight_max = weight.abs().max().item()

        # Trained weights should NOT be all zeros
        assert weight_max > 0.0, (
            f"Adapter weights are all zeros (not trained)! Max: {weight_max}"
        )
        print(f"\n  Adapter weight max: {weight_max:.6f} (trained)")

    def test_load_adapter_weights_into_t3(self, t3_checkpoint):
        """Load checkpoint weights into T3 and verify match."""
        from chatterbox.models.t3 import T3

        print("\n[Loading T3 and applying checkpoint weights...]")
        t3 = T3()

        # Get T3's tfmr state dict keys
        tfmr_state = t3.tfmr.state_dict()

        # Load only adapter weights
        adapter_state = {}
        for k, v in t3_checkpoint.items():
            # Keys in checkpoint are like "tfmr.layers.0.input_adapter..."
            # We need to strip "tfmr." prefix for tfmr.load_state_dict
            if k.startswith("tfmr.") and "adapter" in k:
                new_key = k.replace("tfmr.", "")
                adapter_state[new_key] = v

        if len(adapter_state) == 0:
            pytest.skip("No adapter keys with 'tfmr.' prefix found")

        # Load weights
        missing, unexpected = t3.tfmr.load_state_dict(adapter_state, strict=False)

        # There will be many missing keys (non-adapter weights)
        # But there should be no unexpected keys
        assert len(unexpected) == 0, f"Unexpected keys: {unexpected[:5]}..."

        # Verify a specific weight matches
        sample_key = list(adapter_state.keys())[0]
        model_weight = t3.tfmr.state_dict()[sample_key].cpu()
        ckpt_weight = adapter_state[sample_key].cpu()

        assert torch.equal(model_weight, ckpt_weight), (
            f"Weight '{sample_key}' does not match after loading!"
        )
        print(f"  [✓] Adapter weights loaded and verified")


# =================== Test: Accuracy vs Ground Truth ===================
class TestAccuracyVsGroundTruth:
    """Test predicted embeddings vs ground truth from real audio.

    WARNING: This test is SLOW (~30s) because it loads ChatterboxTTS.
    Run with: pytest test_instruction_loading.py::TestAccuracyVsGroundTruth -v -s
    """

    VAL_MANIFEST = Path("data/final_data_val.txt")
    NUM_SAMPLES = 5
    COSINE_THRESHOLD = 0.4  # Minimum acceptable cosine similarity

    @pytest.fixture(scope="class")
    def chatterbox_model(self):
        """Load ChatterboxTTS for ground truth extraction."""
        from chatterbox.tts import ChatterboxTTS

        print("\n[Loading ChatterboxTTS for GT extraction...]")
        model = ChatterboxTTS.from_pretrained(device=DEVICE)
        return model

    @pytest.fixture(scope="class")
    def loaded_encoder_mapper(self, mapper_checkpoint):
        """Load encoder and mapper with trained weights."""
        from chatterbox.models.t3.modules.instruction_encoder import (
            InstructionEncoderT5,
        )
        from chatterbox.models.t3.modules.instruction_mapper_slice import (
            InstructionMapper,
        )

        encoder = InstructionEncoderT5(model_name="google/flan-t5-large")
        encoder.load_state_dict(mapper_checkpoint["encoder"], strict=False)
        encoder.to(DEVICE).eval()

        mapper = InstructionMapper()
        mapper.load_state_dict(mapper_checkpoint["mapper"])
        mapper.to(DEVICE).eval()

        return encoder, mapper

    @pytest.fixture(scope="class")
    def validation_samples(self):
        """Load first N samples from validation manifest."""
        if not self.VAL_MANIFEST.exists():
            pytest.skip(f"Validation manifest not found: {self.VAL_MANIFEST}")

        samples = []
        with open(self.VAL_MANIFEST, "r") as f:
            for i, line in enumerate(f):
                if i >= self.NUM_SAMPLES:
                    break
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    samples.append(
                        {
                            "audio_path": parts[0],
                            "text": parts[1],
                            "instruction": parts[2],
                        }
                    )

        if len(samples) == 0:
            pytest.skip("No valid samples found in manifest")

        return samples

    def test_spk_emb_accuracy(
        self, chatterbox_model, loaded_encoder_mapper, validation_samples, tokenizer
    ):
        """Predicted SpkEmb has reasonable cosine similarity with GT."""
        import librosa
        from chatterbox.models.s3tokenizer import S3_SR

        encoder, mapper = loaded_encoder_mapper
        voice_encoder = chatterbox_model.ve

        cosine_scores = []

        for sample in validation_samples:
            audio_path = sample["audio_path"]
            instruction = sample["instruction"]

            # Ground truth: VoiceEncoder
            wav_16k, _ = librosa.load(audio_path, sr=S3_SR, mono=True)
            gt_spk_np = voice_encoder.embeds_from_wavs(
                [wav_16k], sample_rate=S3_SR, as_spk=False
            )
            gt_spk = torch.from_numpy(gt_spk_np[0]).to(DEVICE)

            # Prediction: Encoder + Mapper
            tokens = tokenizer(
                instruction, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = tokens.input_ids.to(DEVICE)
            attention_mask = tokens.attention_mask.to(DEVICE)

            with torch.no_grad():
                style_emb = encoder(input_ids, attention_mask)
                pred_spk, _ = mapper.inference(style_emb, num_steps=20)

            # Cosine similarity
            cos = torch.nn.functional.cosine_similarity(
                pred_spk, gt_spk.unsqueeze(0)
            ).item()
            cosine_scores.append(cos)

        avg_cos = sum(cosine_scores) / len(cosine_scores)
        print(f"\n  SpkEmb cosines: {[f'{c:.3f}' for c in cosine_scores]}")
        print(f"  SpkEmb avg cosine: {avg_cos:.4f}")

        assert avg_cos > self.COSINE_THRESHOLD, (
            f"SpkEmb avg cosine {avg_cos:.4f} < threshold {self.COSINE_THRESHOLD}"
        )

    def test_x_vector_accuracy(
        self, chatterbox_model, loaded_encoder_mapper, validation_samples, tokenizer
    ):
        """Predicted X-Vector has reasonable cosine similarity with GT."""
        import librosa
        from chatterbox.models.s3tokenizer import S3_SR

        encoder, mapper = loaded_encoder_mapper
        speaker_encoder = chatterbox_model.s3gen.speaker_encoder
        speaker_encoder.eval()

        cosine_scores = []

        for sample in validation_samples:
            audio_path = sample["audio_path"]
            instruction = sample["instruction"]

            # Ground truth: CAMPPlus (s3gen.speaker_encoder)
            wav_16k, _ = librosa.load(audio_path, sr=S3_SR, mono=True)
            wav_tensor = torch.from_numpy(wav_16k).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                gt_xvec = speaker_encoder.inference(wav_tensor).squeeze(0)

            # Prediction: Encoder + Mapper
            tokens = tokenizer(
                instruction, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = tokens.input_ids.to(DEVICE)
            attention_mask = tokens.attention_mask.to(DEVICE)

            with torch.no_grad():
                style_emb = encoder(input_ids, attention_mask)
                _, pred_xvec = mapper.inference(style_emb, num_steps=20)

            # Cosine similarity
            cos = torch.nn.functional.cosine_similarity(
                pred_xvec, gt_xvec.unsqueeze(0)
            ).item()
            cosine_scores.append(cos)

        avg_cos = sum(cosine_scores) / len(cosine_scores)
        print(f"\n  X-Vector cosines: {[f'{c:.3f}' for c in cosine_scores]}")
        print(f"  X-Vector avg cosine: {avg_cos:.4f}")

        assert avg_cos > self.COSINE_THRESHOLD, (
            f"X-Vector avg cosine {avg_cos:.4f} < threshold {self.COSINE_THRESHOLD}"
        )


# =================== Main ===================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
