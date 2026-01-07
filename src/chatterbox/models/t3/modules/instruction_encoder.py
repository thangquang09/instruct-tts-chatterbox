"""
InstructionEncoder
=============================================
Encodes instruction text into style embeddings using T5 + Attention Pooling.

Architecture:
  Instruction Text → T5 (frozen) → Attention Pooling → style_emb [B, 1024]
"""

import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer


class InstructionEncoderT5(nn.Module):
    """
    Encodes instruction text into a style embedding vector.

    Output: style_emb [Batch, hidden_size=1024] - Raw pooled representation.

    Downstream tasks should apply their own projection heads:
    - InstructionMapper: cond_proj (1024 → 512) for Flow Matching
    - AdaRMSNormAdapter: adapter (1024 → hidden_size) for T3 modulation
    """

    def __init__(self, model_name="google/flan-t5-large"):
        super().__init__()
        print(f"Loading T5 Encoder: {model_name}...")
        self.t5 = T5EncoderModel.from_pretrained(model_name)

        # Fix Safetensors tied weights issue
        if hasattr(self.t5, "shared") and hasattr(self.t5.encoder, "embed_tokens"):
            if (
                self.t5.shared.weight.data_ptr()
                == self.t5.encoder.embed_tokens.weight.data_ptr()
            ):
                print(
                    "Info: Cloning T5 shared embeddings to fix Safetensors saving issue."
                )
                self.t5.encoder.embed_tokens.weight.data = (
                    self.t5.encoder.embed_tokens.weight.data.clone()
                )

        # Freeze T5 - only attention pooling is trainable
        for param in self.t5.parameters():
            param.requires_grad = False

        self.hidden_size = self.t5.config.d_model  # 1024 for flan-t5-large

        # Trainable attention pooling components
        # query: Learnable query vector for cross-attention
        self.query = nn.Parameter(
            torch.randn(1, 1, self.hidden_size) * 0.02
        )  # Scaled init

        # attn: Cross-attention to extract style from T5 hidden states
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=8, batch_first=True
        )

        # Proper initialization for stability
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        if self.attn.in_proj_bias is not None:
            nn.init.zeros_(self.attn.in_proj_bias)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [Batch, Seq_Len] - Tokenized instruction text
            attention_mask: [Batch, Seq_Len] - 1 for real tokens, 0 for padding

        Returns:
            style_emb: [Batch, hidden_size=1024] - Style embedding vector
        """
        # CRITICAL: Disable autocast for entire forward pass to avoid FP16 NaN issues
        with torch.amp.autocast("cuda", enabled=False):
            # Get T5 encoder output (frozen, no gradient, FP32)
            with torch.no_grad():
                outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask)
                encoder_hidden_states = outputs.last_hidden_state.detach().float()

            # Attention Pooling (trainable, FP32)
            batch_size = input_ids.shape[0]
            query = self.query.float().expand(batch_size, -1, -1)

            # Prepare key_padding_mask for PyTorch MultiheadAttention
            key_padding_mask = (attention_mask == 0).to(
                device=encoder_hidden_states.device
            )

            # Handle edge case: all tokens masked (would cause NaN)
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask[all_masked] = False

            # Cross-attention: Query attends to T5 hidden states
            style_emb, _ = self.attn(
                query,
                encoder_hidden_states,
                encoder_hidden_states,
                key_padding_mask=key_padding_mask,
            )

            # Squeeze sequence dimension: [B, 1, D] → [B, D]
            style_emb = style_emb.squeeze(1)

        return style_emb


class InstructionEncoderBert(nn.Module):
    """
    Encodes instruction text into a style embedding vector.

    Output: style_emb [Batch, hidden_size=1024] - Raw pooled representation.

    Downstream tasks should apply their own projection heads:
    - InstructionMapper: cond_proj (1024 → 512) for Flow Matching
    - AdaRMSNormAdapter: adapter (1024 → hidden_size) for T3 modulation
    """

    def __init__(self, model_name="FacebookAI/roberta-large"):
        super().__init__()
        print(f"Loading T5 Encoder: {model_name}...")
        self.t5 = T5EncoderModel.from_pretrained(model_name)

        # Fix Safetensors tied weights issue
        if hasattr(self.t5, "shared") and hasattr(self.t5.encoder, "embed_tokens"):
            if (
                self.t5.shared.weight.data_ptr()
                == self.t5.encoder.embed_tokens.weight.data_ptr()
            ):
                print(
                    "Info: Cloning T5 shared embeddings to fix Safetensors saving issue."
                )
                self.t5.encoder.embed_tokens.weight.data = (
                    self.t5.encoder.embed_tokens.weight.data.clone()
                )

        # Freeze T5 - only attention pooling is trainable
        for param in self.t5.parameters():
            param.requires_grad = False

        self.hidden_size = self.t5.config.d_model  # 1024 for flan-t5-large

        # Trainable attention pooling components
        # query: Learnable query vector for cross-attention
        self.query = nn.Parameter(
            torch.randn(1, 1, self.hidden_size) * 0.02
        )  # Scaled init

        # attn: Cross-attention to extract style from T5 hidden states
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=8, batch_first=True
        )

        # Proper initialization for stability
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        if self.attn.in_proj_bias is not None:
            nn.init.zeros_(self.attn.in_proj_bias)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [Batch, Seq_Len] - Tokenized instruction text
            attention_mask: [Batch, Seq_Len] - 1 for real tokens, 0 for padding

        Returns:
            style_emb: [Batch, hidden_size=1024] - Style embedding vector
        """
        # CRITICAL: Disable autocast for entire forward pass to avoid FP16 NaN issues
        with torch.amp.autocast("cuda", enabled=False):
            # Get T5 encoder output (frozen, no gradient, FP32)
            with torch.no_grad():
                outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask)
                encoder_hidden_states = outputs.last_hidden_state.detach().float()

            # Attention Pooling (trainable, FP32)
            batch_size = input_ids.shape[0]
            query = self.query.float().expand(batch_size, -1, -1)

            # Prepare key_padding_mask for PyTorch MultiheadAttention
            key_padding_mask = (attention_mask == 0).to(
                device=encoder_hidden_states.device
            )

            # Handle edge case: all tokens masked (would cause NaN)
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask[all_masked] = False

            # Cross-attention: Query attends to T5 hidden states
            style_emb, _ = self.attn(
                query,
                encoder_hidden_states,
                encoder_hidden_states,
                key_padding_mask=key_padding_mask,
            )

            # Squeeze sequence dimension: [B, 1, D] → [B, D]
            style_emb = style_emb.squeeze(1)

        return style_emb


if __name__ == "__main__":
    MODEL_NAME = "google/flan-t5-large"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running test on {DEVICE}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = InstructionEncoderT5(model_name=MODEL_NAME).to(DEVICE)

    instruction_prompts = [
        "A young adult female delivers her speech with a slightly expressive tone.",
        "Speak slowly and sadly.",
    ]

    # Tokenize
    inputs = tokenizer(
        instruction_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    print("\nInput Shapes:")
    print(f"Input IDs: {inputs.input_ids.shape}")
    print(f"Attn Mask: {inputs.attention_mask.shape}")

    # Forward
    try:
        style_emb = model(inputs.input_ids, inputs.attention_mask)

        print("\nOutput Check:")
        print(f"Style Emb Shape: {style_emb.shape}")

        expected_dim = model.hidden_size  # 1024
        if style_emb.shape == (2, expected_dim):
            print(f"✅ TEST PASSED: Output dimension is correct ({expected_dim}).")
        else:
            print("❌ TEST FAILED: Wrong output dimension.")

        if torch.isnan(style_emb).any():
            print("❌ TEST FAILED: Output contains NaN.")
        else:
            print("✅ TEST PASSED: No NaN values.")

        # Check trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"\nTrainable params: {trainable:,}")
        print(f"Frozen params: {frozen:,}")

    except Exception as e:
        print(f"\n❌ ERROR during forward pass: {e}")
        import traceback

        traceback.print_exc()
