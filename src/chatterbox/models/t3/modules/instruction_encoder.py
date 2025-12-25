import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer

class InstructionEncoder(nn.Module):
    def __init__(self, model_name="google/flan-t5-large", output_dim=1024):
        super().__init__()
        print(f"Loading T5 Encoder: {model_name}...")
        self.t5 = T5EncoderModel.from_pretrained(model_name)
        
        if hasattr(self.t5, "shared") and hasattr(self.t5.encoder, "embed_tokens"):
             if self.t5.shared.weight.data_ptr() == self.t5.encoder.embed_tokens.weight.data_ptr():
                print("Info: Cloning T5 shared embeddings to fix Safetensors saving issue.")
                self.t5.encoder.embed_tokens.weight.data = self.t5.encoder.embed_tokens.weight.data.clone()
        
        for param in self.t5.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.t5.config.d_model
        
        # Shape: [1, 1, Dim] -> Batch dimension sẽ được expand trong forward
        self.query = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        
        # Multihead Attention
        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8, batch_first=True)
        
        # Project, ChatterBox dùng 1024
        self.proj = nn.Linear(self.hidden_size, output_dim)
        
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [Batch, Seq_Len]
        attention_mask: [Batch, Seq_Len] (1 cho token thật, 0 cho padding)
        """
        
        with torch.no_grad():
            outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask)
            # last_hidden_state: [Batch, Seq_Len, 1024]
            encoder_hidden_states = outputs.last_hidden_state
            
        # Attention Pooling
        batch_size = input_ids.shape[0]
        # Expand Query cho khớp batch size: [Batch, 1, Dim]
        query = self.query.expand(batch_size, -1, -1)
        
        # Xử lý Mask cho PyTorch MultiheadAttention
        key_padding_mask = (attention_mask == 0) 
        
        # Attn(Q, K, V)
        # style_emb: [Batch, 1, Dim]
        style_emb, _ = self.attn(
            query, 
            encoder_hidden_states, 
            encoder_hidden_states, 
            key_padding_mask=key_padding_mask
        )
        
        # 3. Projection & Output
        # Squeeze(1) để mất chiều sequence -> [Batch, Output_Dim]
        style_vector = self.proj(style_emb).squeeze(1) 
        
        return style_vector

if __name__ == "__main__":
    MODEL_NAME = "google/flan-t5-large"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running test on {DEVICE}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = InstructionEncoder(model_name=MODEL_NAME, output_dim=1024).to(DEVICE)
    
    instruction_prompts = [
        "A young adult female delivers her speech with a slightly expressive and animated tone, her words flowing with a slight urgency. Her voice carries a moderate pitch, harmonious yet passionate, and she speaks at a slightly fast pace, imbuing the conversation with a sense of urgency and enthusiasm.",
        "Speak slowly and sadly." 
    ]
    
    # Tokenize (Padding & Truncation)
    inputs = tokenizer(
        instruction_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    ).to(DEVICE)
    
    print("\nInput Shapes:")
    print(f"Input IDs: {inputs.input_ids.shape}")       # [2, Seq_Len]
    print(f"Attn Mask: {inputs.attention_mask.shape}")  # [2, Seq_Len]
    
    # 4. Chạy Forward
    try:
        style_vector = model(inputs.input_ids, inputs.attention_mask)
        
        print("\nOutput Check:")
        print(f"Style Vector Shape: {style_vector.shape}")
        
        if style_vector.shape == (2, 1024):
            print("✅ TEST PASSED: Output dimension is correct.")
        else:
            print("❌ TEST FAILED: Wrong output dimension.")
            
        if torch.isnan(style_vector).any():
            print("❌ TEST FAILED: Output contains NaN.")
        else:
            print("✅ TEST PASSED: No NaN values.")
            
    except Exception as e:
        print(f"\n❌ ERROR during forward pass: {e}")