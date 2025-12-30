```mermaid
graph TD
    %% Định nghĩa Style
    classDef input fill:#f9f,stroke:#333,stroke-width:2px,color:black;
    classDef process fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:black;
    classDef model fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:black;
    classDef storage fill:#e0e0e0,stroke:#333,stroke-width:1px,stroke-dasharray:5 5,color:black;
    classDef output fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:black;
    classDef new fill:#ffcdd2,stroke:#c62828,stroke-width:3px,color:black;
    classDef adapter fill:#ce93d8,stroke:#7b1fa2,stroke-width:2px,color:black;

    subgraph Inputs ["Dữ liệu Đầu vào"]
        Text["Text Input"]:::input
        RefAudio["Reference Audio / Prompt"]:::input
        InstructText["🆕 Instruction Text<br/>(Style Description)"]:::new
    end

    subgraph Preprocessing ["Tiền xử lý & Conditioning"]
        direction TB
        
        %% Text Path
        Norm["Text Normalization<br/>(punc_norm)"]:::process
        TextTok["EnTokenizer<br/>(Text to Tokens)"]:::process
        TextEmb["Text Embeddings<br/>(nn.Embedding, 704 vocab)"]:::storage
        TextPosEmb["Learned Position Embeddings<br/>(Text)"]:::process
        
        %% Audio Path - Resample
        RefResample16k["Resample to 16kHz"]:::process
        RefResample24k["Resample to 24kHz<br/>(S3GEN_SR)"]:::process
        
        %% Voice Encoder Path (for T3)
        VE["Voice Encoder<br/>(3-layer LSTM + Linear)"]:::model
        SpkEmb["Speaker Embedding<br/>(256-dim, L2-normed)"]:::storage
        SpkDropout["🆕 Speaker Emb Dropout<br/>(20% → zeros during training)"]:::new
        
        %% Prompt Token Path (for T3)
        S3Tok_T3["S3 Tokenizer<br/>(speech_tokenizer_v2_25hz)"]:::model
        PromptToks["Speech Prompt Tokens<br/>(max 150 tokens)"]:::storage
        PromptEmb["Speech Prompt Embeddings"]:::storage
        PromptPosEmb["Learned Position Embeddings<br/>(Speech)"]:::process
        Perceiver["Perceiver Resampler"]:::model
        
        %% Emotion/Exaggeration
        EmotionAdv["Emotion Exaggeration<br/>(scalar → Linear → 1024-dim)"]:::process
        
        %% T3 Conditionals Assembly
        T3CondEnc["T3CondEnc Module"]:::process
        T3Cond["T3 Conditionals<br/>(Speaker + Prompt + Emotion)"]:::storage
    end

    subgraph InstructionEncoding ["🆕 Instruction Encoder (Frozen T5 + Trainable Adapter)"]
        direction TB
        T5Tok["T5 Tokenizer<br/>(google/flan-t5-large)"]:::new
        T5Enc["Flan-T5-Large Encoder<br/>(FROZEN, 1024-dim)"]:::model
        T5Hidden["T5 Hidden States<br/>[Batch, Seq, 1024]"]:::storage
        AttnPool["🆕 Attention Pooling<br/>(Learnable Query + MHA)"]:::adapter
        InstrProj["🆕 Linear Projection<br/>(1024 → 1024)"]:::adapter
        InstrEmb["Instruction Embedding<br/>[Batch, 1024]"]:::new
    end

    subgraph T3_Model ["T3 Model: Instruct-TTS 520M (CustomLlamaModel)"]
        direction TB
        InputPrep["Prepare Input Embeddings<br/>Concat: [Conds | Text+PosEmb | Speech_BOS+PosEmb]"]:::process
        
        subgraph CustomLlama ["🆕 CustomLlamaModel (30 Layers)"]
            direction TB
            
            subgraph DecoderLayer ["CustomLlamaDecoderLayer × 30"]
                direction TB
                
                subgraph PreAttn ["Pre-Attention"]
                    InputAdapter["🆕 AdaRMSNormAdapter<br/>(input_adapter)"]:::adapter
                    OrigInputNorm["Original input_layernorm<br/>(frozen)"]:::model
                end
                
                SelfAttn["Self-Attention<br/>(SDPA + RoPE)"]:::model
                
                subgraph PostAttn ["Post-Attention"]
                    PostAdapter["🆕 AdaRMSNormAdapter<br/>(post_attention_adapter)"]:::adapter
                    OrigPostNorm["Original post_attention_layernorm<br/>(frozen)"]:::model
                end
                
                MLP["MLP (SiLU)"]:::model
            end
            
            subgraph AdaLNDetail ["🆕 AdaRMSNorm Adapter Logic"]
                AdaLNInput["instruction_emb"]:::new
                AdaLNMLP["Linear → SiLU → Linear<br/>(1024 → hidden → 2048)"]:::adapter
                GammaBeta["Split: γ_ada, β_ada<br/>(1024 each)"]:::adapter
                AdaLNFormula["output = normed_x × (1 + γ) + β"]:::process
            end
        end
        
        SpeechHead["Speech Head<br/>(Linear → 8194 vocab)"]:::process
        
        subgraph Sampling ["Sampling với CFG"]
            CFG["Classifier-Free Guidance<br/>(cfg_weight)"]:::process
            SamplingStrat["Sampling Strategy<br/>(Temp, Top-P, Rep. Penalty)"]:::process
        end
        
        AutoReg["Auto-regressive Loop<br/>(KV-Cache)"]:::process
        SpeechTokens["Generated Speech Tokens<br/>(< 6561)"]:::storage
    end

    subgraph S3Gen_Decoder ["S3Gen Decoder (S3Token2Wav) - Unchanged"]
        direction TB
        
        subgraph S3Gen_RefProc ["Reference Processing (24kHz)"]
            S3Tok_Gen["S3 Tokenizer<br/>(Ref → prompt_token)"]:::model
            MelExtract["Mel Spectrogram<br/>(Ref → prompt_feat)"]:::process
            CAMPPlus["CAMPPlus X-Vector<br/>(Speaker Encoder → embedding)"]:::model
        end
        
        subgraph CFM_Decoder ["CFM Decoder (Token → Mel)"]
            UpsampleEnc["UpsampleConformerEncoder<br/>(6 blocks, 512-dim, 8 heads)"]:::model
            CondDecoder["ConditionalDecoder<br/>(4+12 blocks, Flow Matching)"]:::model
            FlowMatch["CausalConditionalCFM<br/>(10 timesteps)"]:::model
        end
        
        subgraph Vocoder ["HiFi-GAN Vocoder (Mel → Wav)"]
            F0Pred["ConvRNNF0Predictor"]:::model
            HiFT["HiFTGenerator<br/>(Upsample: 8×5×3 = 120)"]:::model
        end
        
        Watermark["Perth Implicit Watermarker"]:::process
    end

    OutputAudio(("Generated Waveform<br/>(24kHz)")):::output

    %% ========== Flow Connections ==========
    
    %% Text Path
    Text --> Norm --> TextTok --> TextEmb
    TextEmb --> TextPosEmb
    
    %% Reference Audio Resampling
    RefAudio --> RefResample16k
    RefAudio --> RefResample24k
    
    %% T3 Conditioning Path (16kHz)
    RefResample16k --> VE --> SpkEmb --> SpkDropout
    RefResample16k --> S3Tok_T3 --> PromptToks
    PromptToks --> PromptEmb --> PromptPosEmb --> Perceiver
    
    %% T3Cond Assembly
    SpkDropout --> T3CondEnc
    Perceiver --> T3CondEnc
    EmotionAdv --> T3CondEnc
    T3CondEnc --> T3Cond
    
    %% 🆕 Instruction Encoding Path
    InstructText --> T5Tok --> T5Enc --> T5Hidden
    T5Hidden --> AttnPool --> InstrProj --> InstrEmb
    
    %% T3 Input Preparation
    T3Cond --> InputPrep
    TextPosEmb --> InputPrep
    
    %% 🆕 Instruction injection into CustomLlamaModel
    InstrEmb --> InputAdapter
    InstrEmb --> PostAdapter
    
    %% AdaLN Detail Flow
    InstrEmb --> AdaLNInput --> AdaLNMLP --> GammaBeta --> AdaLNFormula
    OrigInputNorm --> AdaLNFormula
    OrigPostNorm --> AdaLNFormula
    
    %% Decoder Layer Flow
    InputPrep --> InputAdapter --> SelfAttn --> PostAdapter --> MLP
    MLP --> SpeechHead
    
    SpeechHead --> CFG --> SamplingStrat --> AutoReg
    AutoReg -->|"Next Token Embed + PosEmb"| InputAdapter
    AutoReg -->|"Speech Tokens"| SpeechTokens
    
    %% S3Gen Reference Processing (24kHz)
    RefResample24k --> S3Tok_Gen
    RefResample24k --> MelExtract
    RefResample16k --> CAMPPlus
    
    %% S3Gen Forward
    SpeechTokens --> UpsampleEnc
    S3Tok_Gen --> FlowMatch
    MelExtract --> FlowMatch
    CAMPPlus --> FlowMatch
    UpsampleEnc --> CondDecoder --> FlowMatch
    
    %% Vocoder
    FlowMatch --> F0Pred --> HiFT
    HiFT --> Watermark --> OutputAudio
```