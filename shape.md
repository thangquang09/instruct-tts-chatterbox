# ChatterBox Architecture Shapes

Recorded during inference with `example_tts.py` (Sequence Lengths may vary depending on input audio).

## T3 Model (Text-To-Token)

| Component | Variable | Shape | Notes |
|-----------|----------|-------|-------|
| **Speaker Embedding** | `cond.speaker_emb` | `[1, 256]` | Output of Voice Encoder. |
| **Perceiver Input** | `cond.cond_prompt_speech_emb` | `[1, 150, 1024]` | Embedded S3 tokens for prompting. |
| **Emotion Scalar** | `cond.emotion_adv` | `[1, 1, 1]` | Scalar value (default 0.5). |

## S3Gen Model (Token-To-Wave)

| Component | Variable | Shape | Notes |
|-----------|----------|-------|-------|
| **Reference Waveform** | `ref_wav_16` | `[1, L]` (e.g., `102400`) | 16kHz audio input. |
| **Reference Mels** | `ref_mels_24` | `[1, L_mel, 80]` (e.g., `320`) | 24kHz mel spectrogram features. |
| **Reference Tokens** | `ref_speech_tokens` | `[1, L_tok]` (e.g., `160`) | S3 Tokenizer output. |
| **Speaker X-Vector** | `ref_x_vector` | `[1, 192]` | CAM++ output. |
