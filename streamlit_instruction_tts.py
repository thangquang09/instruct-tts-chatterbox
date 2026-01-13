#!/usr/bin/env python3
"""
Streamlit Demo for InstructionChatterBox - Instruction-only TTS
With integrated WER and UTMOS evaluation

Usage:
    streamlit run streamlit_instruction_tts.py --server.port 7860
"""

import random
import numpy as np
import torch
import torchaudio.transforms as T
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox.tts import InstructionChatterBox


# =================== Configuration ===================
T3_CKPT_DIR = "checkpoints/t3_instruct_ddp_2query"
MAPPER_CKPT = "checkpoints/mapper_flow/best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


@st.cache_resource
def load_tts_model():
    """Load TTS model once and cache it."""
    model = InstructionChatterBox.from_local(
        t3_ckpt_dir=T3_CKPT_DIR,
        mapper_ckpt_path=MAPPER_CKPT,
        device=DEVICE,
    )
    return model


@st.cache_resource
def load_whisper_model():
    """Load Whisper model for ASR/WER calculation."""
    import whisper

    model = whisper.load_model("large-v3-turbo", device=DEVICE)
    return model


@st.cache_resource
def load_utmos_model():
    """Load UTMOS model for MOS prediction."""
    import utmosv2

    model = utmosv2.create_model(pretrained=True, device=DEVICE)
    return model


def calculate_wer_score(
    audio_tensor: torch.Tensor, sample_rate: int, ground_truth: str, whisper_model
) -> dict:
    """Calculate WER and CER using Whisper ASR."""
    from whisper.normalizers import EnglishTextNormalizer
    from jiwer import wer as calc_wer, cer as calc_cer

    normalizer = EnglishTextNormalizer()

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = T.Resample(sample_rate, 16000).to(DEVICE)
        audio_tensor = resampler(audio_tensor.to(DEVICE))

    # Convert to numpy for Whisper
    audio_np = audio_tensor.squeeze().cpu().numpy()

    # Transcribe
    result = whisper_model.transcribe(audio_np, fp16=(DEVICE == "cuda"), language="en")
    pred_text = normalizer(result["text"].strip())
    gt_text = normalizer(ground_truth)

    # Calculate WER and CER
    wer_score = round(calc_wer(gt_text, pred_text), 4)
    cer_score = round(calc_cer(gt_text, pred_text), 4)

    return {
        "predicted_text": pred_text,
        "ground_truth": gt_text,
        "wer": wer_score,
        "cer": cer_score,
    }


def calculate_utmos_score(
    audio_tensor: torch.Tensor, sample_rate: int, utmos_model
) -> float:
    """Calculate UTMOS score for audio quality."""
    # Ensure correct shape (batch_size, sequence_length) or (sequence_length,)
    if audio_tensor.dim() == 2:
        # [channels, samples] -> [samples] (take first channel or average)
        audio_tensor = audio_tensor.mean(dim=0)

    # Predict MOS directly from tensor
    with torch.inference_mode():
        mos_result = utmos_model.predict(
            data=audio_tensor.cpu().numpy(), sr=sample_rate
        )

    # Extract score
    if isinstance(mos_result, (torch.Tensor, np.ndarray)):
        mos_score = float(np.array(mos_result).squeeze())
    else:
        mos_score = float(mos_result)

    return round(mos_score, 4)


def main():
    st.set_page_config(
        page_title="Instruction TTS with Evaluation",
        page_icon="🎤",
        layout="wide",
    )

    st.title("🎤 Instruction-based Text-to-Speech")
    st.markdown("Generate speech with **WER** and **UTMOS** evaluation built-in!")
    st.markdown("---")

    # Load models
    with st.spinner("Loading TTS model..."):
        tts_model = load_tts_model()

    # Lazy load evaluation models (only when needed)
    eval_models_loaded = False

    # Two columns layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Input")

        # Text input
        text = st.text_area(
            "Text to Synthesize",
            value="Hello! Welcome to our text to speech demonstration. I hope you're having a wonderful day.",
            height=100,
        )

        # Instruction input
        instruction = st.text_area(
            "Voice Style Instruction",
            value="Speak in a warm, friendly female voice with a cheerful tone.",
            height=80,
        )

        # Example instructions
        st.markdown("**Example Instructions:**")
        examples = [
            "Speak in a warm, friendly female voice with a cheerful tone.",
            "Use a deep male voice, speaking slowly and calmly.",
            "Speak with excitement and enthusiasm.",
            "Use a professional business tone.",
            "Speak softly and gently, like telling a bedtime story.",
            "Use a confident and authoritative voice.",
        ]
        selected_example = st.selectbox(
            "Or select an example:",
            ["(Custom)"] + examples,
        )
        if selected_example != "(Custom)":
            instruction = selected_example

        st.markdown("---")
        st.subheader("⚙️ Parameters")

        # Parameters
        exaggeration = st.slider(
            "Exaggeration (emotion intensity)",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.05,
            help="0.5 = neutral, higher = more expressive",
        )

        cfg_weight = st.slider(
            "CFG Weight (text adherence)",
            min_value=0.0,
            max_value=1.5,
            value=0.5,
            step=0.05,
            help="Higher = more accurate text but less natural",
        )

        with st.expander("Advanced Options"):
            temperature = st.slider(
                "Temperature (randomness)",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.05,
            )
            seed = st.number_input(
                "Seed (0 = random)",
                min_value=0,
                max_value=999999,
                value=0,
            )

        # Evaluation checkbox
        st.markdown("---")
        st.subheader("📊 Evaluation Options")
        run_wer = st.checkbox("Calculate WER (Word Error Rate)", value=True)
        run_utmos = st.checkbox("Calculate UTMOS (Audio Quality)", value=True)

        # Generate button
        generate_btn = st.button(
            "🎵 Generate Speech", type="primary", use_container_width=True
        )

    with col2:
        st.subheader("🔊 Output")

        if generate_btn:
            if not text.strip():
                st.error("Please enter some text to synthesize!")
            elif not instruction.strip():
                st.error("Please enter a voice style instruction!")
            else:
                # Set seed if specified
                if seed != 0:
                    set_seed(int(seed))

                # Generate speech
                with st.spinner("🎵 Generating speech..."):
                    try:
                        wav = tts_model.generate(
                            text=text,
                            instruction=instruction,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                            temperature=temperature,
                        )

                        audio_data = wav.squeeze(0).numpy()
                        sample_rate = tts_model.sr

                        # Display audio
                        st.audio(audio_data, sample_rate=sample_rate)
                        st.success("✅ Generation complete!")

                    except Exception as e:
                        st.error(f"Generation Error: {str(e)}")
                        import traceback

                        st.code(traceback.format_exc())
                        return

                # Evaluation results
                st.markdown("---")
                st.subheader("📊 Evaluation Results")

                eval_col1, eval_col2 = st.columns(2)

                # WER Calculation
                if run_wer:
                    with eval_col1:
                        with st.spinner("🔤 Calculating WER..."):
                            try:
                                whisper_model = load_whisper_model()
                                wer_results = calculate_wer_score(
                                    wav, sample_rate, text, whisper_model
                                )

                                # Display WER metrics
                                st.metric(
                                    "WER (Word Error Rate)", f"{wer_results['wer']:.2%}"
                                )
                                st.metric(
                                    "CER (Character Error Rate)",
                                    f"{wer_results['cer']:.2%}",
                                )

                                with st.expander("ASR Details"):
                                    st.markdown(
                                        f"**Ground Truth:** {wer_results['ground_truth']}"
                                    )
                                    st.markdown(
                                        f"**Predicted:** {wer_results['predicted_text']}"
                                    )

                            except Exception as e:
                                st.error(f"WER Error: {str(e)}")

                # UTMOS Calculation
                if run_utmos:
                    with eval_col2:
                        with st.spinner("🎧 Calculating UTMOS..."):
                            try:
                                utmos_model = load_utmos_model()
                                mos_score = calculate_utmos_score(
                                    wav, sample_rate, utmos_model
                                )

                                # Display MOS with color coding
                                if mos_score >= 4.0:
                                    color = "🟢"
                                    quality = "Excellent"
                                elif mos_score >= 3.5:
                                    color = "🟡"
                                    quality = "Good"
                                elif mos_score >= 3.0:
                                    color = "🟠"
                                    quality = "Fair"
                                else:
                                    color = "🔴"
                                    quality = "Poor"

                                st.metric("UTMOS Score", f"{mos_score:.3f} / 5.0")
                                st.markdown(f"{color} **Quality:** {quality}")

                            except Exception as e:
                                st.error(f"UTMOS Error: {str(e)}")

                # Generation details
                with st.expander("Generation Details"):
                    st.markdown(f"""
                    - **Text**: {text[:100]}{"..." if len(text) > 100 else ""}
                    - **Instruction**: {instruction}
                    - **Exaggeration**: {exaggeration}
                    - **CFG Weight**: {cfg_weight}
                    - **Temperature**: {temperature}
                    - **Seed**: {seed if seed != 0 else "random"}
                    """)
        else:
            st.info("👆 Click 'Generate Speech' to create audio")

    # Footer
    st.markdown("---")
    st.markdown("""
    ### Parameter Guide
    | Parameter | Description | Recommended |
    |-----------|-------------|-------------|
    | **Exaggeration** | Emotion intensity | 0.3 - 0.7 |
    | **CFG Weight** | Text accuracy (higher = accurate but robotic) | 0.3 - 0.7 |
    | **Temperature** | Randomness | 0.7 - 1.0 |
    
    ### Evaluation Metrics
    | Metric | Description | Good Range |
    |--------|-------------|------------|
    | **WER** | Word Error Rate (lower = better) | < 10% |
    | **CER** | Character Error Rate (lower = better) | < 5% |
    | **UTMOS** | Mean Opinion Score (higher = better) | > 3.5 |
    """)


if __name__ == "__main__":
    main()
