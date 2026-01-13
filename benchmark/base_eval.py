from pathlib import Path

import audeer
import audonnx
import numpy as np
import penn
import torch
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
from huggingface_hub import hf_hub_download
from pyannote.audio import Model
from pyannote.audio.core.model import Introspection
from pyannote.audio.core.task import Specifications, Problem, Resolution

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_ID = 0 if torch.cuda.is_available() else None  # Cho penn pitch detection
print(f"--- Sử dụng device: {DEVICE} ---")

print("--- Đang nạp Brouhaha model (cho speed detection) ---")
with torch.serialization.safe_globals(
    [
        torch.torch_version.TorchVersion,
        Introspection,
        Specifications,
        Problem,
        Resolution,
    ]
):
    BROUHAHA_MODEL = Model.from_pretrained(
        Path(hf_hub_download(repo_id="ylacombe/brouhaha-best", filename="best.ckpt")),
        strict=False,
    )
BROUHAHA_MODEL.to(DEVICE)
BROUHAHA_PIPELINE = RegressiveActivityDetectionPipeline(
    segmentation=BROUHAHA_MODEL, batch_size=1
)
BROUHAHA_PIPELINE.to(torch.device(DEVICE))
print("--- Đã nạp xong Brouhaha model ---")

print("--- Đang nạp Age-Gender model ---")
AGE_GENDER_URL = "https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip"
AGE_GENDER_CACHE_ROOT = audeer.mkdir("cache")
AGE_GENDER_MODEL_ROOT = audeer.mkdir("model")
archive_path = audeer.download_url(AGE_GENDER_URL, AGE_GENDER_CACHE_ROOT, verbose=True)
audeer.extract_archive(archive_path, AGE_GENDER_MODEL_ROOT)

AGE_GENDER_MODEL = audonnx.load(AGE_GENDER_MODEL_ROOT, device="cuda")
print("--- Đã nạp xong Age-Gender model ---")

# ==============================================================================
# CONSTANTS
# ==============================================================================
AGE_LABELS = ["child", "teenager", "young adult", "middle-aged adult", "elderly"]
GENDER_LABELS = ["female", "male"]


def pitch_apply(waveform):
    """Tính pitch mean và std từ waveform."""
    hopsize = 0.01
    fmin = 30.0
    fmax = 1000.0
    checkpoint = None
    center = "half-hop"
    interp_unvoiced_at = 0.065
    sampling_rate = 16000
    penn_batch_size = 4096
    waveform = torch.Tensor(waveform).unsqueeze(0)
    pitch, periodicity = penn.from_audio(
        waveform.float(),
        sampling_rate,
        hopsize=hopsize,
        fmin=fmin,
        fmax=fmax,
        checkpoint=checkpoint,
        batch_size=penn_batch_size,
        center=center,
        interp_unvoiced_at=interp_unvoiced_at,
        gpu=GPU_ID,  # Sử dụng GPU nếu có
    )

    pitch_mean = pitch.mean().cpu().numpy()
    pitch_std = pitch.std().cpu().numpy()

    return pitch_mean, pitch_std


def speed_apply(waveform):
    """Tính speech duration từ waveform sử dụng Brouhaha pipeline (global model)."""
    sampling_rate = 16000
    waveform_tensor = torch.Tensor(waveform).unsqueeze(0)

    device = BROUHAHA_PIPELINE._models["segmentation"].device

    res = BROUHAHA_PIPELINE(
        {"sample_rate": sampling_rate, "waveform": waveform_tensor.to(device).float()}
    )

    speech_duration = sum(map(lambda x: x[0].duration, res["annotation"].itertracks()))

    return speech_duration


def age_gender_apply(waveform):
    """Nhận diện age và gender từ waveform sử dụng global model."""
    sampling_rate = 16000

    result = AGE_GENDER_MODEL(waveform, sampling_rate)

    # Process age
    age_value = result["logits_age"].squeeze() * 100.0
    if age_value <= 12:
        age_label = "child"
    elif age_value <= 19:
        age_label = "teenager"
    elif age_value <= 39:
        age_label = "young adult"
    elif age_value <= 64:
        age_label = "middle-aged adult"
    else:
        age_label = "elderly"

    # Process gender
    gender_logits = result["logits_gender"].squeeze()
    gender_logits = gender_logits[:2]  # Remove child
    gender_idx = np.argmax(gender_logits)

    return age_label, GENDER_LABELS[gender_idx]
