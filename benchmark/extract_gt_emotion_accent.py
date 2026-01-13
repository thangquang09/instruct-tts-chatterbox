"""
Script để extract gt_emotion và gt_accent từ groundtruth audio trong data/final_data_test.txt

Unique key: audio_path (cột đầu tiên)
Tags cần lưu:
- gt_emotion
- gt_accent
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from tqdm import tqdm

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = "/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/benchmark"
VOX_DIR = os.path.join(BASE_DIR, "vox-profile-release")

if VOX_DIR not in sys.path:
    sys.path.append(VOX_DIR)

try:
    from src.model.emotion.whisper_emotion import WhisperWrapper
    from src.model.accent.wavlm_accent import WavLMWrapper
except ImportError as e:
    print(f"Lỗi Import: {e}. Hãy kiểm tra lại thư mục: {VOX_DIR}")
    sys.exit(1)

# Danh sách nhãn Emotion
emotion_list = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
    "Other",
]

# Danh sách nhãn Accent chuẩn của Vox (16 nhãn)
english_accent_list = [
    "East Asia",
    "English",
    "Germanic",
    "Irish",
    "North America",
    "Northern Irish",
    "Oceania",
    "Other",
    "Romance",
    "Scottish",
    "Semitic",
    "Slavic",
    "South African",
    "Southeast Asia",
    "South Asia",
    "Welsh",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
print("--- Đang nạp model Whisper Emotion từ Hugging Face ---")
emotion_model = WhisperWrapper.from_pretrained(
    "tiantiaf/whisper-large-v3-msp-podcast-emotion"
).to(device)
emotion_model.eval()

print("--- Đang nạp model WavLM Accent từ Hugging Face ---")
accent_model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-narrow-accent").to(
    device
)
accent_model.eval()


def load_audio(path, target_sr=16000, max_duration=15):
    """
    Chuẩn hóa audio: 16kHz, Mono, tối đa 15 giây.
    """
    waveform, sample_rate = torchaudio.load(path, backend="soundfile")
    waveform = waveform.to(device)

    # Chuyển về Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample về 16kHz
    if sample_rate != target_sr:
        resampler = T.Resample(sample_rate, target_sr).to(device)
        waveform = resampler(waveform)

    # Cắt audio nếu dài hơn 15 giây
    max_length = int(target_sr * max_duration)
    if waveform.shape[1] > max_length:
        waveform = waveform[:, :max_length]

    return waveform.float()


def get_pred_emotion(data):
    """Dự đoán emotion từ audio data."""
    logits, embedding, _, _, _, _ = emotion_model(data, return_feature=True)
    emotion_prob = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(emotion_prob).detach().cpu().item()
    return emotion_list[pred_idx]


def get_pred_accent(data):
    """Dự đoán accent từ audio data."""
    logits, embeddings = accent_model(data, return_feature=True)
    prob = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(prob).detach().cpu().item()
    return english_accent_list[pred_idx]


def main():
    # Đọc file data test
    data_file = "/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/data/final_data_test.txt"
    result_folder = os.path.join(BASE_DIR, "results")
    os.makedirs(result_folder, exist_ok=True)
    save_path = os.path.join(result_folder, "gt_emotion_accent.csv")

    # Đọc file data test (format: audio_path|text|instruction)
    print("Đang đọc file data test...")
    lines = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("|")
                if len(parts) >= 1:
                    lines.append({"audio_path": parts[0]})

    full_df = pd.DataFrame(lines)
    print(f"Tổng số file cần xử lý: {len(full_df)}")

    # Kiểm tra kết quả đã xử lý để resume
    processed_paths = set()
    if os.path.exists(save_path):
        try:
            existing_df = pd.read_csv(save_path, sep="|")
            processed_paths = set(existing_df["audio_path"].unique())
            print(
                f"--- Đã tìm thấy file kết quả cũ. Đã xử lý: {len(processed_paths)} file. ---"
            )
        except Exception as e:
            print(
                f"Lưu ý: File kết quả cũ trống hoặc lỗi, sẽ bắt đầu mới. Chi tiết: {e}"
            )

    # Lọc ra các file chưa xử lý
    df_to_process = full_df[~full_df["audio_path"].isin(processed_paths)]

    if len(df_to_process) == 0:
        print("--- Tất cả các file đã được xử lý xong! ---")
        return

    print(
        f"--- Đang thực hiện extract emotion/accent cho {len(df_to_process)} file còn lại ---"
    )

    with torch.inference_mode():
        for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
            audio_path = row["audio_path"]

            if os.path.exists(audio_path):
                try:
                    # Load audio
                    data = load_audio(audio_path)

                    # Inference Emotion và Accent
                    gt_emotion = get_pred_emotion(data)
                    gt_accent = get_pred_accent(data)

                    # Tạo dictionary kết quả với audio_path làm unique key
                    res_dict = {
                        "audio_path": audio_path,
                        "gt_emotion": gt_emotion,
                        "gt_accent": gt_accent,
                    }

                    # Ghi vào file CSV (Append mode)
                    res_df = pd.DataFrame([res_dict])
                    file_exists = os.path.isfile(save_path)
                    res_df.to_csv(
                        save_path,
                        mode="a",
                        index=False,
                        sep="|",
                        header=not file_exists,
                    )

                except Exception as e:
                    print(f"\nLỗi xử lý file {audio_path}: {e}")
            else:
                print(f"\nKhông tìm thấy file GT: {audio_path}")

    # Thống kê sau khi hoàn thành
    print("\n" + "=" * 60)
    print("Quá trình hoàn tất!")
    print("=" * 60)

    final_df = pd.read_csv(save_path, sep="|")
    print(f"Tổng số file trong kết quả: {len(final_df)}")
    print(f"Kết quả được lưu tại: {save_path}")

    # Thống kê phân phối các tags
    print("\n--- Phân phối Emotion ---")
    if "gt_emotion" in final_df.columns:
        print(final_df["gt_emotion"].value_counts().to_string())

    print("\n--- Phân phối Accent ---")
    if "gt_accent" in final_df.columns:
        print(final_df["gt_accent"].value_counts().to_string())


if __name__ == "__main__":
    main()
