import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = "/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/benchmark"
VOX_DIR = os.path.join(BASE_DIR, "vox-profile-release")

if VOX_DIR not in sys.path:
    sys.path.append(VOX_DIR)

try:
    # Import WavLMWrapper cho Accent
    from src.model.accent.wavlm_accent import WavLMWrapper
except ImportError as e:
    print(f"Lỗi Import: {e}. Hãy kiểm tra lại thư mục: {VOX_DIR}")
    sys.exit(1)

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

# 1. Load model Accent từ Hugging Face
print("--- Đang nạp model WavLM Accent từ Hugging Face ---")
wavlm_model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-narrow-accent").to(
    device
)
wavlm_model.eval()


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


def get_pred_accent(data):
    # Model Accent trả về logits và embeddings
    logits, embeddings = wavlm_model(data, return_feature=True)

    # Tính xác suất
    prob = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(prob).detach().cpu().item()
    return english_accent_list[pred_idx]


def main():
    # 2. Đường dẫn dữ liệu
    metadata_path = os.path.join(BASE_DIR, "test_metadata.csv")
    test_dir = os.path.join(BASE_DIR, "output_wavs/2query_t3_freeze")
    result_folder = os.path.join(BASE_DIR, "results")
    os.makedirs(result_folder, exist_ok=True)
    save_path = os.path.join(result_folder, "accent_results.csv")

    # Đọc metadata gốc (ID là string)
    full_df = pd.read_csv(metadata_path, dtype={"id": str})
    # full_df = full_df.head(5)  # Mock test

    # 3. LOGIC RESUMABLE
    processed_ids = set()
    if os.path.exists(save_path):
        try:
            existing_df = pd.read_csv(save_path, sep="|", dtype={"id": str})
            processed_ids = set(existing_df["id"].unique())
            print(
                f"--- Đã tìm thấy file kết quả cũ. Đã xử lý: {len(processed_ids)} file. ---"
            )
        except Exception as e:
            print(f"Lưu ý: File kết quả cũ trống hoặc lỗi, bắt đầu mới. Chi tiết: {e}")

    df_to_process = full_df[~full_df["id"].isin(processed_ids)]

    if len(df_to_process) == 0:
        print("--- Tất cả các file Accent đã được xử lý xong! ---")
        return

    print(
        f"--- Đang thực hiện nhận diện Accent cho {len(df_to_process)} file còn lại ---"
    )

    with torch.inference_mode():
        for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
            audio_name = row["id"]
            gt_audio_path = row["audio_path"]
            audio_path = os.path.join(test_dir, f"{audio_name}.wav")

            if os.path.exists(audio_path) and os.path.exists(gt_audio_path):
                try:
                    # Load audio
                    data = load_audio(audio_path)
                    gt_data = load_audio(gt_audio_path)

                    # Inference Accent
                    pred_accent = get_pred_accent(data)
                    gt_accent = get_pred_accent(gt_data)

                    # Tạo dictionary kết quả
                    res_dict = row.to_dict()
                    res_dict.update(
                        {
                            "pred_accent": pred_accent,
                            "gt_accent": gt_accent,
                            "accent_acc": 1 if pred_accent == gt_accent else 0,
                        }
                    )

                    # Ghi trực tiếp vào CSV (Append mode)
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
                    print(f"\nLỗi xử lý file {audio_name}: {e}")
            else:
                # Bỏ qua nếu thiếu file
                pass

    # 4. Thống kê sau khi hoàn thành
    print("\n--- Quá trình hoàn tất! ---")
    final_df = pd.read_csv(save_path, sep="|", dtype={"id": str})
    print(f"Tổng số file hiện có trong kết quả: {len(final_df)}")
    if "accent_acc" in final_df.columns:
        overall_acc = final_df["accent_acc"].mean() * 100
        print(f"Độ nhất quán Accent tổng thể (Consistency): {overall_acc:.2f}%")


if __name__ == "__main__":
    main()
