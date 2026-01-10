import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from tqdm import tqdm
from pathlib import Path

BASE_DIR = "/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/benchmark"
VOX_DIR = os.path.join(BASE_DIR, "vox-profile-release")

if VOX_DIR not in sys.path:
    sys.path.append(VOX_DIR)

try:
    from src.model.emotion.whisper_emotion import WhisperWrapper
except ImportError as e:
    print(f"Lỗi Import: {e}. Hãy kiểm tra lại thư mục: {VOX_DIR}")
    sys.exit(1)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("--- Đang nạp model Whisper Emotion từ Hugging Face ---")
model = WhisperWrapper.from_pretrained(
    "tiantiaf/whisper-large-v3-msp-podcast-emotion"
).to(device)
model.eval()


def load_audio_emotion(path, target_sr=16000, max_duration=15):
    waveform, sample_rate = torchaudio.load(path, backend="soundfile")
    waveform = waveform.to(device)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != target_sr:
        resampler = T.Resample(sample_rate, target_sr).to(device)
        waveform = resampler(waveform)
    max_length = int(target_sr * max_duration)
    if waveform.shape[1] > max_length:
        waveform = waveform[:, :max_length]
    return waveform.float()


def get_pred_emotion(data):
    logits, embedding, _, _, _, _ = model(data, return_feature=True)
    emotion_prob = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(emotion_prob).detach().cpu().item()
    return emotion_list[pred_idx]


def main():
    metadata_path = os.path.join(BASE_DIR, "test_metadata.csv")
    test_dir = os.path.join(BASE_DIR, "output_wavs/2query_t3_freeze")
    result_folder = os.path.join(BASE_DIR, "results")
    os.makedirs(result_folder, exist_ok=True)
    save_path = os.path.join(result_folder, "emotion_results.csv")

    full_df = pd.read_csv(metadata_path, dtype={"id": str})

    processed_ids = set()
    if os.path.exists(save_path):
        try:
            existing_df = pd.read_csv(save_path, sep="|", dtype={"id": str})
            processed_ids = set(existing_df["id"].unique())
            print(
                f"--- Đã tìm thấy file kết quả cũ. Đã xử lý: {len(processed_ids)} file. ---"
            )
        except Exception as e:
            print(
                f"Lưu ý: File kết quả cũ trống hoặc lỗi, sẽ bắt đầu mới. Chi tiết: {e}"
            )

    df_to_process = full_df[~full_df["id"].isin(processed_ids)]

    if len(df_to_process) == 0:
        print("--- Tất cả các file đã được xử lý xong! ---")
        return

    print(f"--- Đang thực hiện nhận diện cho {len(df_to_process)} file còn lại ---")

    with torch.inference_mode():
        for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
            audio_name = row["id"]
            gt_audio_path = row["audio_path"]
            audio_path = os.path.join(test_dir, f"{audio_name}.wav")

            if os.path.exists(audio_path) and os.path.exists(gt_audio_path):
                try:
                    # Xử lý Inference
                    data = load_audio_emotion(audio_path)
                    gt_data = load_audio_emotion(gt_audio_path)

                    pred_emotion = get_pred_emotion(data)
                    gt_emotion = get_pred_emotion(gt_data)

                    # Tạo dictionary kết quả
                    res_dict = row.to_dict()
                    res_dict.update(
                        {
                            "pred_emotion": pred_emotion,
                            "gt_emotion": gt_emotion,
                            "emotion_acc": 1 if pred_emotion == gt_emotion else 0,
                        }
                    )

                    # Ghi trực tiếp vào file CSV (Append mode)
                    res_df = pd.DataFrame([res_dict])
                    # Viết header nếu file chưa tồn tại, ngược lại thì không viết header
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
                # Nếu thiếu file, có thể log lại hoặc bỏ qua
                pass

    # 4. Thống kê sau khi hoàn thành
    print("Quá trình hoàn tất!")
    final_df = pd.read_csv(save_path, sep="|", dtype={"id": str})
    print(f"Tổng số file hiện có trong kết quả: {len(final_df)}")
    if "emotion_acc" in final_df.columns:
        overall_acc = final_df["emotion_acc"].mean() * 100
        print(f"Độ nhất quán cảm xúc tổng thể (Consistency): {overall_acc:.2f}%")


if __name__ == "__main__":
    main()
