"""
Script để extract các tags từ groundtruth audio trong data/final_data_test.txt

Unique key: audio_path (cột đầu tiên)
Tags cần lưu (với tiền tố gt_):
- gt_pitch
- gt_speech_monotony
- gt_speaking_rate
- gt_age
- gt_gender
"""

import os
import json
import bisect
import librosa
import pandas as pd
from tqdm import tqdm
import torch

from base_eval import pitch_apply, speed_apply, age_gender_apply


SPEAKER_RATE_BINS = [
    "very slowly",
    "slowly",
    "slightly slowly",
    "moderate speed",
    "slightly fast",
    "fast",
    "very fast",
]

UTTERANCE_LEVEL_STD = [
    "very monotone",
    "monotone",
    "slightly expressive and animated",
    "expressive and animated",
    "very expressive and animated",
]

SPEAKER_LEVEL_PITCH_BINS = [
    "very low-pitch",
    "low-pitch",
    "slightly low-pitch",
    "moderate pitch",
    "slightly high-pitch",
    "high-pitch",
    "very high-pitch",
]

BASE_DIR = "/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/benchmark"

# Load text bins từ file JSON
with open(os.path.join(BASE_DIR, "bin.json")) as json_file:
    TEXT_BINS_DICT = json.load(json_file)


def get_pitch_label(pitch_mean, gender):
    """Map pitch_mean thành nhãn pitch dựa trên gender."""
    if gender == "male":
        index = bisect.bisect_right(TEXT_BINS_DICT["pitch_bins_male"], pitch_mean) - 1
    else:
        index = bisect.bisect_right(TEXT_BINS_DICT["pitch_bins_female"], pitch_mean) - 1
    index = max(0, min(index, len(SPEAKER_LEVEL_PITCH_BINS) - 1))
    return SPEAKER_LEVEL_PITCH_BINS[index]


def get_monotony_label(pitch_std):
    """Map pitch_std thành nhãn monotony."""
    index = bisect.bisect_right(TEXT_BINS_DICT["speech_monotony"], pitch_std) - 1
    index = max(0, min(index, len(UTTERANCE_LEVEL_STD) - 1))
    return UTTERANCE_LEVEL_STD[index]


def get_speed_label(speech_duration):
    """Map speech_duration thành nhãn speed."""
    index = bisect.bisect_right(TEXT_BINS_DICT["speaking_rate"], speech_duration) - 1
    index = max(0, min(index, len(SPEAKER_RATE_BINS) - 1))
    return SPEAKER_RATE_BINS[index]


def process_audio(audio_path):
    """Xử lý một file audio và trả về tất cả các đặc tính."""
    waveform, _ = librosa.load(audio_path, sr=16000)
    with torch.inference_mode():
        # Tính các đặc tính
        age, gender = age_gender_apply(waveform)
        pitch_mean, pitch_std = pitch_apply(waveform)
        speech_duration = speed_apply(waveform)

        # Map thành nhãn
        pitch_label = get_pitch_label(pitch_mean, gender)
        monotony_label = get_monotony_label(pitch_std)
        speed_label = get_speed_label(speech_duration)

    return {
        "pitch": pitch_label,
        "speech_monotony": monotony_label,
        "speaking_rate": speed_label,
        "age": age,
        "gender": gender,
    }


def main():
    # Đọc file data test
    data_file = "/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/data/final_data_test.txt"
    result_folder = os.path.join(BASE_DIR, "results")
    os.makedirs(result_folder, exist_ok=True)
    save_path = os.path.join(result_folder, "gt_tags.csv")

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

    print(f"--- Đang thực hiện extract tags cho {len(df_to_process)} file còn lại ---")

    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
        audio_path = row["audio_path"]

        if os.path.exists(audio_path):
            try:
                # Xử lý audio ground truth
                gt_result = process_audio(audio_path)

                # Tạo dictionary kết quả với 5 cột tags + audio_path làm unique key
                res_dict = {
                    "audio_path": audio_path,
                    "gt_pitch": gt_result["pitch"],
                    "gt_speech_monotony": gt_result["speech_monotony"],
                    "gt_speaking_rate": gt_result["speaking_rate"],
                    "gt_age": gt_result["age"],
                    "gt_gender": gt_result["gender"],
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
    print("\n--- Phân phối các Tags ---")
    for col in [
        "gt_pitch",
        "gt_speech_monotony",
        "gt_speaking_rate",
        "gt_age",
        "gt_gender",
    ]:
        if col in final_df.columns:
            print(f"\n{col}:")
            print(final_df[col].value_counts().to_string())


if __name__ == "__main__":
    main()
