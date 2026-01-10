import os
import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import whisper
from whisper.normalizers import EnglishTextNormalizer
from jiwer import wer as calculate_wer
from jiwer import cer as calculate_cer
from tqdm import tqdm

normalizer = EnglishTextNormalizer()
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("large-v3-turbo", device=device)


def load_audio_to_tensor_16k(path, target_sr=16000):
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.to(device)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != target_sr:
        resampler = T.Resample(sample_rate, target_sr).to(device)
        waveform = resampler(waveform)
    return waveform.squeeze()


def asr_process(audio_input):
    if torch.is_tensor(audio_input):
        audio_input = audio_input.cpu().numpy()
    result = whisper_model.transcribe(
        audio_input, fp16=(device == "cuda"), language="en"
    )
    pred = result["text"].strip()
    return normalizer(pred)


# Load data
df = pd.read_csv(
    "/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/benchmark/test_metadata.csv",
    dtype={"id": str},
)

# df = df.head(5) # Comment dòng này để chạy full 1849 file

test_dir = "/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/benchmark/output_wavs/2query_t3_freeze"
save_df = "benchmark/results/wer_result.csv"
results = []

with torch.inference_mode():
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_name = row["id"]
        audio_path = os.path.join(test_dir, f"{audio_name}.wav")
        gt_text = normalizer(str(row["text"]))

        if os.path.exists(audio_path):
            try:
                audio_tensor = load_audio_to_tensor_16k(audio_path)
                pred_text = asr_process(audio_tensor)
                wer = round(calculate_wer(gt_text, pred_text), 3)
                cer = round(calculate_cer(gt_text, pred_text), 3)
                res_row = row.to_dict()
                res_row.update(
                    {"asr_preds": pred_text, "wer_preds": wer, "cer_preds": cer}
                )
                results.append(res_row)
            except Exception as e:
                print(f"Lỗi xử lý file {audio_name}: {e}")

if results:
    final_df = pd.DataFrame(results)
    final_df.to_csv(save_df, index=False, sep="|")
    print(f"\nAverage WER: {final_df['wer_preds'].mean():.4f}")
    print(f"Average CER: {final_df['cer_preds'].mean():.4f}")
else:
    print("Không tìm thấy file audio nào để xử lý.")
