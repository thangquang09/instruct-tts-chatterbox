# import utmosv2
# import pandas as pd
# import torch

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# model = utmosv2.create_model(pretrained=True, device=device)

# mos = model.predict(
#     input_dir="/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/benchmark/output_wavs/2query",
#     batch_size=32,
#     num_workers=8,
# )
# pd.DataFrame(mos).to_csv(
#     "utmosv2_2query.csv", index=False, columns=["file_path", "predicted_mos"], sep="|"
# )

import os
import utmosv2
import pandas as pd
import torch
import time

INPUT_DIR = "/data1/speech/nhandt23/06_thang/instruct-tts-chatterbox/benchmark/output_wavs/2query_t3_freeze"
OUTPUT_CSV = "benchmark/results/utmosv2_batch_results.csv"


num_files = len(
    [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]
)
print(f"Tổng số file trong {INPUT_DIR}: {num_files}")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("--- Đang nạp model và checkpoint ---")
model = utmosv2.create_model(pretrained=True, device=device)

print(f"--- Đang xử lý toàn bộ file trong: {INPUT_DIR} ---")
start_time = time.time()

with torch.inference_mode():
    results = model.predict(input_dir=INPUT_DIR, batch_size=64, num_workers=8)

end_time = time.time()
total_seconds = end_time - start_time

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False, sep="|")

print("-" * 30)
print(f"Hoàn thành xử lý {len(df)} file.")
print(f"Tổng thời gian: {total_seconds:.2f} giây.")
print(f"Trung bình: {total_seconds / len(df):.4f} giây/file.")
print(f"Kết quả lưu tại: {OUTPUT_CSV}")
