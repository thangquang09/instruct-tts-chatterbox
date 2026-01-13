```bash
cd /data1/speech/nhandt23/06_thang/instruct-tts-chatterbox
conda activate chatterbox_bm
mkdir -p benchmark/logs

CUDA_VISIBLE_DEVICES=0 nohup python benchmark/run_base_eval.py \
    --wav_dir benchmark/output_wavs/2query_t3_unfreeze \
    --output_name 2query_t3_unfreeze_base.csv \
    --input_txt data/full_test.txt \
    > benchmark/logs/unfreeze_base_eval.log 2>&1 &
```

```bash
cd /data1/speech/nhandt23/06_thang/instruct-tts-chatterbox
conda activate vox_profile

CUDA_VISIBLE_DEVICES=1 nohup python benchmark/run_emotion_accent_eval.py \
    --wav_dir benchmark/output_wavs/2query_t3_unfreeze \
    --output_name 2query_t3_unfreeze_emotion_accent.csv \
    --input_txt data/full_test.txt \
    > benchmark/logs/unfreeze_emotion_accent_eval.log 2>&1 &
```


Merge
```bash
python benchmark/merge_eval_results.py \
    --pred_base benchmark/results/2query_t3_unfreeze_base.csv \
    --pred_emotion_accent benchmark/results/2query_t3_unfreeze_emotion_accent.csv \
    --gt_tags benchmark/results/gt_tags.csv \
    --gt_emotion_accent benchmark/results/gt_emotion_accent.csv \
    --output benchmark/results/2query_t3_unfreeze_full.csv
```