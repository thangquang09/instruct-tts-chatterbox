import argparse
import csv
import logging
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import numpy as np
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback,
    Trainer,
    PretrainedConfig,
    AutoTokenizer,
)
from transformers import TrainingArguments as HfTrainingArguments
from datasets import load_dataset, DatasetDict, VerificationMode, Audio
import datasets

from chatterbox.tts import ChatterboxTTS, Conditionals, punc_norm, REPO_ID
from chatterbox.models.t3.t3 import T3, T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config

from chatterbox.models.t3.modules.instruction_mapper_slice import InstructionMapper
from chatterbox.models.s3tokenizer import S3_SR, SPEECH_VOCAB_SIZE
from chatterbox.models.s3gen import S3GEN_SR

# from chatterbox.utils.t3data_arguments import DataArguments
# from chatterbox.utils.t3dataset import SpeechFineTuningDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)

load_dotenv()


# --- Custom Training Arguments ---
@dataclass
class CustomTrainingArguments(HfTrainingArguments):
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": "Enable early stopping with specified patience. Default: None (disabled)."
        },
    )


# --- Argument Classes (ModelArguments, DataArguments) ---
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    local_model_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to local directory containing ve.safetensors, t3_cfg.safetensors, etc. Overrides model_name_or_path for loading."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    freeze_voice_encoder: bool = field(
        default=True, metadata={"help": "Freeze the Voice Encoder."}
    )
    freeze_s3gen: bool = field(
        default=True,
        metadata={"help": "Freeze the S3Gen model (speech token to waveform)."},
    )
    mapper_ckpt_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to trained InstructionMapper checkpoint (.pt file containing 'encoder' and 'mapper' keys). Required for instruction-only training."
        },
    )
    instruction_dropout_prob: float = field(
        default=0.5,
        metadata={
            "help": "Probability of using instruction-only mode (mapper predicts SpkEmb, prompt=zeros). Default: 0.5 (50/50 split)."
        },
    )


@dataclass
class DataArguments:
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the directory containing audio files and text files. Used if dataset_name is not provided."
        },
    )
    metadata_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a training metadata file. Used if dataset_name is not provided."
        },
    )
    eval_metadata_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a validation metadata file. If not provided and do_eval=True, will split from metadata_file using eval_split_size."
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the Hugging Face datasets library)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the Hugging Face datasets library)."
        },
    )
    train_split_name: str = field(
        default="train", metadata={"help": "The name of the training data set split."}
    )
    eval_split_name: Optional[str] = field(
        default="validation",
        metadata={"help": "The name of the evaluation data set split."},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the text column in the HF dataset."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the audio column in the HF dataset."},
    )
    max_text_len: int = field(
        default=256,
        metadata={"help": "Maximum length of text tokens (including BOS/EOS)."},
    )
    max_speech_len: int = field(
        default=800,
        metadata={"help": "Maximum length of speech tokens (including BOS/EOS)."},
    )
    audio_prompt_duration_s: float = field(
        default=3.0,
        metadata={
            "help": "Duration of audio (from start) to use for T3 conditioning prompt tokens (in seconds)."
        },
    )
    eval_split_size: float = field(
        default=0.0005,
        metadata={
            "help": "Fraction of data to use for evaluation if splitting manually. Not used if dataset_name provides eval split."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    ignore_verifications: bool = field(
        default=False, metadata={"help": "Set to true to ignore dataset verifications."}
    )

    # Instruction Prompt
    instruction_column_name: str = field(
        default="instruction",
        metadata={
            "help": "The name of the instruction/style column in the dataset. If missing, will use empty string."
        },
    )

    # Cache options (for faster data loading)
    use_cache: bool = field(
        default=False,
        metadata={
            "help": "Use pre-cached data instead of processing on-the-fly. Run scripts/preprocess_cache.py first."
        },
    )
    train_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory containing cached .pt files for training from preprocess_cache.py"
        },
    )
    eval_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory containing cached .pt files for evaluation. If not set, uses train_cache_dir."
        },
    )


# --- Dataset Class ---
class SpeechFineTuningDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        chatterbox_model: ChatterboxTTS,
        t3_config: T3Config,
        hf_dataset: Union[datasets.Dataset, List[Dict[str, str]]],
        is_hf_format: bool,
    ):
        self.data_args = data_args
        self.chatterbox_model = chatterbox_model
        self.chatterbox_t3_config = t3_config
        self.dataset_source = hf_dataset
        self.is_hf_format = is_hf_format

        self.text_tokenizer = chatterbox_model.tokenizer
        self.speech_tokenizer = chatterbox_model.s3gen.tokenizer
        self.voice_encoder = chatterbox_model.ve

        print("Loading T5 Tokenizer for Instructions...")
        self.instruction_tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-large", use_fast=True
        )

        self.s3_sr = S3_SR
        self.enc_cond_audio_len_samples = int(
            data_args.audio_prompt_duration_s * self.s3_sr
        )

    def __len__(self):
        return len(self.dataset_source)

    def _load_audio_text_from_item(self, idx):
        instruction_text = ""

        if self.is_hf_format:
            item = self.dataset_source[idx]
            text = item[self.data_args.text_column_name]

            if self.data_args.instruction_column_name in item:
                val = item[self.data_args.instruction_column_name]
                if val is not None:
                    instruction_text = str(val)

            audio_data = item[self.data_args.audio_column_name]

            if isinstance(audio_data, str):
                wav_array, original_sr = librosa.load(audio_data, sr=None, mono=True)
            elif (
                isinstance(audio_data, dict)
                and "array" in audio_data
                and "sampling_rate" in audio_data
            ):
                wav_array = audio_data["array"]
                original_sr = audio_data["sampling_rate"]
            else:
                logger.error(
                    f"Unexpected audio data format for item {idx}: {type(audio_data)}. Skipping."
                )
                return None, None, None

            if not isinstance(wav_array, np.ndarray):
                logger.error(
                    f"Audio array is not numpy for item {idx}: {type(wav_array)}. Skipping."
                )
                return None, None, None

            if original_sr != self.s3_sr:
                wav_16k = librosa.resample(
                    wav_array, orig_sr=original_sr, target_sr=self.s3_sr
                )
            else:
                wav_16k = wav_array.copy()

            if wav_16k.ndim > 1:
                wav_16k = librosa.to_mono(wav_16k)
            if wav_16k.dtype != np.float32:
                wav_16k = wav_16k.astype(np.float32)

            item_info_for_log = f"Item {idx} (text: '{text[:30]}...', audio_len: {len(wav_16k)}, audio_dtype: {wav_16k.dtype})"

            return wav_16k, text, instruction_text
        else:
            item = self.dataset_source[idx]
            audio_path = item["audio"]
            text = item["text"]

            if "instruction" in item:
                instruction_text = item["instruction"]
            elif self.data_args.instruction_column_name in item:
                instruction_text = item[self.data_args.instruction_column_name]

            try:
                wav_16k, _ = librosa.load(audio_path, sr=self.s3_sr, mono=True)
                return wav_16k, text, instruction_text
            except Exception as e:
                logger.error(f"Error loading audio {audio_path}: {e}")
                return None, None, None

    def __getitem__(self, idx) -> Optional[Dict[str, Union[torch.Tensor, float]]]:
        wav_16k, text, instruction_text = self._load_audio_text_from_item(idx)
        if wav_16k is None or text is None or len(wav_16k) == 0:
            return None

        try:
            speaker_emb_np = self.voice_encoder.embeds_from_wavs(
                [wav_16k], sample_rate=self.s3_sr
            )
            speaker_emb = torch.from_numpy(speaker_emb_np[0])
        except Exception as e:
            logger.error(
                f"Error getting speaker embedding for item {idx}: {e}. Skipping."
            )
            return None

        normalized_text = punc_norm(text)
        raw_text_tokens = self.text_tokenizer.text_to_tokens(normalized_text).squeeze(0)
        text_tokens = F.pad(
            raw_text_tokens, (1, 0), value=self.chatterbox_t3_config.start_text_token
        )
        text_tokens = F.pad(
            text_tokens, (0, 1), value=self.chatterbox_t3_config.stop_text_token
        )
        if len(text_tokens) > self.data_args.max_text_len:
            text_tokens = text_tokens[: self.data_args.max_text_len - 1]
            text_tokens = torch.cat(
                [
                    text_tokens,
                    torch.tensor(
                        [self.chatterbox_t3_config.stop_text_token],
                        device=text_tokens.device,
                    ),
                ]
            )
        text_token_len = torch.tensor(len(text_tokens), dtype=torch.long)

        try:
            raw_speech_tokens_batch, speech_token_lengths_batch = (
                self.speech_tokenizer.forward([wav_16k])
            )
            if raw_speech_tokens_batch is None or speech_token_lengths_batch is None:
                logger.error(f"S3Tokenizer returned None for item {idx}. Skipping.")
                return None
            raw_speech_tokens = raw_speech_tokens_batch.squeeze(0)[
                : speech_token_lengths_batch.squeeze(0).item()
            ]
        except Exception as e:
            logger.error(f"Error getting speech tokens for item {idx}: {e}. Skipping.")
            return None

        speech_tokens = F.pad(
            raw_speech_tokens,
            (1, 0),
            value=self.chatterbox_t3_config.start_speech_token,
        )
        speech_tokens = F.pad(
            speech_tokens, (0, 1), value=self.chatterbox_t3_config.stop_speech_token
        )
        if len(speech_tokens) > self.data_args.max_speech_len:
            speech_tokens = speech_tokens[: self.data_args.max_speech_len - 1]
            speech_tokens = torch.cat(
                [
                    speech_tokens,
                    torch.tensor(
                        [self.chatterbox_t3_config.stop_speech_token],
                        device=speech_tokens.device,
                    ),
                ]
            )
        speech_token_len = torch.tensor(len(speech_tokens), dtype=torch.long)

        # Instruction
        instruction_input_ids = self.instruction_tokenizer(
            instruction_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # T5 limit, instruction hiếm khi dài hơn
            add_special_tokens=True,
        ).input_ids.squeeze(0)  # [Seq_Len]

        cond_audio_segment = wav_16k[: self.enc_cond_audio_len_samples]
        if len(cond_audio_segment) == 0:
            cond_prompt_speech_tokens = torch.zeros(
                self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long
            )
        else:
            try:
                cond_prompt_tokens_batch, _ = self.speech_tokenizer.forward(
                    [cond_audio_segment],
                    max_len=self.chatterbox_t3_config.speech_cond_prompt_len,
                )
                if cond_prompt_tokens_batch is None:
                    #  logger.error(f"S3Tokenizer returned None for cond_prompt for item {idx}. Using zeros.")
                    cond_prompt_speech_tokens = torch.zeros(
                        self.chatterbox_t3_config.speech_cond_prompt_len,
                        dtype=torch.long,
                    )
                else:
                    cond_prompt_speech_tokens = cond_prompt_tokens_batch.squeeze(0)
            except Exception as e:
                # logger.error(f"Error getting cond prompt tokens for item {idx}: {e}. Using zeros.")
                cond_prompt_speech_tokens = torch.zeros(
                    self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long
                )

        if (
            cond_prompt_speech_tokens.size(0)
            != self.chatterbox_t3_config.speech_cond_prompt_len
        ):
            current_len = cond_prompt_speech_tokens.size(0)
            target_len = self.chatterbox_t3_config.speech_cond_prompt_len
            if current_len > target_len:
                cond_prompt_speech_tokens = cond_prompt_speech_tokens[:target_len]
            else:
                cond_prompt_speech_tokens = F.pad(
                    cond_prompt_speech_tokens, (0, target_len - current_len), value=0
                )

        emotion_adv_scalar = 0.5
        emotion_adv_scalar_tensor = torch.tensor(emotion_adv_scalar, dtype=torch.float)

        return_dict = {
            "text_tokens": text_tokens.long(),
            "text_token_lens": text_token_len.long(),
            "speech_tokens": speech_tokens.long(),
            "speech_token_lens": speech_token_len.long(),
            "t3_cond_speaker_emb": speaker_emb.float(),
            "t3_cond_prompt_speech_tokens": cond_prompt_speech_tokens.long(),
            "t3_cond_emotion_adv": emotion_adv_scalar_tensor,
            "instruction_input_ids": instruction_input_ids.long(),
        }

        return return_dict


# --- Cached Dataset Class (Fast Loading) ---
class CachedSpeechDataset(Dataset):
    """
    Dataset that loads from pre-cached .pt files generated by scripts/preprocess_cache.py.
    This is 10-100x faster than SpeechFineTuningDataset since all expensive operations
    (audio loading, resampling, tokenization, embedding) are pre-computed.
    """

    def __init__(
        self,
        data_args: DataArguments,
        chatterbox_model: ChatterboxTTS,
        t3_config: T3Config,
        cache_dir: str,
    ):
        self.data_args = data_args
        self.chatterbox_t3_config = t3_config
        self.text_tokenizer = chatterbox_model.tokenizer

        # Load all cache batches
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            raise ValueError(f"Cache directory not found: {cache_dir}")

        # Load cache metadata
        meta_path = cache_path / "cache_meta.pt"
        if meta_path.exists():
            self.cache_meta = torch.load(meta_path)
            logger.info(f"Loaded cache metadata: {self.cache_meta}")
        else:
            self.cache_meta = {}

        # Load all items from batch files
        self.items = []
        batch_files = sorted(cache_path.glob("cache_batch_*.pt"))
        logger.info(f"Loading cache from {len(batch_files)} batch files...")

        for batch_file in batch_files:
            batch_items = torch.load(batch_file)
            self.items.extend(batch_items)

        logger.info(f"Loaded {len(self.items)} items from cache")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Optional[Dict[str, Union[torch.Tensor, float]]]:
        item = self.items[idx]

        try:
            # Text tokenization (fast, ~1ms)
            normalized_text = punc_norm(item["text"])
            raw_text_tokens = self.text_tokenizer.text_to_tokens(
                normalized_text
            ).squeeze(0)
            text_tokens = F.pad(
                raw_text_tokens,
                (1, 0),
                value=self.chatterbox_t3_config.start_text_token,
            )
            text_tokens = F.pad(
                text_tokens, (0, 1), value=self.chatterbox_t3_config.stop_text_token
            )

            if len(text_tokens) > self.data_args.max_text_len:
                text_tokens = text_tokens[: self.data_args.max_text_len - 1]
                text_tokens = torch.cat(
                    [
                        text_tokens,
                        torch.tensor([self.chatterbox_t3_config.stop_text_token]),
                    ]
                )
            text_token_len = torch.tensor(len(text_tokens), dtype=torch.long)

            # Speech tokens (from cache)
            raw_speech_tokens = item["speech_tokens"]
            speech_tokens = F.pad(
                raw_speech_tokens,
                (1, 0),
                value=self.chatterbox_t3_config.start_speech_token,
            )
            speech_tokens = F.pad(
                speech_tokens, (0, 1), value=self.chatterbox_t3_config.stop_speech_token
            )

            if len(speech_tokens) > self.data_args.max_speech_len:
                speech_tokens = speech_tokens[: self.data_args.max_speech_len - 1]
                speech_tokens = torch.cat(
                    [
                        speech_tokens,
                        torch.tensor([self.chatterbox_t3_config.stop_speech_token]),
                    ]
                )
            speech_token_len = torch.tensor(len(speech_tokens), dtype=torch.long)

            # All other items directly from cache
            return_dict = {
                "text_tokens": text_tokens.long(),
                "text_token_lens": text_token_len.long(),
                "speech_tokens": speech_tokens.long(),
                "speech_token_lens": speech_token_len.long(),
                "t3_cond_speaker_emb": item["speaker_emb"].float(),
                "t3_cond_prompt_speech_tokens": item["cond_prompt_tokens"].long(),
                "t3_cond_emotion_adv": torch.tensor(0.5, dtype=torch.float),
                "instruction_input_ids": item["instruction_ids"].long(),
            }

            return return_dict

        except Exception as e:
            logger.error(f"Error loading cached item {idx}: {e}")
            return None


# --- Data Collator ---
@dataclass
class SpeechDataCollator:
    t3_config: T3Config  # Chatterbox T3Config
    text_pad_token_id: int
    speech_pad_token_id: int

    instruction_pad_token_id: int = 0

    def __call__(self, features: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        valid_features = [f for f in features if f is not None]

        if not valid_features:
            logger.warning(
                "SpeechDataCollator received no valid features. Returning empty batch."
            )
            return {}
        features = valid_features

        batch_size = len(features)
        text_tokens_list = [f["text_tokens"] for f in features]
        speech_tokens_list = [f["speech_tokens"] for f in features]
        max_text_len = max(len(t) for t in text_tokens_list)
        max_speech_len = max(len(t) for t in speech_tokens_list)

        # Pad text tokens
        padded_text_tokens = torch.stack(
            [
                F.pad(t, (0, max_text_len - len(t)), value=self.text_pad_token_id)
                for t in text_tokens_list
            ]
        )  # shape: (B, max_text_len)

        # Pad speech tokens
        padded_speech_tokens = torch.stack(
            [
                F.pad(s, (0, max_speech_len - len(s)), value=self.speech_pad_token_id)
                for s in speech_tokens_list
            ]
        )  # shape: (B, max_speech_len)

        # Collect lengths
        text_token_lens = torch.stack([f["text_token_lens"] for f in features])  # (B,)
        speech_token_lens = torch.stack(
            [f["speech_token_lens"] for f in features]
        )  # (B,)

        # Collect conditionals
        t3_cond_speaker_emb = torch.stack(
            [f["t3_cond_speaker_emb"] for f in features]
        )  # (B, D_speaker)
        t3_cond_prompt_speech_tokens = torch.stack(
            [f["t3_cond_prompt_speech_tokens"] for f in features]
        )  # (B, prompt_len)
        emotion_adv_scalars = torch.stack(
            [f["t3_cond_emotion_adv"] for f in features]
        )  # (B, 1, 1)
        t3_cond_emotion_adv = emotion_adv_scalars.view(batch_size, 1, 1)

        IGNORE_ID = -100
        prompt_len = self.t3_config.speech_cond_prompt_len

        # --- Build labels_text ---
        # Shift off BOS from padded_text_tokens: new length = max_text_len - 1
        shifted_text = padded_text_tokens[
            :, 1:
        ].contiguous()  # shape: (B, max_text_len - 1)
        T_text = shifted_text.size(1)

        # Mask positions t >= (text_len - 1)
        text_lens_minus_one = (text_token_lens - 1).clamp(min=0)  # (B,)
        arange_text = torch.arange(T_text, device=shifted_text.device)  # (T_text,)
        mask_pad_text = arange_text[None] >= text_lens_minus_one[:, None]  # (B, T_text)

        labels_text = shifted_text.clone()  # (B, T_text)
        labels_text[mask_pad_text] = IGNORE_ID  # set pad/beyond to -100

        # --- Build labels_speech ---
        # Shift off BOS from padded_speech_tokens: new length = max_speech_len - 1
        shifted_speech = padded_speech_tokens[
            :, 1:
        ].contiguous()  # shape: (B, max_speech_len - 1)
        T_speech = shifted_speech.size(1)

        # Mask positions t >= (speech_len - 1)
        speech_lens_minus_one = (speech_token_lens - 1).clamp(min=0)  # (B,)
        arange_speech = torch.arange(
            T_speech, device=shifted_speech.device
        )  # (T_speech,)
        mask_pad_speech = (
            arange_speech[None] >= speech_lens_minus_one[:, None]
        )  # (B, T_speech)

        # Mask positions t < prompt_len
        mask_prompt = (
            arange_speech[None] < prompt_len
        )  # (1, T_speech) -> broadcast to (B, T_speech)
        mask_prompt = mask_prompt.expand(batch_size, T_speech)

        # Combine masks
        mask_speech_total = mask_pad_speech | mask_prompt  # (B, T_speech)

        labels_speech = shifted_speech.clone()  # (B, T_speech)
        labels_speech[mask_speech_total] = IGNORE_ID  # set prompt & pad to -100

        # Pad Instruction Tokens
        instruction_list = [f["instruction_input_ids"] for f in features]
        max_instr_len = max(len(t) for t in instruction_list)

        if max_instr_len == 0:
            max_instr_len = 1

        padded_instruction_ids = torch.stack(
            [
                F.pad(
                    t, (0, max_instr_len - len(t)), value=self.instruction_pad_token_id
                )
                for t in instruction_list
            ]
        )

        instruction_attention_mask = (
            padded_instruction_ids != self.instruction_pad_token_id
        ).long()

        return {
            "text_tokens": padded_text_tokens,
            "text_token_lens": text_token_lens,
            "speech_tokens": padded_speech_tokens,
            "speech_token_lens": speech_token_lens,
            "t3_cond_speaker_emb": t3_cond_speaker_emb,
            "t3_cond_prompt_speech_tokens": t3_cond_prompt_speech_tokens,
            "t3_cond_emotion_adv": t3_cond_emotion_adv,
            "labels_text": labels_text,  # (B, max_text_len - 1) masked with -100
            "labels_speech": labels_speech,  # (B, max_speech_len - 1) masked with -100
            "instruction_input_ids": padded_instruction_ids,
            "instruction_attention_mask": instruction_attention_mask,
        }


# --- Model Wrapper ---
class T3ForFineTuning(torch.nn.Module):
    def __init__(
        self,
        t3_model: T3,
        chatterbox_t3_config: T3Config,
        mapper: Optional[InstructionMapper] = None,
        instruction_dropout_prob: float = 0.5,
    ):
        super().__init__()
        self.t3 = t3_model
        self.chatterbox_t3_config = chatterbox_t3_config
        self.mapper = mapper
        self.instruction_dropout_prob = instruction_dropout_prob

        class HFCompatibleConfig(PretrainedConfig):
            model_type = "chatterbox_t3_finetune"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        hf_config_instance = HFCompatibleConfig()
        hf_config_instance.llama_config_name = chatterbox_t3_config.llama_config_name
        hf_config_instance.text_tokens_dict_size = (
            chatterbox_t3_config.text_tokens_dict_size
        )
        hf_config_instance.speech_tokens_dict_size = (
            chatterbox_t3_config.speech_tokens_dict_size
        )
        hf_config_instance.max_text_tokens = chatterbox_t3_config.max_text_tokens
        hf_config_instance.max_speech_tokens = chatterbox_t3_config.max_speech_tokens
        hf_config_instance.speech_cond_prompt_len = (
            chatterbox_t3_config.speech_cond_prompt_len
        )
        hf_config_instance.start_text_token = chatterbox_t3_config.start_text_token
        hf_config_instance.stop_text_token = chatterbox_t3_config.stop_text_token
        hf_config_instance.start_speech_token = chatterbox_t3_config.start_speech_token
        hf_config_instance.stop_speech_token = chatterbox_t3_config.stop_speech_token
        self.config = hf_config_instance

        self._debug_step_count = 0

    def forward(
        self,
        text_tokens,
        text_token_lens,
        speech_tokens,
        speech_token_lens,
        t3_cond_speaker_emb,
        t3_cond_prompt_speech_tokens,
        t3_cond_emotion_adv,
        labels_text=None,
        labels_speech=None,
        instruction_input_ids=None,
        instruction_attention_mask=None,
    ):
        B = t3_cond_speaker_emb.shape[0]
        device = t3_cond_speaker_emb.device

        # ============ 50/50 Mixing Strategy ============
        # Clone inputs to avoid in-place modifications
        final_speaker_emb = t3_cond_speaker_emb.clone()
        final_prompt_tokens = t3_cond_prompt_speech_tokens.clone()
        final_instruction_ids = instruction_input_ids
        final_instruction_mask = (
            instruction_attention_mask.clone()
            if instruction_attention_mask is not None
            else None
        )

        # ============ Smart Instruction Mode Logic ============
        # - Training: Use random mixing based on instruction_dropout_prob
        # - Evaluation: If instruction_dropout_prob >= 1.0, use 100% instruction mode
        #               Otherwise, use 100% audio mode (original behavior)

        use_instruction_mode = self.mapper is not None and (
            self.training  # Always apply during training
            or self.instruction_dropout_prob
            >= 1.0  # During eval, only if 100% instruction mode
        )

        if use_instruction_mode and instruction_input_ids is not None:
            if self.training:
                # Training: Random assignment per sample
                use_instruction = (
                    torch.rand(B, device=device) < self.instruction_dropout_prob
                )  # [B]
            else:
                # Evaluation with 100% instruction mode: All samples use instruction
                use_instruction = torch.ones(B, dtype=torch.bool, device=device)

            num_instruction = use_instruction.sum().item()
            num_audio = B - num_instruction

            # --- Process Instruction-Only samples ---
            if use_instruction.any():
                instr_indices = use_instruction.nonzero(as_tuple=True)[0]

                # Get style_emb via InstructionEncoder (frozen)
                with torch.no_grad():
                    style_emb = self.t3.instr_encoder(
                        instruction_input_ids[instr_indices].to(device),
                        instruction_attention_mask[instr_indices].to(device)
                        if instruction_attention_mask is not None
                        else None,
                    )
                    # Mapper inference -> predict SpkEmb
                    pred_spk, _ = self.mapper.inference(style_emb)

                # Replace GT SpkEmb with predicted SpkEmb
                final_speaker_emb[instr_indices] = pred_spk.to(final_speaker_emb.dtype)
                # Zero out prompt tokens (no reference audio)
                final_prompt_tokens[instr_indices] = 0

            # --- Process Audio-Only samples (only during training with mixed mode) ---
            if self.training and (~use_instruction).any():
                audio_indices = (~use_instruction).nonzero(as_tuple=True)[0]

                # Mask instruction for these samples (set attention_mask to 0)
                if final_instruction_mask is not None:
                    final_instruction_mask[audio_indices] = 0

        # ============ Build T3Cond ============
        current_t3_cond = T3Cond(
            speaker_emb=final_speaker_emb,
            cond_prompt_speech_tokens=final_prompt_tokens,
            cond_prompt_speech_emb=None,
            emotion_adv=t3_cond_emotion_adv,
        ).to(device=self.t3.device)

        # Debug logging
        if self._debug_step_count < 3:
            logger.info(f"[DEBUG] Step {self._debug_step_count}: Checking inputs...")
            logger.info(
                f"[DEBUG] text_tokens: shape={text_tokens.shape}, device={text_tokens.device}, dtype={text_tokens.dtype}"
            )
            logger.info(
                f"[DEBUG] speech_tokens: shape={speech_tokens.shape}, device={speech_tokens.device}, dtype={speech_tokens.dtype}"
            )
            logger.info(
                f"[DEBUG] final_speaker_emb: shape={final_speaker_emb.shape}, has_nan={torch.isnan(final_speaker_emb).any().item()}"
            )
            if final_instruction_ids is not None:
                logger.info(
                    f"[DEBUG] instruction_input_ids: shape={final_instruction_ids.shape}, device={final_instruction_ids.device}"
                )

        loss_text, loss_speech, speech_logits = self.t3.loss(
            t3_cond=current_t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            labels_text=labels_text,
            labels_speech=labels_speech,
            instruction_input_ids=final_instruction_ids,
            instruction_attention_mask=final_instruction_mask,
        )

        total_loss = loss_text + loss_speech

        # Debug logging (only on first few steps)
        if self._debug_step_count < 3:
            logger.info(
                f"[DEBUG] Step {self._debug_step_count}: loss_text={loss_text.item():.6f}, loss_speech={loss_speech.item():.6f}, total_loss={total_loss.item():.6f}"
            )
            logger.info(
                f"[DEBUG] labels_text valid tokens: {(labels_text != -100).sum().item()}, labels_speech valid tokens: {(labels_speech != -100).sum().item()}"
            )
            logger.info(
                f"[DEBUG] speech_logits has_nan: {torch.isnan(speech_logits).any().item()}, has_inf: {torch.isinf(speech_logits).any().item()}"
            )
            self._debug_step_count += 1

        # Return loss and logits in format Trainer expects
        # For HuggingFace Trainer to compute eval_loss, we need to return an object with .loss attribute
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=speech_logits,  # Using speech logits as primary output
        )


# --- Custom Trainer for T3 (to ensure eval_loss is properly computed) ---
class T3Trainer(Trainer):
    """
    Custom Trainer that ensures eval_loss is properly computed and returned.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Override compute_loss to handle our custom model output.
        """
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to properly extract loss during evaluation.
        This ensures eval_loss is available in metrics.
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss

            if loss is not None:
                loss = loss.mean().detach()

        # Return (loss, logits, labels)
        # For our use case, we mainly care about loss
        return (loss, None, None)


trainer_instance: Optional[Trainer] = None


def main():
    global trainer_instance

    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    set_seed(training_args.seed)

    logger.info("Loading ChatterboxTTS model...")

    original_model_dir_for_copy: Optional[Path] = None
    if model_args.local_model_dir:
        logger.info(f"Loading model from local directory: {model_args.local_model_dir}")
        local_dir_path = Path(model_args.local_model_dir)
        chatterbox_model = ChatterboxTTS.from_local(
            ckpt_dir=str(local_dir_path), device="cpu"
        )
        original_model_dir_for_copy = local_dir_path
    else:
        repo_to_download = model_args.model_name_or_path or REPO_ID
        logger.info(f"Loading model from Hugging Face Hub: {repo_to_download}")
        download_dir = Path(training_args.output_dir) / "pretrained_model_download"
        download_dir.mkdir(parents=True, exist_ok=True)
        files_to_download = [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
        ]

        from huggingface_hub import hf_hub_download as hf_download
        import torch.distributed as dist

        # ============ DDP: Only rank 0 downloads, others wait ============
        is_distributed = dist.is_initialized()
        is_main_process = (not is_distributed) or (dist.get_rank() == 0)

        if is_main_process:
            logger.info("[DDP] Rank 0: Downloading model files from HuggingFace...")
            for f in files_to_download:
                try:
                    hf_download(
                        repo_id=repo_to_download,
                        filename=f,
                        local_dir=download_dir,
                        local_dir_use_symlinks=False,
                        cache_dir=model_args.cache_dir,
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not download {f} from {repo_to_download}: {e}."
                    )

            try:
                hf_download(
                    repo_id=repo_to_download,
                    filename="conds.pt",
                    local_dir=download_dir,
                    local_dir_use_symlinks=False,
                    cache_dir=model_args.cache_dir,
                )
            except:
                logger.info(
                    "conds.pt not found on Hub or failed to download for this model."
                )

        # All ranks wait for rank 0 to finish downloading
        if is_distributed:
            logger.info(
                f"[DDP] Rank {dist.get_rank()}: Waiting at barrier for model download..."
            )
            dist.barrier()
            logger.info(
                f"[DDP] Rank {dist.get_rank()}: Barrier passed, loading model..."
            )

        chatterbox_model = ChatterboxTTS.from_local(ckpt_dir=download_dir, device="cpu")
        original_model_dir_for_copy = download_dir

    t3_model = chatterbox_model.t3
    chatterbox_t3_config_instance = t3_model.hp

    if model_args.freeze_voice_encoder:
        for param in chatterbox_model.ve.parameters():
            param.requires_grad = False
        logger.info("Voice Encoder frozen.")
    if model_args.freeze_s3gen:
        for param in chatterbox_model.s3gen.parameters():
            param.requires_grad = False
        logger.info("S3Gen model frozen.")

    # ============ Load InstructionMapper (if provided) ============
    instruction_mapper = None
    if model_args.mapper_ckpt_path:
        logger.info(f"Loading InstructionMapper from: {model_args.mapper_ckpt_path}")
        mapper_ckpt = torch.load(model_args.mapper_ckpt_path, map_location="cpu")

        # Initialize and load Mapper
        instruction_mapper = InstructionMapper()
        instruction_mapper.load_state_dict(mapper_ckpt["mapper"])
        instruction_mapper.eval()
        for p in instruction_mapper.parameters():
            p.requires_grad = False
        logger.info("InstructionMapper loaded and frozen.")

        # Load InstructionEncoder weights from same checkpoint
        if hasattr(t3_model, "instr_encoder") and "encoder" in mapper_ckpt:
            # Verify what keys we're loading
            encoder_keys = list(mapper_ckpt["encoder"].keys())
            adapter_keys = [k for k in encoder_keys if not k.startswith("t5.")]
            t5_keys = [k for k in encoder_keys if k.startswith("t5.")]
            logger.info(
                f"  -> Mapper Encoder checkpoint contains: {len(t5_keys)} T5 keys, {len(adapter_keys)} adapter keys"
            )
            logger.info(f"  -> Adapter keys: {adapter_keys}")

            # Load with strict=False (T5 weights already loaded from HF, we only need adapters)
            missing, unexpected = t3_model.instr_encoder.load_state_dict(
                mapper_ckpt["encoder"], strict=False
            )
            if missing:
                logger.warning(
                    f"  -> Missing keys (OK if these are new): {missing[:5]}..."
                )
            logger.info("InstructionEncoder weights loaded from Mapper checkpoint.")

            # VERIFY: Check that adapter weights differ from default init
            if hasattr(t3_model.instr_encoder, "query"):
                query_norm = t3_model.instr_encoder.query.data.norm().item()
                logger.info(
                    f"  -> [VERIFY] InstructionEncoder.query L2 norm: {query_norm:.4f} (should be non-zero)"
                )
            if hasattr(t3_model.instr_encoder, "attn"):
                attn_out_norm = (
                    t3_model.instr_encoder.attn.out_proj.weight.data.norm().item()
                )
                logger.info(
                    f"  -> [VERIFY] InstructionEncoder.attn.out_proj L2 norm: {attn_out_norm:.4f}"
                )

    # Set T3 model to trainable
    for param in t3_model.parameters():
        param.requires_grad = True
    logger.info("T3 model set to trainable.")

    # ============ Freeze InstructionEncoder COMPLETELY ============
    # (Not just T5, but also Query, Attention, and any projection layers)
    if hasattr(t3_model, "instr_encoder"):
        for param in t3_model.instr_encoder.parameters():
            param.requires_grad = False
        instr_enc_params = sum(p.numel() for p in t3_model.instr_encoder.parameters())
        logger.info(f"InstructionEncoder FULLY frozen ({instr_enc_params:,} params).")

        # FIX SAFETENSORS ERROR: Unshare weights between t5.shared and t5.encoder.embed_tokens
        # Safetensors does not allow shared tensors. Since we freeze this anyway, cloning is safe.
        if hasattr(t3_model.instr_encoder, "t5"):
            t5_model = t3_model.instr_encoder.t5
            if (
                hasattr(t5_model, "shared")
                and hasattr(t5_model, "encoder")
                and hasattr(t5_model.encoder, "embed_tokens")
            ):
                if (
                    t5_model.shared.weight.data_ptr()
                    == t5_model.encoder.embed_tokens.weight.data_ptr()
                ):
                    logger.info("Unsharing T5 weights to fix Safetensors saving...")
                    t5_model.shared.weight = torch.nn.Parameter(
                        t5_model.shared.weight.clone()
                    )
                    t5_model.shared.weight.requires_grad = (
                        False  # Ensure it remains frozen
                    )

    # ============ Log Trainable Parameters Breakdown ============
    def count_params(module, name):
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in module.parameters())
        return trainable, total

    logger.info("=" * 50)
    logger.info("TRAINABLE PARAMETERS BREAKDOWN:")
    logger.info("=" * 50)

    # T3 overall
    t3_trainable, t3_total = count_params(t3_model, "T3")
    logger.info(
        f"  T3 Total: {t3_trainable:,} / {t3_total:,} ({100 * t3_trainable / t3_total:.2f}%)"
    )

    # Breakdown by component
    if hasattr(t3_model, "tfmr"):
        tfmr_train, tfmr_total = count_params(t3_model.tfmr, "LlamaModel")
        logger.info(f"    - LlamaModel (tfmr): {tfmr_train:,} trainable")

        # AdaRMSNorm Adapters (inside each layer)
        ada_train_total = 0
        if hasattr(t3_model.tfmr, "layers"):
            for layer in t3_model.tfmr.layers:
                if hasattr(layer, "input_adapter"):
                    ada_train_total += sum(
                        p.numel()
                        for p in layer.input_adapter.parameters()
                        if p.requires_grad
                    )
                if hasattr(layer, "post_attention_adapter"):
                    ada_train_total += sum(
                        p.numel()
                        for p in layer.post_attention_adapter.parameters()
                        if p.requires_grad
                    )
        if ada_train_total > 0:
            num_layers = (
                len(t3_model.tfmr.layers) if hasattr(t3_model.tfmr, "layers") else 0
            )
            logger.info(
                f"    - AdaRMSNorm Adapters ({num_layers} layers × 2): {ada_train_total:,} trainable"
            )

    if hasattr(t3_model, "text_emb"):
        emb_train, _ = count_params(t3_model.text_emb, "text_emb")
        logger.info(f"    - Text Embedding: {emb_train:,} trainable")

    if hasattr(t3_model, "speech_emb"):
        emb_train, _ = count_params(t3_model.speech_emb, "speech_emb")
        logger.info(f"    - Speech Embedding: {emb_train:,} trainable")

    if hasattr(t3_model, "text_head"):
        head_train, _ = count_params(t3_model.text_head, "text_head")
        logger.info(f"    - Text Head: {head_train:,} trainable")

    if hasattr(t3_model, "speech_head"):
        head_train, _ = count_params(t3_model.speech_head, "speech_head")
        logger.info(f"    - Speech Head: {head_train:,} trainable")

    if hasattr(t3_model, "instr_encoder"):
        enc_train, enc_total = count_params(
            t3_model.instr_encoder, "InstructionEncoder"
        )
        logger.info(f"    - InstructionEncoder: {enc_train:,} trainable (should be 0)")

    logger.info("=" * 50)

    logger.info("Loading and processing dataset...")
    raw_datasets = DatasetDict()
    verification_mode = (
        VerificationMode.NO_CHECKS
        if data_args.ignore_verifications
        else VerificationMode.BASIC_CHECKS
    )

    train_hf_dataset: Union[datasets.Dataset, List[Dict[str, str]]]
    eval_hf_dataset: Optional[Union[datasets.Dataset, List[Dict[str, str]]]] = None

    if data_args.dataset_name:
        logger.info(
            f"Loading dataset '{data_args.dataset_name}' from Hugging Face Hub."
        )
        raw_datasets_loaded = load_dataset(  # Use a different var name to avoid conflict with outer raw_datasets
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            verification_mode=verification_mode,
            # trust_remote_code=True # If dataset script requires it
        )
        if data_args.train_split_name not in raw_datasets_loaded:
            raise ValueError(
                f"Train split '{data_args.train_split_name}' not found. Available: {list(raw_datasets_loaded.keys())}"
            )
        train_hf_dataset = raw_datasets_loaded[data_args.train_split_name]

        if training_args.do_eval:
            if (
                data_args.eval_split_name
                and data_args.eval_split_name in raw_datasets_loaded
            ):
                eval_hf_dataset = raw_datasets_loaded[data_args.eval_split_name]
            elif "validation" in raw_datasets_loaded:
                eval_hf_dataset = raw_datasets_loaded["validation"]
            elif "test" in raw_datasets_loaded:
                eval_hf_dataset = raw_datasets_loaded["test"]
            elif (
                data_args.eval_split_size > 0 and len(train_hf_dataset) > 1
            ):  # Ensure dataset is splittable
                logger.info(
                    f"Splitting train dataset for evaluation with ratio {data_args.eval_split_size}"
                )
                split_dataset = train_hf_dataset.train_test_split(
                    test_size=data_args.eval_split_size, seed=training_args.seed
                )
                train_hf_dataset, eval_hf_dataset = (
                    split_dataset["train"],
                    split_dataset["test"],
                )
                logger.info(f"Evaluation set size: {len(eval_hf_dataset)}")
            else:
                logger.warning(
                    "Evaluation requested but no eval split found/configured or train dataset too small to split. Skipping eval dataset."
                )
        is_hf_format_train, is_hf_format_eval = True, True
    else:
        all_files = []
        if data_args.metadata_file:
            metadata_path = Path(data_args.metadata_file)
            dataset_root = metadata_path.parent
            with open(metadata_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="|")
                for line_idx, parts in enumerate(reader):
                    if not parts:
                        continue

                    if len(parts) >= 2:
                        audio_file = parts[0]
                        text = parts[1]

                        instruction = parts[2] if len(parts) > 2 else ""

                        audio_path = (
                            Path(audio_file)
                            if Path(audio_file).is_absolute()
                            else dataset_root / audio_file
                        )
                        if audio_path.exists():
                            all_files.append(
                                {
                                    "audio": str(audio_path),
                                    "text": text,
                                    "instruction": instruction,
                                }
                            )
                        else:
                            logger.warning(
                                f"Audio file not found: {audio_path} (line {line_idx + 1}). Skipping."
                            )
                    else:
                        logger.warning(
                            f"Skipping malformed line in metadata (line {line_idx + 1}): {parts}"
                        )
        elif data_args.dataset_dir:
            dataset_path = Path(data_args.dataset_dir)
            for audio_file_path in dataset_path.rglob("*.wav"):
                text_file_path = audio_file_path.with_suffix(".txt")
                if text_file_path.exists():
                    with open(text_file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    all_files.append({"audio": str(audio_file_path), "text": text})
        if not all_files:
            raise ValueError(
                "No data files found from local paths. Check dataset_dir or metadata_file."
            )
        np.random.shuffle(all_files)
        train_hf_dataset = all_files  # type: ignore

        # ============ Handle Validation Data ============
        if training_args.do_eval:
            if data_args.eval_metadata_file:
                # Load validation data from separate file
                eval_files = []
                eval_metadata_path = Path(data_args.eval_metadata_file)
                eval_dataset_root = eval_metadata_path.parent
                with open(eval_metadata_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="|")
                    for line_idx, parts in enumerate(reader):
                        if not parts:
                            continue
                        if len(parts) >= 2:
                            audio_file = parts[0]
                            text = parts[1]
                            instruction = parts[2] if len(parts) > 2 else ""
                            audio_path = (
                                Path(audio_file)
                                if Path(audio_file).is_absolute()
                                else eval_dataset_root / audio_file
                            )
                            if audio_path.exists():
                                eval_files.append(
                                    {
                                        "audio": str(audio_path),
                                        "text": text,
                                        "instruction": instruction,
                                    }
                                )
                            else:
                                logger.warning(
                                    f"Eval audio file not found: {audio_path} (line {line_idx + 1}). Skipping."
                                )
                        else:
                            logger.warning(
                                f"Skipping malformed line in eval metadata (line {line_idx + 1}): {parts}"
                            )
                if eval_files:
                    eval_hf_dataset = eval_files
                    logger.info(
                        f"Loaded {len(eval_files)} samples from eval_metadata_file: {data_args.eval_metadata_file}"
                    )
                else:
                    logger.warning(
                        "No valid samples found in eval_metadata_file. Skipping evaluation."
                    )
            elif data_args.eval_split_size > 0 and len(all_files) > 1:
                # Fall back to splitting from training data
                split_idx = int(len(all_files) * (1 - data_args.eval_split_size))
                if split_idx == 0:
                    split_idx = 1
                if split_idx == len(all_files):
                    split_idx = len(all_files) - 1
                train_hf_dataset, eval_hf_dataset = (
                    all_files[:split_idx],
                    all_files[split_idx:],
                )
                logger.info(
                    f"Split training data: {len(train_hf_dataset)} train, {len(eval_hf_dataset)} eval"
                )

        is_hf_format_train, is_hf_format_eval = False, False

    # ============ Create Datasets ============
    if data_args.use_cache and data_args.train_cache_dir:
        # Use cached data (10-100x faster)
        logger.info(f"Using CACHED data from: {data_args.train_cache_dir}")
        train_dataset = CachedSpeechDataset(
            data_args,
            chatterbox_model,
            chatterbox_t3_config_instance,
            cache_dir=data_args.train_cache_dir,
        )

        eval_dataset = None
        if training_args.do_eval:
            eval_cache = data_args.eval_cache_dir or data_args.train_cache_dir
            if data_args.eval_cache_dir:
                logger.info(f"Using CACHED eval data from: {eval_cache}")
                eval_dataset = CachedSpeechDataset(
                    data_args,
                    chatterbox_model,
                    chatterbox_t3_config_instance,
                    cache_dir=eval_cache,
                )
            else:
                logger.warning(
                    "No eval_cache_dir specified. Using train cache for eval (not recommended)."
                )
                eval_dataset = train_dataset
    else:
        # Original on-the-fly processing
        train_dataset = SpeechFineTuningDataset(
            data_args,
            chatterbox_model,
            chatterbox_t3_config_instance,
            train_hf_dataset,
            is_hf_format_train,
        )

        eval_dataset = None
        if eval_hf_dataset and training_args.do_eval:
            eval_dataset = SpeechFineTuningDataset(
                data_args,
                chatterbox_model,
                chatterbox_t3_config_instance,
                eval_hf_dataset,
                is_hf_format_eval,
            )

    data_collator = SpeechDataCollator(
        chatterbox_t3_config_instance,
        chatterbox_t3_config_instance.stop_text_token,
        chatterbox_t3_config_instance.stop_speech_token,
    )

    hf_trainable_model = T3ForFineTuning(
        t3_model,
        chatterbox_t3_config_instance,
        mapper=instruction_mapper,
        instruction_dropout_prob=model_args.instruction_dropout_prob,
    )

    callbacks = []
    if (
        training_args.early_stopping_patience is not None
        and training_args.early_stopping_patience > 0
    ):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience
            )
        )

    trainer_instance = T3Trainer(
        model=hf_trainable_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )

    if training_args.label_names is None:
        trainer_instance.label_names = ["labels"]

    if training_args.do_train:
        logger.info("*** Training T3 model ***")
        train_result = trainer_instance.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer_instance.save_model()

        logger.info("Saving finetuned T3 model weights for ChatterboxTTS...")
        t3_to_save = (
            trainer_instance.model.t3
            if hasattr(trainer_instance.model, "t3")
            else trainer_instance.model.module.t3
        )
        finetuned_t3_state_dict = t3_to_save.state_dict()

        # ========== OPTIMIZATION: Filter out frozen InstructionEncoder weights ==========
        # InstructionEncoder is FROZEN during Stage 2 and its weights come from mapper.pt.
        # T5 weights will be loaded fresh from HF cache at inference time.
        # Adapter weights (query, attn) come from mapper.pt['encoder'].
        original_keys = len(finetuned_t3_state_dict)
        original_size_mb = (
            sum(v.numel() * v.element_size() for v in finetuned_t3_state_dict.values())
            / 1024
            / 1024
        )

        # Filter out ALL keys that start with 'instr_encoder.' (frozen - both T5 and adapters)
        finetuned_t3_state_dict = {
            k: v.clone().contiguous()
            for k, v in finetuned_t3_state_dict.items()
            if not k.startswith("instr_encoder.")
        }

        filtered_keys = len(finetuned_t3_state_dict)
        filtered_size_mb = (
            sum(v.numel() * v.element_size() for v in finetuned_t3_state_dict.values())
            / 1024
            / 1024
        )
        logger.info(
            f"  -> Filtered T5 weights: {original_keys} -> {filtered_keys} keys"
        )
        logger.info(
            f"  -> Size reduction: {original_size_mb:.1f} MB -> {filtered_size_mb:.1f} MB (saved {original_size_mb - filtered_size_mb:.1f} MB)"
        )

        output_t3_safetensor_path = (
            Path(training_args.output_dir) / "t3_cfg.safetensors"
        )
        from safetensors.torch import save_file

        save_file(finetuned_t3_state_dict, output_t3_safetensor_path)
        logger.info(f"Finetuned T3 model weights saved to {output_t3_safetensor_path}")

        # Also copy Mapper checkpoint to output dir for convenience
        if model_args.mapper_ckpt_path:
            import shutil

            mapper_dest = Path(training_args.output_dir) / "mapper.pt"
            shutil.copy2(model_args.mapper_ckpt_path, mapper_dest)
            logger.info(f"Mapper checkpoint copied to {mapper_dest}")

        if original_model_dir_for_copy:
            import shutil

            for f_name in ["ve.safetensors", "s3gen.safetensors", "tokenizer.json"]:
                src_path = original_model_dir_for_copy / f_name
                if src_path.exists():
                    shutil.copy2(src_path, Path(training_args.output_dir) / f_name)
            if (original_model_dir_for_copy / "conds.pt").exists():
                shutil.copy2(
                    original_model_dir_for_copy / "conds.pt",
                    Path(training_args.output_dir) / "conds.pt",
                )
            logger.info(
                f"Full model components structured in {training_args.output_dir}"
            )

        metrics = train_result.metrics
        trainer_instance.log_metrics("train", metrics)
        trainer_instance.save_metrics("train", metrics)
        trainer_instance.save_state()

    if training_args.do_eval and eval_dataset:
        logger.info("*** Evaluating T3 model ***")
        metrics = trainer_instance.evaluate()
        trainer_instance.log_metrics("eval", metrics)
        trainer_instance.save_metrics("eval", metrics)

    logger.info("Finetuning script finished.")


if __name__ == "__main__":
    main()
