from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """

    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(t3=self.t3.__dict__, gen=self.gen)
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs["t3"]), kwargs["gen"])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> "ChatterboxTTS":
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device("cpu")
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]

        missing_keys, unexpected_keys = t3.load_state_dict(t3_state, strict=False)

        if len(missing_keys) > 0:
            print(
                f"WARN: Missing keys in state_dict (Normal for Adapters): {len(missing_keys)} keys."
            )
            # Bạn có thể uncomment dòng dưới để check kỹ xem có phải chỉ thiếu adapter không
            # print(missing_keys)

        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(
                device
            )

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> "ChatterboxTTS":
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
                )
            device = "cpu"

        for fpath in [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(
            s3gen_ref_wav, S3GEN_SR, device=self.device
        )

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [ref_16k_wav[: self.ENC_COND_LEN]], max_len=plen
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(
                self.device
            )

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(
            self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
        )
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, (
                "Please `prepare_conditionals` first or specify `audio_prompt_path`"
            )

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat(
                [text_tokens, text_tokens], dim=0
            )  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def generate_with_instruction(
        self,
        text,
        instruction,
        instruction_tokenizer=None,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        max_instruction_len=128,
    ):
        """
        Generate speech with style controlled by a text instruction.

        Args:
            text: The text content to synthesize
            instruction: Style instruction (e.g., "Speak happily and excited")
            instruction_tokenizer: T5 tokenizer for encoding instructions.
                                   If None, will auto-load from google/flan-t5-large
            audio_prompt_path: Optional path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor (0.0 - 1.0)
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            max_instruction_len: Maximum instruction token length

        Returns:
            torch.Tensor: Generated waveform
        """
        # Lazy load instruction tokenizer if not provided
        if instruction_tokenizer is None:
            if (
                not hasattr(self, "_instruction_tokenizer")
                or self._instruction_tokenizer is None
            ):
                from transformers import AutoTokenizer

                self._instruction_tokenizer = AutoTokenizer.from_pretrained(
                    "google/flan-t5-large"
                )
            instruction_tokenizer = self._instruction_tokenizer

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, (
                "Please `prepare_conditionals` first or specify `audio_prompt_path`"
            )

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Tokenize instruction
        instruction_inputs = instruction_tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_instruction_len,
        )
        instruction_input_ids = instruction_inputs.input_ids.to(self.device)
        instruction_attention_mask = instruction_inputs.attention_mask.to(self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                instruction_input_ids=instruction_input_ids,
                instruction_attention_mask=instruction_attention_mask,
            )
            # Extract only the conditional batch
            speech_tokens = speech_tokens[0]

            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)


class InstructionChatterBox:
    """
    Instruction-only TTS: Generate speech from text instructions without reference audio.

    Uses:
    - InstructionMapper to predict speaker_emb (for T3) and x_vector (for S3Gen)
    - T3 (finetuned) for text-to-token generation
    - S3Gen with zero prompts for token-to-waveform generation
    """

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        tokenizer: EnTokenizer,
        mapper,  # InstructionMapper
        instruction_tokenizer,  # T5 Tokenizer
        device: str,
    ):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.tokenizer = tokenizer
        self.mapper = mapper
        self.instruction_tokenizer = instruction_tokenizer
        self.device = device
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(
        cls,
        t3_ckpt_dir,  # Dir containing t3_cfg.safetensors, s3gen.safetensors, tokenizer.json
        mapper_ckpt_path,  # Path to mapper checkpoint (.pt)
        device,
    ) -> "InstructionChatterBox":
        """
        Load InstructionChatterBox from local checkpoints.

        Args:
            t3_ckpt_dir: Directory containing finetuned T3 weights and other model files
            mapper_ckpt_path: Path to InstructionMapper checkpoint (best_model.pt)
            device: Device to load models on
        """
        from transformers import AutoTokenizer
        from .models.t3.modules.instruction_mapper_slice import InstructionMapper

        t3_ckpt_dir = Path(t3_ckpt_dir)
        mapper_ckpt_path = Path(mapper_ckpt_path)

        # Load T3 (finetuned) - includes style_query/style_attn if trained
        print(f"Loading T3 from {t3_ckpt_dir}...")
        t3 = T3()
        t3_state = load_file(t3_ckpt_dir / "t3_cfg.safetensors")

        # Check if T3 checkpoint contains style adapter weights
        style_keys = [
            k for k in t3_state.keys() if k.startswith("instr_encoder.style_")
        ]
        if style_keys:
            print(f"  -> Found {len(style_keys)} style adapter keys in T3 checkpoint")

        missing_keys, unexpected_keys = t3.load_state_dict(t3_state, strict=False)
        if len(missing_keys) > 0:
            print(f"  -> Missing keys (adapter/encoder): {len(missing_keys)}")
        t3.to(device).eval()

        # Load InstructionMapper
        print(f"Loading InstructionMapper from {mapper_ckpt_path}...")
        mapper_ckpt = torch.load(mapper_ckpt_path, map_location="cpu")

        # Load mapper
        mapper = InstructionMapper()
        mapper.load_state_dict(mapper_ckpt["mapper"])
        mapper.to(device).eval()

        # Load InstructionEncoder weights (query/attn from mapper) into T3
        # Note: style_query/style_attn already loaded from T3 checkpoint above
        if "encoder" in mapper_ckpt and hasattr(t3, "instr_encoder"):
            missing, unexpected = t3.instr_encoder.load_state_dict(
                mapper_ckpt["encoder"], strict=False
            )
            print(
                f"  -> Loaded mapper encoder weights ({len(mapper_ckpt['encoder'])} keys)"
            )

        # Load S3Gen
        print(f"Loading S3Gen from {t3_ckpt_dir}...")
        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(t3_ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        # Load text tokenizer
        tokenizer = EnTokenizer(str(t3_ckpt_dir / "tokenizer.json"))

        # Load instruction tokenizer (T5)
        print("Loading T5 tokenizer...")
        instruction_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

        print("InstructionChatterBox loaded successfully!")
        return cls(t3, s3gen, tokenizer, mapper, instruction_tokenizer, device)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        instruction: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        max_instruction_len: int = 128,
    ) -> torch.Tensor:
        """
        Generate speech from text and instruction only (no reference audio).

        Args:
            text: Text content to synthesize
            instruction: Style instruction (e.g., "Speak happily and excited")
            exaggeration: Emotion exaggeration factor
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            max_instruction_len: Maximum instruction token length

        Returns:
            torch.Tensor: Generated waveform
        """
        # 1. Tokenize instruction
        instruction_inputs = self.instruction_tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_instruction_len,
        )
        instruction_ids = instruction_inputs.input_ids.to(self.device)
        attention_mask = instruction_inputs.attention_mask.to(self.device)

        # 2. Get style embedding from InstructionEncoder (use mapper's query/attn)
        style_emb = self.t3.instr_encoder(
            instruction_ids, attention_mask, use_for="mapper"
        )

        # 3. Predict speaker_emb [256] and x_vector [192] from Mapper
        spk_emb, x_vector = self.mapper.inference(style_emb)

        # 4. Build T3 conditioning (with zeros for prompt tokens)
        t3_cond = T3Cond(
            speaker_emb=spk_emb,
            cond_prompt_speech_tokens=torch.zeros(
                1,
                self.t3.hp.speech_cond_prompt_len,
                dtype=torch.long,
                device=self.device,
            ),
            emotion_adv=exaggeration * torch.ones(1, 1, 1, device=self.device),
        ).to(device=self.device)

        # 5. Tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        # 6. Generate speech tokens via T3
        speech_tokens = self.t3.inference(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            max_new_tokens=1000,
            temperature=temperature,
            cfg_weight=cfg_weight,
            instruction_input_ids=instruction_ids if cfg_weight > 0 else None,
            instruction_attention_mask=attention_mask if cfg_weight > 0 else None,
        )
        speech_tokens = speech_tokens[0]  # Extract conditional batch
        speech_tokens = drop_invalid_tokens(speech_tokens)
        speech_tokens = speech_tokens.to(self.device)

        # 7. Build S3Gen ref_dict with predicted x_vector and ZEROS for prompts
        ref_dict = {
            "prompt_token": torch.zeros(1, 1, dtype=torch.long, device=self.device),
            "prompt_token_len": torch.tensor([1], device=self.device),
            "prompt_feat": torch.zeros(1, 1, 80, device=self.device),
            "prompt_feat_len": None,
            "embedding": x_vector,
        }

        # 8. Generate waveform via S3Gen
        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=ref_dict,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
        watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)

        return torch.from_numpy(watermarked_wav).unsqueeze(0)
