import os
import argparse
from huggingface_hub import HfApi, create_repo
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def push_to_hub(model_dir, repo_id, token=None, private=True):
    """
    Push T3 checkpoint directory to Hugging Face Hub (excluding intermediate checkpoints).
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"Error: Directory {model_dir} does not exist.")
        return

    api = HfApi(token=HF_TOKEN)

    # Create repo if not exists
    print(f"Creating/Checking repo: {repo_id}")
    try:
        create_repo(repo_id, token=HF_TOKEN, private=private, exist_ok=True)
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    print("Uploading files...")

    # Allow list of files to upload
    allow_patterns = [
        "*.safetensors",  # t3_cfg.safetensors, s3gen.safetensors, ve.safetensors
        "*.pt",  # mapper.pt, conds.pt
        "*.json",  # tokenizer.json, config files
        "*.bin",  # training_args.bin (optional)
        "*.txt",  # Manifests/Logs (optional)
    ]

    # Implicitly ignore checkpoint-* folders via ignore_patterns
    ignore_patterns = ["checkpoint-*", "*.DS_Store", "__pycache__", "wandb", "cache"]

    try:
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            token=HF_TOKEN,
        )
        print(f"\n✅ Successfully uploaded to https://huggingface.co/{repo_id}")
        print("\nTo use this model:")
        print(f"git clone https://huggingface.co/{repo_id}")
        print("Then load with InstructionChatterBox.from_local('./' + repo_name)")

    except Exception as e:
        print(f"Error uploading: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push Chatterbox T3 to Hugging Face Hub"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to local checkpoint directory",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repo ID (e.g. username/chatterbox-t3)",
    )
    parser.add_argument(
        "--public", action="store_true", help="Make repo public (default is private)"
    )

    args = parser.parse_args()

    push_to_hub(args.model_dir, args.repo_id, private=not args.public)
