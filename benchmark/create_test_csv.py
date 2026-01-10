"""
Script to convert test data from pipe-delimited format to CSV with unique IDs.
This creates a metadata file that can be used for resumable inference.
"""

import csv
from pathlib import Path


def main():
    input_file = Path("data/final_data_test.txt")
    output_file = Path("benchmark/test_metadata.csv")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Read all lines
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Write CSV with unique IDs
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["id", "audio_path", "text", "instruction"])

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) >= 3:
                audio_path = parts[0]
                text = parts[1]
                instruction = parts[2]

                # Unique ID: zero-padded 5 digits
                unique_id = f"{idx:05d}"

                writer.writerow([unique_id, audio_path, text, instruction])

    print(f"Created CSV with {idx + 1} entries: {output_file}")


if __name__ == "__main__":
    main()
