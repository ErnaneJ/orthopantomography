"""
Download DentexChallenge 2023 from Kaggle.
Dataset: truthisneverlinear/dentex-challenge-2023
Annotations: COCO JSON format (quadrant-enumeration-disease task)
Run: python src/data/download_dentex.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
OUT  = ROOT / "data" / "dentex_raw"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DentexChallenge 2023 → {OUT}")
    print("(~11 GB — pode demorar 10-30 min dependendo da conexão)\n")

    result = subprocess.run(
        ["kaggle", "datasets", "download",
         "truthisneverlinear/dentex-challenge-2023",
         "-p", str(OUT), "--unzip"],
        check=False,
    )

    if result.returncode != 0:
        print("ERROR: Download falhou. Verifique ~/.kaggle/kaggle.json ou o token.")
        sys.exit(1)

    # Confirm key file exists
    json_path = OUT / "training_data" / "training_data" / "quadrant-enumeration-disease" / "train_quadrant_enumeration_disease.json"
    if json_path.exists():
        print(f"\n✓ Download OK — annotation JSON em: {json_path}")
    else:
        # List what we have
        print("\nArquivos baixados:")
        for p in sorted(OUT.rglob("*.json")):
            print(" ", p.relative_to(ROOT))

    print("\nPróximo passo: python src/data/prepare_dentex.py")


if __name__ == "__main__":
    main()
