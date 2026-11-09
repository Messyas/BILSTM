"""Download violence detection datasets from Kaggle. / Baixa datasets de detecção de violência do Kaggle."""

import argparse
import subprocess
import zipfile

from bilstm_pipeline.download import DATASET_REF, DEFAULT_VERSION, download_dataset


def main() -> None:
    """Parse options and download. / Interpreta opções e baixa o dataset."""
    parser = argparse.ArgumentParser(description="Download RLVS or external benchmark datasets (e.g. rwf2000, scvd, hockey) from Kaggle")
    parser.add_argument("--dataset", default=DATASET_REF, help="Dataset alias (rlvs, rwf2000, scvd, hockey) or Kaggle '<owner>/<slug>'")
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION, help="Kaggle dataset version (default: 1)")
    parser.add_argument("--output", help="New destination directory (default: data/raw/<dataset>-v<VERSION>)")
    args = parser.parse_args()
    try:
        target = download_dataset(dataset=args.dataset, version=args.version, output=args.output)
    except (ValueError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Download failed: {error}\nCheck Kaggle authentication and dataset access, then retry.\n")
    print(f"Dataset saved to: {target}")


if __name__ == "__main__":
    main()
