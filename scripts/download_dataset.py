"""Download RLVS from Kaggle. / Baixa o RLVS do Kaggle."""

import argparse
import subprocess
import zipfile

from bilstm_pipeline.download import DEFAULT_VERSION, download_dataset


def main() -> None:
    """Parse options and download. / Interpreta opções e baixa o dataset."""
    parser = argparse.ArgumentParser(description="Download the RLVS dataset from Kaggle")
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION, help="Kaggle dataset version (default: 1)")
    parser.add_argument("--output", help="New destination directory (default: data/raw/rlvs-v<VERSION>)")
    args = parser.parse_args()
    try:
        target = download_dataset(version=args.version, output=args.output)
    except (ValueError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Download failed: {error}\nCheck Kaggle authentication and dataset access, then retry.\n")
    print(f"Dataset saved to: {target}")


if __name__ == "__main__":
    main()
