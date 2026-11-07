"""Split a dataset manifest. / Divide um manifesto de dados."""

import argparse
import json
from bilstm_pipeline.workflows import split_manifest


def main() -> None:
    """Run a grouped split. / Executa uma divisão agrupada."""
    parser = argparse.ArgumentParser(description="Split dataset records by group")
    parser.add_argument("--input", required=True, help="Input JSON dataset manifest")
    parser.add_argument("--output", required=True, help="New output JSON manifest")
    parser.add_argument("--proportions", nargs=3, type=float, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VALIDATION", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        summary = split_manifest(args.input, args.output, proportions=args.proportions, seed=args.seed)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
