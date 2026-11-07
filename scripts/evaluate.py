"""Evaluate prediction CSV files. / Avalia arquivos CSV de predições."""

import argparse
import json
from bilstm_pipeline.workflows import evaluate_csv


def main() -> None:
    """Run evaluation from the terminal. / Executa avaliação pelo terminal."""
    parser = argparse.ArgumentParser(description="Evaluate binary predictions from CSV")
    parser.add_argument("--input", required=True, help="CSV with sample_id, true_label and predicted_label")
    parser.add_argument("--output", required=True, help="New directory for evaluation artifacts")
    parser.add_argument("--source", required=True, help="Data provenance")
    parser.add_argument("--unit", required=True, help="Unit of each observation")
    parser.add_argument("--threshold", type=float, help="Threshold used to derive predictions from scores")
    args = parser.parse_args()
    try:
        result = evaluate_csv(args.input, args.output, source=args.source, unit=args.unit, threshold=args.threshold)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
