"""Run and inspect demonstrations. / Executa e consulta demonstrações."""

import argparse
import json
from bilstm_pipeline.execution import load_run, run


def main() -> None:
    """Parse CLI arguments. / Interpreta argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="BILSTM experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("run", help="Run the synthetic demonstration")
    start.add_argument("--config", required=True, help="TOML path relative to project root or absolute")
    start.add_argument("--id", help="Optional new run ID")
    start.add_argument("--seed", type=int)
    start.add_argument("--points", type=int)
    show = commands.add_parser("show", help="Read a completed run")
    show.add_argument("--id", required=True)
    args = parser.parse_args()
    try:
        if args.command == "run":
            overrides = {key: getattr(args, key) for key in ("seed", "points") if getattr(args, key) is not None}
            run_id = run(args.config, overrides, run_id=args.id)
            print(f"Synthetic demonstration completed. Run ID: {run_id}")
        else:
            result = load_run(args.id)
            print(json.dumps({"manifest": result["manifest"], "config": result["config"]}, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
