"""Research pipeline CLI. / Interface do pipeline de pesquisa."""

import argparse
import subprocess


def main() -> None:
    """Dispatch research steps. / Executa as etapas de pesquisa."""
    parser = argparse.ArgumentParser(description="RLVS research pipeline (proposed reconstruction)")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Inventory and split videos by class and group")
    prepare.add_argument("--config", default="configs/rlvs_baseline.toml")
    prepare.add_argument("--dataset-root")
    prepare.add_argument("--manifest")
    prepare.add_argument("--groups", help="Reviewed CSV mapping every sample_id to a group_id")
    prepare.add_argument("--dataset-name", default="RLVS")
    prepare.add_argument("--source", default="mohamedmustafa/real-life-violence-situations-dataset")
    train = commands.add_parser("train", help="Train the configured model")
    train.add_argument("--config", default="configs/rlvs_baseline.toml")
    train.add_argument("--dataset-root")
    train.add_argument("--manifest")
    train.add_argument("--seed", type=int)
    train.add_argument("--device", choices=("cpu", "cuda", "auto"))
    train.add_argument("--backbone", choices=("mobilenet_v2", "densenet121", "inception_v3"))
    train.add_argument("--attention", action=argparse.BooleanOptionalAction)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--min-learning-rate", type=float)
    train.add_argument("--epochs", type=int)
    evaluate = commands.add_parser("evaluate", help="Evaluate a trained run")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--dataset-root")
    evaluate.add_argument("--external-manifest")
    evaluate.add_argument("--device", choices=("cpu", "cuda", "auto"))
    evaluate.add_argument("--robustness", action="store_true")
    evaluate.add_argument("--video-compression", action="store_true", help="Include H.264 video recompression (requires --robustness and FFmpeg with libx264)")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            from bilstm_pipeline.research_data import prepare_dataset
            output = prepare_dataset(args.config, dataset_root=args.dataset_root, manifest_path=args.manifest, dataset_name=args.dataset_name, source=args.source, groups_path=args.groups)
        elif args.command == "train":
            from bilstm_pipeline.research_training import train_model
            output = train_model(
                args.config,
                dataset_root=args.dataset_root,
                manifest_path=args.manifest,
                seed=args.seed,
                device=args.device,
                backbone=args.backbone,
                attention=args.attention,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                min_learning_rate=args.min_learning_rate,
                epochs=args.epochs,
            )
        else:
            from bilstm_pipeline.research_evaluation import evaluate_model
            output = evaluate_model(args.run_id, dataset_root=args.dataset_root, external_manifest=args.external_manifest, device=args.device, robustness=args.robustness, video_compression=args.video_compression)
    except (ValueError, OSError, RuntimeError, KeyError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
