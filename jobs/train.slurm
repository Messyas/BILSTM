#!/bin/bash
#SBATCH -J bilstm-train
#SBATCH -p gpu
#SBATCH --gres=gpu:4090:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00
#SBATCH -o logs/%x-%j.out
# Uso: sbatch -J train-<bb> --export=ALL,BACKBONE=<bb>[,EPOCHS=1] jobs/train.slurm
source jobs/_common.sh
: "${BACKBONE:?defina BACKBONE}"
EXTRA=()
[[ -n "${EPOCHS:-}" ]] && EXTRA+=(--epochs "$EPOCHS")
R train --config configs/rlvs_baseline.toml --backbone "$BACKBONE" --attention --temporal-head bilstm "${EXTRA[@]}"
echo "end=$(date -Is)"
