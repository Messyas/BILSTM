#!/bin/bash
#SBATCH -J bilstm-train
#SBATCH -p gpu
#SBATCH --gres=gpu:4090:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00
#SBATCH --array=0-2%2
#SBATCH -o logs/%x-%A_%a.out
# No máximo 2 GPUs ao mesmo tempo (%2). Uso: sbatch jobs/train_array.slurm
BACKBONES=(densenet121 inception_v3 mobilenet_v2)
export BACKBONE=${BACKBONES[$SLURM_ARRAY_TASK_ID]}
echo "backbone=$BACKBONE"
source jobs/_common.sh
R train --config configs/rlvs_baseline.toml --backbone "$BACKBONE" --attention --temporal-head bilstm
echo "end=$(date -Is)"
