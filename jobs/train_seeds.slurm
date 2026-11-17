#!/bin/bash
#SBATCH -J bilstm-seeds
#SBATCH -p gpu
#SBATCH --gres=gpu:4090:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00
#SBATCH --array=0-14%2
#SBATCH -o logs/%x-%A_%a.out
# Adendo suplementar: seeds de treino 1-5 x 3 backbones (o split continua o do manifesto, seed 42).
# No máximo 2 GPUs (%2). Grava o RUN_ID em output/_seeds/task_<i>.tsv para o job de avaliação.
BACKBONES=(mobilenet_v2 densenet121 inception_v3)
SEEDS=(1 2 3 4 5)
i=$SLURM_ARRAY_TASK_ID
export BACKBONE=${BACKBONES[$((i % 3))]} SEED=${SEEDS[$((i / 3))]}
echo "backbone=$BACKBONE seed=$SEED"
source jobs/_common.sh
R train --config configs/rlvs_baseline.toml --backbone "$BACKBONE" --attention --temporal-head bilstm --seed "$SEED" | tee "$TMPDIR/train.out"
run=$(grep -o 'Run ID: rlvs-[^ ]*' "$TMPDIR/train.out" | awk '{print $3}')
[[ -n "$run" ]] || { echo "RUN_ID não encontrado"; exit 1; }
printf '%s\t%s\t%s\n' "$BACKBONE" "$SEED" "$run" > "output/_seeds/task_${i}.tsv"
echo "end=$(date -Is)"
