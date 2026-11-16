#!/bin/bash
#SBATCH -J bilstm-eval
#SBATCH -p gpu
#SBATCH --gres=gpu:4090:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00
#SBATCH --array=0-2%2
#SBATCH -o logs/%x-%A_%a.out
# No máximo 2 GPUs. Uso: sbatch --export=ALL,RUN_IDS="rlvs-a rlvs-b rlvs-c" jobs/evaluate_array.slurm
# Ordem: calibrar na validação -> teste RLVS + robustez + H.264 -> RWF-2000 (mesmo limiar congelado).
read -r -a IDS <<< "${RUN_IDS:?defina RUN_IDS}"
export RUN_ID=${IDS[$SLURM_ARRAY_TASK_ID]}
echo "run=$RUN_ID"
source jobs/_common.sh
[[ -f "output/$RUN_ID/threshold.json" ]] || R calibrate --run-id "$RUN_ID"
python3 -c "import json;t=json.load(open('output/$RUN_ID/threshold.json'));print({k:t[k] for k in ('threshold','target_fpr','validation_fpr','validation_recall','target_attained')})"
R evaluate --run-id "$RUN_ID" --robustness --video-compression
R evaluate --run-id "$RUN_ID" --dataset-root data/raw/rwf2000-v1 --external-manifest data/processed/rwf2000-external.json
echo "end=$(date -Is)"
