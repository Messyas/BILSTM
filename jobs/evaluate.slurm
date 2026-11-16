#!/bin/bash
#SBATCH -J bilstm-eval
#SBATCH -p gpu
#SBATCH --gres=gpu:4090:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00
#SBATCH -o logs/%x-%j.out
# Uso: sbatch -J eval-<bb> --export=ALL,RUN_ID=<rlvs-...> jobs/evaluate.slurm
# Ordem obrigatória: calibrar na validação -> teste RLVS + robustez -> RWF-2000 (mesmo limiar).
source jobs/_common.sh
: "${RUN_ID:?defina RUN_ID}"
if [[ ! -f "output/$RUN_ID/threshold.json" ]]; then
  R calibrate --run-id "$RUN_ID"
fi
cat "output/$RUN_ID/threshold.json" | head -20
R evaluate --run-id "$RUN_ID" --robustness --video-compression
R evaluate --run-id "$RUN_ID" --dataset-root data/raw/rwf2000-v1 --external-manifest data/processed/rwf2000-external.json
echo "end=$(date -Is)"
