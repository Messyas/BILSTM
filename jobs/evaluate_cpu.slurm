#!/bin/bash
#SBATCH -J bilstm-eval-cpu
#SBATCH -p cpu
#SBATCH --exclude=work0,work1,work9
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --time=14-00:00:00
#SBATCH -o logs/%x-%j.out
# Avaliação inteira em CPU (nenhuma GPU reservada e ociosa).
# Uso: sbatch -J eval-<bb> --export=ALL,RUN_ID=<rlvs-...> jobs/evaluate_cpu.slurm
# Ordem: calibrar na validação -> teste RLVS + robustez + H.264 -> RWF-2000 (mesmo limiar congelado).
# Os scores de validação, teste e RWF saem todos do mesmo dispositivo (CPU).
export OMP_NUM_THREADS_JOB=${SLURM_CPUS_PER_TASK:-8}
source jobs/_common.sh
: "${RUN_ID:?defina RUN_ID}"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"
[[ -f "output/$RUN_ID/threshold.json" ]] || R calibrate --run-id "$RUN_ID" --device cpu
python3 -c "import json;t=json.load(open('output/$RUN_ID/threshold.json'));print({k:t[k] for k in ('threshold','target_fpr','validation_fpr','validation_recall','target_attained')})"
R evaluate --run-id "$RUN_ID" --device cpu --robustness --video-compression
R evaluate --run-id "$RUN_ID" --device cpu --dataset-root data/raw/rwf2000-v1 --external-manifest data/processed/rwf2000-external.json
echo "end=$(date -Is)"
