#!/bin/bash
#SBATCH -J bilstm-seeds-eval
#SBATCH -p cpu
#SBATCH --exclude=work0,work1,work9
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --time=14-00:00:00
#SBATCH --array=0-14%3
#SBATCH -o logs/%x-%A_%a.out
# Avaliação em CPU dos runs do adendo de seeds (mesma sequência do protocolo principal).
# Os dados são lidos do SSD local (/tmp/bilstm-data), copiados uma vez por nó.
# Uso: sbatch --dependency=afterany:<ARRAY_TREINO> jobs/evaluate_seeds_cpu.slurm
export OMP_NUM_THREADS_JOB=${SLURM_CPUS_PER_TASK:-8}
source jobs/_common.sh
f="output/_seeds/task_${SLURM_ARRAY_TASK_ID}.tsv"
[[ -f "$f" ]] || { echo "sem $f (treino falhou?)"; exit 1; }
read -r BACKBONE SEED RUN_ID < "$f"
echo "backbone=$BACKBONE seed=$SEED run=$RUN_ID"
RLVS=$(stage "data/raw/rlvs-v1/Real Life Violence Dataset")
RWF=$(stage "data/raw/rwf2000-v1")
[[ -f "output/$RUN_ID/threshold.json" ]] || R calibrate --run-id "$RUN_ID" --device cpu --dataset-root "$RLVS"
R evaluate --run-id "$RUN_ID" --device cpu --dataset-root "$RLVS" --robustness --video-compression
R evaluate --run-id "$RUN_ID" --device cpu --dataset-root "$RWF" --external-manifest data/processed/rwf2000-external.json
echo "end=$(date -Is)"
