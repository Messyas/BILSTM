#!/bin/bash
#SBATCH -J bilstm-latency
#SBATCH -p gpu
#SBATCH -w work5
#SBATCH --exclusive
#SBATCH --gres=gpu:4090:1
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH -o logs/%x-%j.out
# Uso: sbatch --export=ALL,RUN_IDS="<id1> <id2> <id3>" jobs/latency.slurm
# Nó fixo e exclusivo: os tempos não sofrem interferência de outros jobs.
source jobs/_common.sh
: "${RUN_IDS:?defina RUN_IDS}"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"; free -g | head -2
for id in $RUN_IDS; do
  R latency --run-id "$id" --device cuda
  R latency --run-id "$id" --device cpu --videos 30
done
echo "end=$(date -Is)"
