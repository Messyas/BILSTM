#!/bin/bash
#SBATCH -J bilstm-latency
#SBATCH -p gpu
#SBATCH -w work6
#SBATCH --gres=gpu:4090:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=14-00:00:00
#SBATCH -o logs/%x-%j.out
# Uso: sbatch --export=ALL,RUN_IDS="<id1> <id2> <id3>" jobs/latency.slurm
# Só GPU (a placa reservada é usada o tempo todo). Nó fixo (work6; o work5 teve travamentos de NFS em 16/11) para que todos os
# backbones sejam medidos no mesmo hardware. Sem --exclusive: reserve apenas o que usa.
source jobs/_common.sh
: "${RUN_IDS:?defina RUN_IDS}"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"; free -g | head -2
for id in $RUN_IDS; do
  R latency --run-id "$id" --device cuda
done
echo "end=$(date -Is)"
