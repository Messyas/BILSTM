#!/bin/bash
#SBATCH -J bilstm-cleanup
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH -o logs/%x-%j.out
# Apaga a cópia local dos dados no nó. Uso: sbatch -w <nó> --dependency=afterany:<jobs> jobs/cleanup_node_data.slurm
du -sh /tmp/bilstm-data 2>/dev/null
rm -rf /tmp/bilstm-data
echo "limpo em $(hostname): $(ls -d /tmp/bilstm-data 2>/dev/null || echo ok)"
