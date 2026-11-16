# Sourced by every job. / Carregado por todos os jobs.
set -euo pipefail
cd /home/james/work/Eduardo-Run/Visao/BILSTM
export PATH="$HOME/.local/bin:$PATH:$HOME/Packages/ffmpeg-env/bin"
# /tmp on the nodes is local SSD scratch; clean it on exit.
export TMPDIR="/tmp/bilstm-${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export OMP_NUM_THREADS=${OMP_NUM_THREADS_JOB:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS_JOB:-4} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS_JOB:-4}
echo "host=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
R() { uv run --no-sync python scripts/research.py "$@"; }
