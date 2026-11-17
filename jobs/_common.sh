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
# Copia um diretório de dados para o SSD local do nó (/tmp/bilstm-data), uma vez por nó.
# Jobs do mesmo nó reaproveitam a cópia (flock). A integridade é conferida pelo SHA-256 do manifesto.
# Limpeza: jobs/cleanup_node_data.slurm ao fim da campanha.
stage() {
  local src="$1" name dst
  name=$(printf '%s' "$src" | tr '/ ' '__')
  dst="/tmp/bilstm-data/$name"
  mkdir -p /tmp/bilstm-data
  (
    flock 9
    if [[ ! -f "$dst/.complete" ]]; then
      echo "copiando $src -> $dst" >&2
      rsync -a "$src/" "$dst/" && touch "$dst/.complete"
    fi
  ) 9>/tmp/bilstm-data/.lock
  printf '%s' "$dst"
}
