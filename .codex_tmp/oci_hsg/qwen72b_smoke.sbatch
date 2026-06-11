#!/bin/bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH --qos=short
#SBATCH -N 2
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH -J veomni-qwen72b-cg
#SBATCH -o /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/zeruiw/veomni_cg_fsdp4/logs/qwen72b-smoke-%j.out

set -euo pipefail

WORK=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/zeruiw/veomni_cg_fsdp4
IMAGE=nvcr.io#nvidia/pytorch:25.05-py3

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURM_NODELIST}"
date

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT=$((25000 + SLURM_JOB_ID % 10000))
GLOBAL_BATCH_SIZE=$((SLURM_NNODES * 4))
echo "master_addr=${MASTER_ADDR}"
echo "master_port=${MASTER_PORT}"
echo "global_batch_size=${GLOBAL_BATCH_SIZE}"

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --nodes="${SLURM_NNODES}" --gres=gpu:4 \
  --container-image="${IMAGE}" \
  --container-mounts=/lustre:/lustre \
  --container-remap-root \
  bash -lc '
set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
export MODELING_BACKEND=hf
export VEOMNI_CG_WORK="'"${WORK}"'"
export DEPS="${VEOMNI_CG_WORK}/deps"
export PYTHONPATH="${DEPS}:${VEOMNI_CG_WORK}/VeOmni:${PYTHONPATH:-}"
export VEOMNI_TORCHRUN_NNODES="'"${SLURM_NNODES}"'"
export VEOMNI_TORCHRUN_LOCAL_WORLD_SIZE=4
export VEOMNI_TORCHRUN_NODE_RANK="${SLURM_NODEID}"
export VEOMNI_GLOBAL_BATCH_SIZE="'"${GLOBAL_BATCH_SIZE}"'"
export MASTER_ADDR="'"${MASTER_ADDR}"'"
export MASTER_PORT="'"${MASTER_PORT}"'"
export HF_HOME="/tmp/veomni_hf_${SLURM_JOB_ID}_${SLURM_NODEID}"
export HF_DATASETS_CACHE="/tmp/veomni_hf_datasets_${SLURM_JOB_ID}_${SLURM_NODEID}"
cd "${VEOMNI_CG_WORK}/VeOmni"

python - <<PY
import torch
print("TORCH", torch.__version__, "CUDA", torch.version.cuda, "NGPU", torch.cuda.device_count(), flush=True)
for idx in range(torch.cuda.device_count()):
    print("GPU", idx, torch.cuda.get_device_name(idx), flush=True)
PY

python - <<PY
missing = []
for mod in ["transformers", "datasets", "einops", "rich"]:
    try:
        __import__(mod)
    except Exception:
        missing.append(mod)
if missing:
    print("MISSING_DEPS", ",".join(missing), flush=True)
    raise SystemExit(3)
print("DEPS_OK", flush=True)
PY

python -u "${VEOMNI_CG_WORK}/VeOmni/.codex_tmp/oci_hsg/run_qwen72b_smoke.py"
'

date
