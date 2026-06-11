#!/bin/bash
#SBATCH -A nvr_elm_llm
#SBATCH -p batch
#SBATCH --qos=short
#SBATCH -N 1
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

srun --ntasks=1 --nodes=1 --gres=gpu:4 \
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
