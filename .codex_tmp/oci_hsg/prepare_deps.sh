#!/bin/bash
set -euo pipefail

WORK=/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/zeruiw/veomni_cg_fsdp4
LOG="$WORK/logs/tmux-env.log"
mkdir -p "$WORK/deps" "$WORK/logs"
exec > >(tee -a "$LOG") 2>&1

date
echo "deps=$WORK/deps"

set +e
PYTHONPATH="$WORK/deps" python3 - <<'PY'
mods = ["transformers", "datasets", "einops", "rich", "tiktoken", "blobfile", "timm", "wandb"]
missing = []
for mod in mods:
    try:
        m = __import__(mod)
        print("OK", mod, getattr(m, "__version__", ""))
    except Exception as exc:
        print("MISSING", mod, type(exc).__name__, exc)
        missing.append(mod)
if missing:
    raise SystemExit(3)
PY
status=$?
set -e

if [ "$status" -eq 3 ]; then
  python3 -m pip install --target "$WORK/deps" --upgrade \
    "tiktoken>=0.9.0" \
    "blobfile>=3.0.0" \
    "timm" \
    "wandb"
elif [ "$status" -ne 0 ]; then
  exit "$status"
fi

PYTHONPATH="$WORK/deps" python3 - <<'PY'
mods = ["transformers", "datasets", "einops", "rich", "tiktoken", "blobfile", "timm", "wandb"]
for mod in mods:
    m = __import__(mod)
    print("FINAL_OK", mod, getattr(m, "__version__", ""))
PY

du -sh "$WORK/deps"
date
