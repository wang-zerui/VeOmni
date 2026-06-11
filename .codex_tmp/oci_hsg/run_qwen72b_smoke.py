import faulthandler
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


WORK = Path(os.environ["VEOMNI_CG_WORK"])
ROOT = WORK / "bench" / "qwen72b_smoke"
CONFIG_DIR = ROOT / "qwen2_5_72b_config"
OUT = ROOT / "out"
TRAIN_SCRIPT = str(WORK / "VeOmni" / "tests" / "train_scripts" / "train_text_test.py")


def note(message: str) -> None:
    print(f"[run_qwen72b_smoke] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def prepare_config() -> None:
    if CONFIG_DIR.exists():
        shutil.rmtree(CONFIG_DIR)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoConfig

        note("download Qwen/Qwen2.5-72B config only")
        cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-72B", trust_remote_code=True)
        cfg.save_pretrained(CONFIG_DIR)
        note("saved Qwen2.5-72B config")
    except Exception as exc:
        note(f"Qwen2.5 config fetch failed, fallback to local Qwen2-72B config: {type(exc).__name__}: {exc}")
        fallback = WORK / "VeOmni" / "configs" / "model_configs" / "qwen" / "Qwen2-72B.json"
        shutil.copyfile(fallback, CONFIG_DIR / "config.json")
    cfg = json.loads((CONFIG_DIR / "config.json").read_text())
    keys = [
        "model_type",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    ]
    note("config " + json.dumps({key: cfg.get(key) for key in keys}, sort_keys=True))


def run_case(name: str, extra: list[str]) -> dict:
    note(f"{name}: start")
    out = OUT / name
    if out.exists():
        shutil.rmtree(out)
    port = 29600 + (abs(hash(name)) % 1000)
    cmd = [
        "timeout",
        "3600",
        "torchrun",
        "--nnodes=1",
        "--nproc_per_node=4",
        f"--master_port={port}",
        TRAIN_SCRIPT,
        f"--model.config_path={CONFIG_DIR}",
        "--data.train_path=dummy_unused",
        "--data.dyn_bsz_buffer_size=1",
        "--data.max_seq_len=128",
        "--train.global_batch_size=4",
        "--train.micro_batch_size=1",
        "--train.init_device=meta",
        "--train.bsz_warmup_ratio=0",
        "--train.num_train_epochs=1",
        "--train.max_steps=2",
        "--train.checkpoint.save_epochs=0",
        "--train.checkpoint.save_steps=0",
        "--train.checkpoint.save_hf_weights=False",
        "--train.enable_full_determinism=True",
        "--train.enable_batch_invariant_mode=False",
        "--train.gradient_checkpointing.enable=False",
        "--train.optimizer.lr=0",
        "--train.optimizer.lr_min=0",
        "--train.optimizer.lr_start=0",
        "--train.optimizer.weight_decay=0",
        "--train.accelerator.fsdp_config.fsdp_mode=fsdp2",
        "--train.accelerator.fsdp_config.mixed_precision.enable=True",
        "--train.accelerator.ulysses_size=1",
        "--train.accelerator.ep_size=1",
        "--model.ops_implementation.attn_implementation=flash_attention_2",
        "--model.ops_implementation.moe_implementation=eager",
        "--model.ops_implementation.cross_entropy_loss_implementation=eager",
        "--model.ops_implementation.rms_norm_implementation=eager",
        "--model.ops_implementation.swiglu_mlp_implementation=eager",
        "--model.ops_implementation.rotary_pos_emb_implementation=eager",
        "--model.ops_implementation.load_balancing_loss_implementation=eager",
        "--model.ops_implementation.rms_norm_gated_implementation=eager",
        "--model.ops_implementation.causal_conv1d_implementation=eager",
        "--model.ops_implementation.chunk_gated_delta_rule_implementation=eager",
        f"--train.checkpoint.output_dir={out}",
        *extra,
    ]
    env = os.environ.copy()
    env["MODELING_BACKEND"] = "hf"
    env["TOKENIZERS_PARALLELISM"] = "false"
    log_file = OUT / f"{name}.log"
    t0 = time.time()
    note(f"{name}: command {' '.join(cmd)}")
    with log_file.open("w") as log:
        proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(f"[{name}] {line}", end="", flush=True)
        returncode = proc.wait(timeout=30)
    result = {"returncode": returncode, "elapsed": time.time() - t0}
    log_dict = out / "log_dict.json"
    if log_dict.exists():
        result.update(json.loads(log_dict.read_text()))
    print("RESULT", name, json.dumps(result, allow_nan=True), flush=True)
    if returncode != 0:
        raise SystemExit(returncode)
    return result


def main() -> None:
    faulthandler.enable(file=sys.stderr)
    faulthandler.dump_traceback_later(180, repeat=True, file=sys.stderr)
    os.environ.setdefault("MODELING_BACKEND", "hf")
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK / "VeOmni")
    prepare_config()
    cuda_graph = run_case(
        "cuda_graph",
        [
            "--train.cuda_graph.enable=True",
            "--train.cuda_graph.scope=auto",
            "--train.cuda_graph.num_warmup_steps=1",
            "--train.cuda_graph.strict=True",
        ],
    )
    print("SUMMARY", json.dumps({"cuda_graph": cuda_graph}, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
