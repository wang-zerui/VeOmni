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
LOCAL_WORLD_SIZE = int(os.environ.get("VEOMNI_TORCHRUN_LOCAL_WORLD_SIZE", "4"))
NNODES = int(os.environ.get("VEOMNI_TORCHRUN_NNODES", "1"))
NODE_RANK = int(os.environ.get("VEOMNI_TORCHRUN_NODE_RANK", "0"))
MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.environ.get("MASTER_PORT")
GLOBAL_BATCH_SIZE = int(os.environ.get("VEOMNI_GLOBAL_BATCH_SIZE", str(LOCAL_WORLD_SIZE * NNODES)))
MAX_STEPS = int(os.environ.get("VEOMNI_MAX_STEPS", "500"))
INIT_RANGE = float(os.environ.get("VEOMNI_INIT_RANGE", "0.006"))
LR = float(os.environ.get("VEOMNI_LR", "1.5e-4"))
LR_MIN = float(os.environ.get("VEOMNI_LR_MIN", "1.5e-6"))
LR_WARMUP_RATIO = float(os.environ.get("VEOMNI_LR_WARMUP_RATIO", "0.1"))
WEIGHT_DECAY = float(os.environ.get("VEOMNI_WEIGHT_DECAY", "0.1"))


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
    cfg["initializer_range"] = INIT_RANGE
    (CONFIG_DIR / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    keys = [
        "model_type",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
    ]
    note("config " + json.dumps({key: cfg.get(key) for key in keys}, sort_keys=True))


def run_case(name: str, train_path: str, extra: list[str]) -> dict:
    note(f"{name}: start")
    out = OUT / name
    if out.exists():
        shutil.rmtree(out)
    port = 29600 + (abs(hash(name)) % 1000)
    cmd = [
        "timeout",
        "3600",
        "torchrun",
        f"--nnodes={NNODES}",
        f"--nproc_per_node={LOCAL_WORLD_SIZE}",
        f"--node_rank={NODE_RANK}",
        f"--master_addr={MASTER_ADDR}",
        f"--master_port={MASTER_PORT or port}",
        TRAIN_SCRIPT,
        f"--model.config_path={CONFIG_DIR}",
        f"--data.train_path={train_path}",
        "--data.dyn_bsz_buffer_size=1",
        "--data.max_seq_len=128",
        f"--train.global_batch_size={GLOBAL_BATCH_SIZE}",
        "--train.micro_batch_size=1",
        "--train.init_device=meta",
        "--train.bsz_warmup_ratio=0",
        "--train.num_train_epochs=1",
        f"--train.max_steps={MAX_STEPS}",
        "--train.checkpoint.save_epochs=0",
        "--train.checkpoint.save_steps=0",
        "--train.checkpoint.save_hf_weights=False",
        "--train.enable_full_determinism=True",
        "--train.enable_batch_invariant_mode=False",
        "--train.gradient_checkpointing.enable=False",
        "--train.optimizer.type=anyprecision_adamw",
        f"--train.optimizer.lr={LR}",
        f"--train.optimizer.lr_min={LR_MIN}",
        "--train.optimizer.lr_start=0",
        f"--train.optimizer.lr_warmup_ratio={LR_WARMUP_RATIO}",
        "--train.optimizer.lr_decay_style=cosine",
        "--train.optimizer.lr_decay_ratio=1.0",
        f"--train.optimizer.weight_decay={WEIGHT_DECAY}",
        "--train.accelerator.fsdp_config.fsdp_mode=fsdp2",
        "--train.accelerator.fsdp_config.mixed_precision.enable=False",
        "--train.accelerator.ulysses_size=1",
        "--train.accelerator.ep_size=1",
        "--model.ops_implementation.attn_implementation=eager",
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
    log_file = OUT / f"{name}.node{NODE_RANK}.log"
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
    note(
        "launcher "
        + json.dumps(
            {
                "local_world_size": LOCAL_WORLD_SIZE,
                "nnodes": NNODES,
                "node_rank": NODE_RANK,
                "master_addr": MASTER_ADDR,
                "master_port": MASTER_PORT,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "init_range": INIT_RANGE,
                "lr": LR,
                "lr_min": LR_MIN,
                "lr_warmup_ratio": LR_WARMUP_RATIO,
                "max_steps": MAX_STEPS,
                "weight_decay": WEIGHT_DECAY,
            },
            sort_keys=True,
        )
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK / "VeOmni")
    prepare_config()
    from tests.tools import DummyDataset

    num_samples = max(GLOBAL_BATCH_SIZE * (MAX_STEPS + 8), 1024)
    dummy = DummyDataset(num_samples=num_samples, seq_len=128, dataset_type="text")
    train_path = dummy.save_path
    note(f"dummy dataset: {train_path}")
    cuda_graph = run_case(
        "cuda_graph",
        train_path,
        [
            "--train.cuda_graph.enable=True",
            "--train.cuda_graph.scope=attn",
            "--train.cuda_graph.num_warmup_steps=1",
            "--train.cuda_graph.strict=True",
        ],
    )
    print("SUMMARY", json.dumps({"cuda_graph": cuda_graph}, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
