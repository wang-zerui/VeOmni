import faulthandler
import json
import math
import os
import re
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
SEQ_LEN = int(os.environ.get("VEOMNI_SEQ_LEN", "128"))
WANDB_PROJECT = os.environ.get("VEOMNI_WANDB_PROJECT", "VeOmni")
WANDB_NAME = os.environ.get("VEOMNI_WANDB_NAME", f"qwen72b-cuda-graph-ab-{os.environ.get('SLURM_JOB_ID', 'local')}")
WANDB_GROUP = os.environ.get("VEOMNI_WANDB_GROUP", "cuda-graph-ab")
WANDB_ANONYMOUS = os.environ.get("VEOMNI_WANDB_ANONYMOUS", "allow")


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


def _extract_tqdm_step_time(log_file: Path, total_steps: int):
    if not log_file.exists():
        return None
    text = log_file.read_text(errors="ignore")
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    matches = re.findall(rf"{total_steps}/{total_steps} \[[^\]]+,\s*([0-9.]+)(s/it|it/s)", text)
    if not matches:
        return None
    value, unit = matches[-1]
    value = float(value)
    return value if unit == "s/it" else 1.0 / value


def _case_summary(name: str, result: dict) -> dict:
    loss = result.get("loss", [])
    grad_norm = result.get("grad_norm", [])
    num_steps = len(loss)
    elapsed = result.get("elapsed")
    wall_step_time = elapsed / num_steps if elapsed and num_steps else None
    train_step_time = result.get("train_step_time_sec") or wall_step_time
    tokens_per_step = GLOBAL_BATCH_SIZE * SEQ_LEN
    return {
        "name": name,
        "num_steps": num_steps,
        "returncode": result.get("returncode"),
        "elapsed_sec": elapsed,
        "wall_step_time_sec": wall_step_time,
        "train_step_time_sec": train_step_time,
        "tokens_per_second": tokens_per_step / train_step_time if train_step_time else None,
        "loss_first": loss[:5],
        "loss_last": loss[-5:],
        "loss_min": min(loss) if loss else None,
        "loss_max": max(loss) if loss else None,
        "grad_norm_first": grad_norm[:5],
        "grad_norm_last": grad_norm[-5:],
        "grad_norm_min": min(grad_norm) if grad_norm else None,
        "grad_norm_max": max(grad_norm) if grad_norm else None,
        "has_nonfinite_loss": any(not math.isfinite(float(item)) for item in loss),
        "has_nonfinite_grad_norm": any(not math.isfinite(float(item)) for item in grad_norm),
    }


def _comparison_summary(baseline: dict, cuda_graph: dict) -> dict:
    baseline_loss = baseline.get("loss", [])
    cuda_loss = cuda_graph.get("loss", [])
    count = min(len(baseline_loss), len(cuda_loss))
    diffs = [abs(float(cuda_loss[idx]) - float(baseline_loss[idx])) for idx in range(count)]
    baseline_time = baseline.get("train_step_time_sec") or baseline.get("wall_step_time_sec")
    cuda_time = cuda_graph.get("train_step_time_sec") or cuda_graph.get("wall_step_time_sec")
    return {
        "loss_diff_steps": count,
        "loss_diff_max_abs": max(diffs) if diffs else None,
        "loss_diff_mean_abs": sum(diffs) / len(diffs) if diffs else None,
        "baseline_train_step_time_sec": baseline_time,
        "cuda_graph_train_step_time_sec": cuda_time,
        "speedup": baseline_time / cuda_time if baseline_time and cuda_time else None,
    }


def upload_wandb(summary: dict, baseline: dict, cuda_graph: dict) -> None:
    if NODE_RANK != 0:
        return
    try:
        import wandb
    except Exception as exc:
        print(f"WANDB_UPLOAD failed import: {type(exc).__name__}: {exc}", flush=True)
        return

    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_NAME,
            group=WANDB_GROUP,
            anonymous=WANDB_ANONYMOUS,
            config=summary["config"],
            settings=wandb.Settings(init_timeout=60),
        )
        baseline_loss = baseline.get("loss", [])
        cuda_loss = cuda_graph.get("loss", [])
        baseline_grad = baseline.get("grad_norm", [])
        cuda_grad = cuda_graph.get("grad_norm", [])
        total_steps = max(len(baseline_loss), len(cuda_loss), len(baseline_grad), len(cuda_grad))
        for idx in range(total_steps):
            metrics = {}
            if idx < len(baseline_loss):
                metrics["baseline/loss"] = baseline_loss[idx]
            if idx < len(cuda_loss):
                metrics["cuda_graph/loss"] = cuda_loss[idx]
            if idx < len(baseline_grad):
                metrics["baseline/grad_norm"] = baseline_grad[idx]
            if idx < len(cuda_grad):
                metrics["cuda_graph/grad_norm"] = cuda_grad[idx]
            if idx < len(baseline_loss) and idx < len(cuda_loss):
                metrics["ab/loss_diff_abs"] = abs(float(cuda_loss[idx]) - float(baseline_loss[idx]))
            wandb.log(metrics, step=idx + 1)
        for case_name in ("baseline", "cuda_graph"):
            case = summary["cases"][case_name]
            for key in ("elapsed_sec", "wall_step_time_sec", "train_step_time_sec", "tokens_per_second"):
                run.summary[f"{case_name}/{key}"] = case[key]
        for key, value in summary["comparison"].items():
            run.summary[f"ab/{key}"] = value
        run.summary["slurm/job_id"] = os.environ.get("SLURM_JOB_ID")
        wandb.finish()
        print(f"WANDB_UPLOAD ok url={run.url}", flush=True)
    except Exception as exc:
        print(f"WANDB_UPLOAD failed log: {type(exc).__name__}: {exc}", flush=True)


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
        f"--data.max_seq_len={SEQ_LEN}",
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
    result["train_step_time_sec"] = _extract_tqdm_step_time(log_file, MAX_STEPS)
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
                "seq_len": SEQ_LEN,
                "weight_decay": WEIGHT_DECAY,
                "wandb_group": WANDB_GROUP,
                "wandb_name": WANDB_NAME,
                "wandb_project": WANDB_PROJECT,
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
    dummy = DummyDataset(num_samples=num_samples, seq_len=SEQ_LEN, dataset_type="text")
    train_path = dummy.save_path
    note(f"dummy dataset: {train_path}")
    baseline = run_case("baseline", train_path, [])
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
    summary = {
        "config": {
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "init_range": INIT_RANGE,
            "local_world_size": LOCAL_WORLD_SIZE,
            "lr": LR,
            "lr_min": LR_MIN,
            "lr_warmup_ratio": LR_WARMUP_RATIO,
            "max_steps": MAX_STEPS,
            "nnodes": NNODES,
            "seq_len": SEQ_LEN,
            "weight_decay": WEIGHT_DECAY,
        },
        "cases": {
            "baseline": _case_summary("baseline", baseline),
            "cuda_graph": _case_summary("cuda_graph", cuda_graph),
        },
        "comparison": _comparison_summary(baseline, cuda_graph),
    }
    (OUT / "ab_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True, sort_keys=True) + "\n")
    upload_wandb(summary, baseline, cuda_graph)
    print("SUMMARY", json.dumps({"baseline": baseline, "cuda_graph": cuda_graph, "ab": summary}, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
