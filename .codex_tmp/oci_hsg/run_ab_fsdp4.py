import faulthandler
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


WORK = Path(os.environ["VEOMNI_CG_WORK"])
ROOT = WORK / "bench"
CFG = ROOT / "tiny_llama_config"
WEIGHTS = ROOT / "tiny_llama_weights"
OUT = ROOT / "out"
TRAIN_SCRIPT = str(WORK / "VeOmni" / "tests" / "train_scripts" / "train_text_test.py")


def note(message: str) -> None:
    print(f"[run_ab_fsdp4] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def main() -> None:
    faulthandler.enable(file=sys.stderr)
    faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)
    note("start")
    ROOT.mkdir(parents=True, exist_ok=True)
    CFG.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 128,
        "initializer_range": 0.02,
        "intermediate_size": 256,
        "max_position_embeddings": 512,
        "model_type": "llama",
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": 1024,
    }
    (CFG / "config.json").write_text(json.dumps(config, indent=2))
    note(f"wrote config: {CFG / 'config.json'}")

    os.chdir(WORK / "VeOmni")
    if WEIGHTS.exists():
        note(f"remove old weights: {WEIGHTS}")
        shutil.rmtree(WEIGHTS)
    note("import test helpers and transformers")
    from tests.tools import DummyDataset
    from transformers import LlamaConfig, LlamaForCausalLM

    note("build tiny model")
    model = LlamaForCausalLM(LlamaConfig.from_pretrained(CFG))
    note(f"save tiny weights: {WEIGHTS}")
    model.save_pretrained(WEIGHTS)
    del model
    note("tiny weights saved")

    note("create dummy dataset")
    dummy = DummyDataset(seq_len=128, dataset_type="text")
    train_path = dummy.save_path
    note(f"dummy dataset: {train_path}")
    OUT.mkdir(exist_ok=True)

    def run(name: str, extra: list[str]) -> dict:
        note(f"run {name}: start")
        out = OUT / name
        if out.exists():
            shutil.rmtree(out)
        port = 29500 + (abs(hash(name)) % 1000)
        cmd = [
            "torchrun",
            "--nnodes=1",
            "--nproc_per_node=4",
            f"--master_port={port}",
            TRAIN_SCRIPT,
            f"--model.config_path={CFG}",
            f"--model.model_path={WEIGHTS}",
            f"--data.train_path={train_path}",
            "--data.dyn_bsz_buffer_size=1",
            "--data.max_seq_len=128",
            "--train.global_batch_size=8",
            "--train.micro_batch_size=1",
            "--train.init_device=meta",
            "--train.bsz_warmup_ratio=0",
            "--train.num_train_epochs=1",
            "--train.max_steps=8",
            "--train.checkpoint.save_epochs=0",
            "--train.checkpoint.save_steps=0",
            "--train.checkpoint.save_hf_weights=False",
            "--train.enable_full_determinism=True",
            "--train.enable_batch_invariant_mode=False",
            "--train.gradient_checkpointing.enable=False",
            "--train.optimizer.lr=5e-5",
            "--train.optimizer.lr_min=0",
            "--train.optimizer.lr_start=0",
            "--train.optimizer.weight_decay=0",
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
        t0 = time.time()
        log_file = OUT / f"{name}.log"
        cmd = ["timeout", "900", *cmd]
        note(f"run {name}: command {' '.join(cmd)}")
        with log_file.open("w") as log:
            proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(f"[{name}] {line}", end="", flush=True)
            returncode = proc.wait(timeout=30)
        elapsed = time.time() - t0
        result = {"returncode": returncode, "elapsed": elapsed}
        log_path = out / "log_dict.json"
        if log_path.exists():
            result.update(json.loads(log_path.read_text()))
        print("RESULT", name, json.dumps(result, allow_nan=True), flush=True)
        if returncode != 0:
            raise SystemExit(returncode)
        return result

    eager = run("eager", [])
    cuda_graph = run(
        "cuda_graph",
        [
            "--train.cuda_graph.enable=True",
            "--train.cuda_graph.scope=auto",
            "--train.cuda_graph.num_warmup_steps=1",
            "--train.cuda_graph.strict=True",
        ],
    )

    summary = {"eager": eager, "cuda_graph": cuda_graph}
    diffs = []
    for left, right in zip(eager.get("loss", []), cuda_graph.get("loss", [])):
        if isinstance(left, float) and isinstance(right, float) and math.isfinite(left) and math.isfinite(right):
            diffs.append(abs(left - right))
        else:
            diffs.append(float("nan"))
    summary["loss_abs_diff"] = diffs
    summary["max_loss_abs_diff"] = max([diff for diff in diffs if math.isfinite(diff)] or [float("nan")])
    print("SUMMARY", json.dumps(summary, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
