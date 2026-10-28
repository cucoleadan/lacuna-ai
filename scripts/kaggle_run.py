"""Run training on Kaggle from this machine.

  python scripts/kaggle_run.py upload                         # data/dataset + train/train.py -> private dataset
  python scripts/kaggle_run.py push --model jhu-clsp/ettin-encoder-32m --name ettin32 [--limit 500] [-- extra train args]
  python scripts/kaggle_run.py status --name ettin32
  python scripts/kaggle_run.py fetch --name ettin32            # outputs -> data/runs/ettin32/
  python scripts/kaggle_run.py push --script teach.py --source mblarge-v3-e6 --name teach-v3 --model answerdotai/ModernBERT-large \
      -- --ckpt '/kaggle/input/**/best.pt'                     # --source: another run's outputs mounted as input

The dataset is always private (scraped postings never leave private storage; see PLAN.md, "Data rules").
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402

USER = config.CFG.get("kaggle_user", "<kaggle-user>")
DS = "jobfit-data"
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")

RUNNER = '''import glob, os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.48", "bitsandbytes"], check=True)
import torch, transformers
print("torch", torch.__version__, "transformers", transformers.__version__, "gpus", torch.cuda.device_count(), flush=True)
subprocess.run(["nvidia-smi", "-L"])
src = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
n = max(1, torch.cuda.device_count())
cmd = ["torchrun", "--nproc_per_node", str(n), f"{src}/{SCRIPT}", "--data", src, "--out", "/kaggle/working/out"] + ARGS
print(" ".join(cmd), flush=True)
r = subprocess.run(cmd)
for f in glob.glob("/kaggle/working/out/*.pt"):  # keep outputs small unless it is a full run
    if KEEP_WEIGHTS is False:
        os.remove(f)
sys.exit(r.returncode)
'''


def sh(*args):
    print("$", " ".join(args))
    return subprocess.run(args, check=False)


def cmd_upload(a):
    src, ds = ROOT / "data" / "dataset", ROOT / "data" / "kaggle_ds"
    shutil.rmtree(ds, ignore_errors=True)
    ds.mkdir(parents=True)
    for f in ("train.jsonl", "val.jsonl", "test.jsonl", "labels.json", "unlabelled.jsonl"):
        if (src / f).exists():
            shutil.copy(src / f, ds / f)
    for f in ("train.py", "teach.py"):
        shutil.copy(ROOT / "train" / f, ds / f)
    (ds / "dataset-metadata.json").write_text(json.dumps(
        {"title": DS, "id": f"{USER}/{DS}", "licenses": [{"name": "other"}]}, indent=1))
    exists = subprocess.run([KAGGLE, "datasets", "status", f"{USER}/{DS}"], capture_output=True, text=True).returncode == 0
    if exists:
        sh(KAGGLE, "datasets", "version", "-p", str(ds), "-m", a.message or "update", "-r", "zip")
    else:
        sh(KAGGLE, "datasets", "create", "-p", str(ds), "-r", "zip")  # private by default


def cmd_push(a, extra):
    k = ROOT / "data" / "kaggle_kernels" / a.name
    shutil.rmtree(k, ignore_errors=True)
    k.mkdir(parents=True)
    args = ["--model", a.model] + (["--limit", str(a.limit)] if a.limit else []) + extra
    (k / "run.py").write_text(f"ARGS = {json.dumps(args)}\nKEEP_WEIGHTS = {not a.limit}\nSCRIPT = {a.script!r}\n" + RUNNER)
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/jobfit-{a.name}", "title": f"jobfit-{a.name}", "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{USER}/{DS}"],
        "kernel_sources": [f"{USER}/jobfit-{a.source}"] if a.source else []}, indent=1))
    sh(KAGGLE, "kernels", "push", "-p", str(k), "--accelerator", "NvidiaTeslaT4")


def cmd_status(a):
    sh(KAGGLE, "kernels", "status", f"{USER}/jobfit-{a.name}")


def cmd_fetch(a):
    out = ROOT / "data" / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    sh(KAGGLE, "kernels", "output", f"{USER}/jobfit-{a.name}", "-p", str(out))
    m = list(out.rglob("metrics.json"))
    if m:
        print(json.dumps(json.loads(m[0].read_text())["test"], indent=1))


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["upload", "push", "status", "fetch"])
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--name", default="run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--script", default="train.py", help="train.py or teach.py")
    ap.add_argument("--source", default="", help="a finished run whose outputs are mounted as input")
    ap.add_argument("-m", "--message", default="")
    a = ap.parse_args(argv)
    if a.cmd == "push":
        cmd_push(a, extra)
    else:
        globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
