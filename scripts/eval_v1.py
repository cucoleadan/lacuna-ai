"""Score finished checkpoints on the frozen v1 test examples, on this machine (CPU is fine, slow for large).

  python scripts/eval_v1.py ettin68-v2e8 mblarge-v2      # data/runs/<name>/out/best.pt -> "test_v1" in results/<name>/metrics.json
  python scripts/eval_v1.py --data data/dataset_v2 mblarge-v2-e6   # score on the dataset version the run trained on

Runs trained with the current train.py already report `test_v1`; this is for checkpoints trained before it did.
Uses each run's own temperatures (fitted on validation), so the numbers match what train.py would report.
"""
import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import train as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset"))
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    torch.set_num_threads(5)
    test = T.load(Path(a.data) / "test.jsonl")
    v1 = [ex for ex in test if ex["cv_id"] <= T.V1_MAX_CV and ex["kind"] in T.V1_KINDS]
    for name in a.runs:
        out = ROOT / "data" / "runs" / name / "out"
        m = json.loads((out / "metrics.json").read_text())
        tok = AutoTokenizer.from_pretrained(out / "tokenizer")
        model = T.LacunaModel(m["model"])
        model.load_state_dict(torch.load(out / "best.pt", map_location="cpu"))
        t0 = time.time()
        enc = [T.encode(tok, ex, m["args"]["max_len"]) for ex in v1]
        res, _ = T.evaluate(model, enc, Namespace(eval_tokens=8192), torch.device("cpu"), tok.pad_token_id, m["temperatures"])
        res["n"] = len(v1)
        print(name, f"{(time.time() - t0) / 60:.0f} min", json.dumps(res), flush=True)
        pub = ROOT / "results" / name / "metrics.json"
        if pub.exists():
            p = json.loads(pub.read_text())
            p["test_v1"] = res
            pub.write_text(json.dumps(p, indent=1))


if __name__ == "__main__":
    main()
