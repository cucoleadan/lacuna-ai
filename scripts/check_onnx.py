"""Exported ONNX models vs the PyTorch checkpoint on validation examples (CPU).

  python scripts/check_onnx.py d68-v3-s0 export/jobfit-small --n 200

Reports macro-F1 per head for PyTorch and each .onnx file, and how often the ONNX argmax matches PyTorch's.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import train as T  # noqa: E402

HEADS = ["verdict", "location", "seniority", "must_met"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("export")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset"))
    ap.add_argument("--src", default="", help="run output dir (default data/runs/<run>/out)")
    ap.add_argument("--onnx", default="*.onnx", help="glob of exported files to check")
    a = ap.parse_args()
    src, exp = Path(a.src) if a.src else ROOT / "data" / "runs" / a.run / "out", Path(a.export)
    m = json.loads((src / "metrics.json").read_text())
    tok = AutoTokenizer.from_pretrained(exp)
    val = T.load(Path(a.data) / "val.jsonl")
    data = val[:: max(1, len(val) // a.n)][: a.n]  # spread over the split
    enc = [T.encode(tok, ex, m["args"]["max_len"]) for ex in data]

    model = T.JobFit(m["model"])
    model.load_state_dict(torch.load(src / "best.pt", map_location="cpu"))
    model.eval()
    def report(name, p):
        f1 = {h: T.macro_f1([e[h] for e in enc], [x[h] for x in p], T.HEADS[h]) for h in HEADS}
        same = {h: np.mean([x[h] == y[h] for x, y in zip(p, ref)]) for h in HEADS} if name != "pytorch" else {}
        print(f"{name:18} F1 " + " ".join(f"{h} {v:.3f}" for h, v in f1.items()) +
              ("  | same argmax as PyTorch: " + " ".join(f"{h} {v:.0%}" for h, v in same.items()) if same else ""), flush=True)

    ref = []
    with torch.no_grad():
        for e in enc:
            ids = torch.tensor([e["input_ids"]])
            heads, _, _ = model(ids, torch.ones_like(ids))
            ref.append({h: int(heads[h][0].argmax()) for h in HEADS})
    report("pytorch", ref)
    for f in sorted(exp.glob(a.onnx)):
        s = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
        p = []
        for e in enc:
            ids = np.array([e["input_ids"]], dtype=np.int64)
            out = dict(zip(T_OUT, s.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})))
            p.append({h: int(out[h][0].argmax()) for h in HEADS})
        report(f.name, p)

T_OUT = ["verdict", "location", "seniority", "must_met", "line_met", "line_must"]

if __name__ == "__main__":
    main()
