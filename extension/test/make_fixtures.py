"""Reference outputs for test/parity.mjs (run with the repo venv): normalize_cv on raw CVs, encode() ids and line token
ranges on validation pairs, and ONNX logits for a few. Written to data/parity/ (gitignored: postings stay private)."""
import json, sqlite3, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "train"), str(ROOT / "scripts")]
import train as T
from cvtext import normalize_cv
out = ROOT / "data" / "parity"; out.mkdir(parents=True, exist_ok=True)
exp = ROOT / "export" / "lacuna-celer"
tok = AutoTokenizer.from_pretrained(exp)
raw = sqlite3.connect(ROOT / "data" / "work.db").execute("select cv from cvs order by id limit 300").fetchall()
(out / "cv.json").write_text(json.dumps([{"raw": r[0], "norm": normalize_cv(r[0])} for r in raw]))
val = T.load(ROOT / "data" / "dataset" / "val.jsonl")[::6][:200]
rows = []
for ex in val:
    e = T.encode(tok, ex, 4096)
    rows.append({"cv": ex["cv"], "job": ex["job"], "ids": e["input_ids"], "lines": [[a, b] for a, b, *_ in e["line_tok"]]})
s = ort.InferenceSession(str(exp / "model_q4f16.onnx"), providers=["CPUExecutionProvider"])
for r in rows[:5]:
    ids = np.array([r["ids"]], dtype=np.int64)
    o = s.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
    r["logits"] = {k: o[i][0].tolist() for i, k in enumerate(["verdict", "location", "seniority", "must_met"])}
(out / "pairs.json").write_text(json.dumps(rows))
print(len(raw), "cvs,", len(rows), "pairs")
