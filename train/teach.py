"""Teacher predictions for distillation (runs on Kaggle 2×T4 with torchrun).

  torchrun --nproc_per_node 2 teach.py --model answerdotai/ModernBERT-large --data <dataset> --ckpt '<glob>/best.pt'

Loads a finished train.py run (best.pt, and the temperatures in the metrics.json beside it), scores train.jsonl and
unlabelled.jsonl, and writes calibrated probabilities to <out>/teacher_r<rank>.jsonl: per head, and per job line
[P(none), P(yes), P(partial), P(no), P(must)]. Rows are keyed "train:<line>" / "unl:<line>" for train.py --teacher.
"""
import argparse
import datetime
import glob
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from train import HEADS, LacunaModel, batches, collate, encode, line_logits, load


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="answerdotai/ModernBERT-large")
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--ckpt", required=True, help="glob for the teacher's best.pt")
    ap.add_argument("--out", default="out")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=4))
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = sorted(glob.glob(args.ckpt, recursive=True))[0]
    temps = json.loads((Path(ckpt).parent / "metrics.json").read_text())["temperatures"]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = LacunaModel(args.model)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device).eval()

    items = [(f"{split}:{i}", ex) for split in ("train", "unl")
             for i, ex in enumerate(load(Path(args.data) / ("unlabelled.jsonl" if split == "unl" else "train.jsonl")))]
    if args.limit:
        items = items[: args.limit // 2] + items[-args.limit // 2:]
    items = items[rank::world]
    t0 = time.time()
    enc = [encode(tok, ex, args.max_len) for _, ex in items]
    if rank == 0:
        print(f"temperatures {temps}; rank 0 encoded {len(enc)} in {time.time() - t0:.0f}s", flush=True)

    bs = batches(enc, args.eval_tokens, False, 0)
    t0 = time.time()
    with open(out / f"teacher_r{rank}.jsonl", "w") as f:
        for n, b in enumerate(bs):
            ids, mask, _, lines, _ = collate([enc[i] for i in b], tok.pad_token_id)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, tok_met, tok_must = model(ids.to(device), mask.to(device))
            probs = {h: F.softmax(logits[h].float() / temps[h], -1).cpu().tolist() for h in HEADS}
            met, must = line_logits(tok_met, tok_must, lines)
            per_ex = [{} for _ in b]
            if met is not None:
                pm, pk = F.softmax(met.float(), -1).cpu().tolist(), torch.sigmoid(must.float()).cpu().tolist()
                for l, a, c in zip(lines, pm, pk):
                    per_ex[l[0]][str(l[5])] = [round(v, 4) for v in a] + [round(c, 4)]
            for j, i in enumerate(b):
                key, ex = items[i]
                f.write(json.dumps({"key": key, "cv_id": ex["cv_id"], "lines": per_ex[j],
                                    "probs": {h: [round(v, 4) for v in probs[h][j]] for h in HEADS}}) + "\n")
            if rank == 0 and (n + 1) % 100 == 0:
                el = time.time() - t0
                print(f"batch {n + 1}/{len(bs)} {el / 60:.0f}m eta {el / (n + 1) * (len(bs) - n - 1) / 60:.0f}m", flush=True)
    print(f"rank {rank} done: {len(items)} rows in {(time.time() - t0) / 60:.0f} min", flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
