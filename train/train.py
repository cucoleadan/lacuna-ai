"""jobfit multi-head training (runs on Kaggle 2×T4 with torchrun, or one GPU / CPU for smoke tests).

  torchrun --nproc_per_node 2 train.py --model answerdotai/ModernBERT-base --data /kaggle/input/jobfit-data

Heads on one encoder pass over "CV [SEP] job":
  verdict (apply/stretch/skip), location (yes/unclear/no), seniority (below/at/above), must_met (yes/no),
  and per job line: met (none/yes/partial/no) + must (0/1), applied per token and averaged over each line's
  tokens (a linear head commutes with mean pooling, so the browser can average token logits per line).
Saves the best checkpoint by validation verdict macro-F1, per-head temperatures fitted on validation,
and test metrics (accuracy, macro-F1, ECE) to <out>/metrics.json, on the full test split and on the v1 test examples (`test_v1`).

Distillation (--teacher): soft targets from a teacher's predictions (teach.py) on train.jsonl and unlabelled.jsonl.
Labelled examples train on (1 - alpha) * hard + alpha * soft cross-entropy, unlabelled ones on the soft targets only.
Validation and test still score against the hard labels.
"""
import argparse
import datetime
import glob
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation at 4k tokens
import torch  # noqa: E402
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

HEADS = {"verdict": 3, "location": 3, "seniority": 3, "must_met": 2}
V1_MAX_CV, V1_KINDS = 1973, {"target", "near", "seniority", "location"}  # the frozen v1 test examples, shared by every dataset version
LINE_MET = 4


# ---------- data ----------

def load(path):
    return [json.loads(l) for l in open(path)]


def encode(tok, ex, max_len):
    """Tokenize CV + job; map each job line's character span to token positions in the job segment."""
    enc = tok(ex["cv"], ex["job"], truncation="longest_first", max_length=max_len, return_offsets_mapping=True)
    seq = enc.sequence_ids()
    line_tok = []
    for li, ln in enumerate(ex["lines"]):
        a, b = ln["span"]
        idx = [i for i, (s, (x, y)) in enumerate(zip(seq, enc["offset_mapping"])) if s == 1 and x < b and y > a]
        if idx:  # unlabelled pairs have no per-line labels: -1
            met = ["none", "yes", "partial", "no"].index(ln["met"]) if "met" in ln else -1
            line_tok.append((idx[0], idx[-1] + 1, met, ln.get("must", -1), li))
    return {"input_ids": enc["input_ids"], "line_tok": line_tok, "soft": ex.get("soft"), **{h: ex.get(h, -1) for h in HEADS}}


def batches(data, tokens_per_batch, shuffle, seed):
    """Length-bucketed batches with a token budget (padding waste stays small)."""
    order = sorted(range(len(data)), key=lambda i: len(data[i]["input_ids"]))
    out, cur, longest = [], [], 0
    for i in order:
        n = len(data[i]["input_ids"])
        if cur and max(longest, n) * (len(cur) + 1) > tokens_per_batch:
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        out.append(cur)
    if shuffle:
        random.Random(seed).shuffle(out)
    return out


def collate(items, pad_id):
    L = max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), L), pad_id)
    mask = torch.zeros((len(items), L), dtype=torch.long)
    for i, x in enumerate(items):
        ids[i, : len(x["input_ids"])] = torch.tensor(x["input_ids"])
        mask[i, : len(x["input_ids"])] = 1
    labels = {h: torch.tensor([x[h] for x in items]) for h in HEADS}
    lines = [(i, a, b, met, must, li) for i, x in enumerate(items) for a, b, met, must, li in x["line_tok"]]
    soft = None
    if any(x["soft"] for x in items):  # teacher probabilities; zero rows (mask 0) where an example has none
        soft = {h: torch.tensor([x["soft"]["probs"][h] if x["soft"] else [0.0] * k for x in items]) for h, k in HEADS.items()}
        soft["mask"] = torch.tensor([float(bool(x["soft"])) for x in items])
        sl = [items[l[0]]["soft"]["lines"].get(str(l[5])) if items[l[0]]["soft"] else None for l in lines]
        soft["line"] = torch.tensor([v if v else [0.0] * (LINE_MET + 1) for v in sl])
        soft["line_mask"] = torch.tensor([float(v is not None) for v in sl])
    return ids, mask, labels, lines, soft


# ---------- model ----------

class JobFit(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name, attn_implementation="sdpa")
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.heads = nn.ModuleDict({k: nn.Linear(h, n) for k, n in HEADS.items()})
        self.line_met = nn.Linear(h, LINE_MET)  # per token, averaged per line
        self.line_must = nn.Linear(h, 1)

    def forward(self, ids, mask):
        x = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(x.dtype)
        pooled = self.drop((x * m).sum(1) / m.sum(1))
        return {k: head(pooled) for k, head in self.heads.items()}, self.line_met(x), self.line_must(x).squeeze(-1)


def line_logits(tok_met, tok_must, lines):
    if not lines:
        return None, None
    met = torch.stack([tok_met[i, a:b].mean(0) for i, a, b, *_ in lines])
    must = torch.stack([tok_must[i, a:b].mean() for i, a, b, *_ in lines])
    return met, must


def soft_ce(logits, p):
    return -(p * F.log_softmax(logits, -1)).sum(-1)


def distil_loss(out, labels, lines, w_line, soft, alpha):
    """Per example and per line: (1 - alpha) * hard CE + alpha * soft CE where both exist, else whichever exists."""
    logits, tok_met, tok_must = out
    dev = tok_met.device
    loss = 0.0 * (tok_met.float().sum() + tok_must.float().sum())
    sm = soft["mask"].to(dev)
    for h in HEADS:
        lg, y = logits[h].float(), labels[h]
        has = (y >= 0).float()
        hard = F.cross_entropy(lg, y.clamp(min=0), reduction="none") * has
        w_hard = has * (1 - alpha * sm)
        w_soft = sm * (1 - has * (1 - alpha))
        loss = loss + (w_hard * hard + w_soft * soft_ce(lg, soft[h].to(dev))).mean()
    met, must = line_logits(tok_met, tok_must, lines)
    if met is not None:
        met, must = met.float(), must.float()
        y_met = torch.tensor([l[3] for l in lines], device=dev)
        y_must = torch.tensor([float(l[4]) for l in lines], device=dev)
        s, slm = soft["line"].to(dev), soft["line_mask"].to(dev)
        has = (y_met >= 0).float()
        w_hard, w_soft = has * (1 - alpha * slm), slm * (1 - has * (1 - alpha))
        hard = F.cross_entropy(met, y_met.clamp(min=0), reduction="none") * has
        loss = loss + w_line * (w_hard * hard + w_soft * soft_ce(met, s[:, :LINE_MET])).mean()
        # must flag only on requirement lines: labelled ones (met > 0), or the teacher thinks it is one
        req_h = (y_met > 0).float()
        req_s = slm * (s[:, 0] < 0.5).float()
        wh, ws = req_h * (1 - alpha * req_s), req_s * (1 - req_h * (1 - alpha))
        n = ((wh > 0) | (ws > 0)).sum()
        if n:
            bh = F.binary_cross_entropy_with_logits(must, y_must.clamp(min=0), reduction="none")
            bs = F.binary_cross_entropy_with_logits(must, s[:, LINE_MET], reduction="none")
            loss = loss + w_line * (wh * bh + ws * bs).sum() / n
    return loss


def loss_fn(out, labels, lines, w_line, soft=None, alpha=0.5):
    if soft is not None:
        return distil_loss(out, labels, lines, w_line, soft, alpha)
    logits, tok_met, tok_must = out
    loss = sum(F.cross_entropy(logits[h].float(), labels[h]) for h in HEADS)
    # zero-weight terms keep the line heads in the graph when a batch has no requirement lines (one-example
    # batches with large models); otherwise DDP stops with "parameters that were not used in producing loss"
    loss = loss + 0.0 * (tok_met.float().sum() + tok_must.float().sum())
    met, must = line_logits(tok_met, tok_must, lines)
    if met is not None:
        y_met = torch.tensor([l[3] for l in lines], device=met.device)
        y_must = torch.tensor([float(l[4]) for l in lines], device=met.device)
        req = y_met > 0  # the must flag only matters on requirement lines
        loss = loss + w_line * F.cross_entropy(met.float(), y_met)
        if req.any():
            loss = loss + w_line * F.binary_cross_entropy_with_logits(must.float()[req], y_must[req])
    return loss


# ---------- metrics ----------

def macro_f1(y, p, k):
    f = []
    for c in range(k):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f) / k


def ece(probs, y, bins=10):
    conf, pred = probs.max(1)
    err = 0.0
    for b in range(bins):
        sel = (conf > b / bins) & (conf <= (b + 1) / bins)
        if sel.any():
            err += sel.float().mean().item() * abs((pred[sel] == y[sel]).float().mean().item() - conf[sel].mean().item())
    return err


def fit_temperature(logits, y):
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        l = F.cross_entropy(logits / t.exp(), y)
        l.backward()
        return l
    opt.step(closure)
    return t.exp().item()


@torch.no_grad()
def evaluate(model, data, args, device, pad_id, temps=None):
    model.eval()
    acc = {h: [] for h in HEADS}
    ys = {h: [] for h in HEADS}
    line_y, line_p = [], []
    for b in batches(data, args.eval_tokens, False, 0):
        ids, mask, labels, lines, _ = collate([data[i] for i in b], pad_id)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, tok_met, tok_must = model(ids.to(device), mask.to(device))
        for h in HEADS:
            acc[h].append(logits[h].float().cpu())
            ys[h].append(labels[h])
        met, _ = line_logits(tok_met, tok_must, lines)
        if met is not None:
            line_y += [l[3] for l in lines]
            line_p += met.argmax(1).cpu().tolist()
    res, raw = {}, {}
    for h, k in HEADS.items():
        lg, y = torch.cat(acc[h]), torch.cat(ys[h])
        t = (temps or {}).get(h, 1.0)
        pr = F.softmax(lg / t, 1)
        p = pr.argmax(1)
        res[h] = {"acc": round((p == y).float().mean().item(), 4), "macro_f1": round(macro_f1(y.tolist(), p.tolist(), k), 4),
                  "ece": round(ece(pr, y), 4)}
        raw[h] = (lg, y)
    if line_y:
        res["line_met"] = {"acc": round(sum(a == b for a, b in zip(line_y, line_p)) / len(line_y), 4),
                           "macro_f1": round(macro_f1(line_y, line_p, LINE_MET), 4)}
    model.train()
    return res, raw


def attach_soft(data, root, pattern):
    """Teacher rows are keyed "train:<line>" / "unl:<line>" (teach.py); unlabelled pairs join the train split."""
    rows = {}
    for f in glob.glob(pattern, recursive=True):
        for l in open(f):
            r = json.loads(l)
            rows[r["key"]] = r
    unl = load(root / "unlabelled.jsonl")
    for split, exs in (("train", data["train"]), ("unl", unl)):
        for i, ex in enumerate(exs):
            r = rows.get(f"{split}:{i}")
            assert r is None or r["cv_id"] == ex["cv_id"], "teacher rows do not match this dataset version"
            ex["soft"] = r and {"probs": r["probs"], "lines": r["lines"]}
    data["train"] = data["train"] + [ex for ex in unl if ex["soft"]]
    print(f"teacher rows {len(rows)}; train with unlabelled {len(data['train'])}", flush=True)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--out", default="out")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--tokens-per-batch", type=int, default=8192, help="per GPU per step")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--w-line", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="train on this many examples (timing runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw")
    ap.add_argument("--teacher", default="", help="glob of teach.py outputs (teacher_r*.jsonl): distil, adding unlabelled.jsonl")
    ap.add_argument("--alpha", type=float, default=0.5, help="soft-target weight on labelled examples")
    args = ap.parse_args()

    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))  # rank 0 evaluates alone; large takes > 10 min
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_proc = rank == 0
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    data = {k: load(Path(args.data) / f"{k}.jsonl") for k in ("train", "val", "test")}
    if args.teacher:
        attach_soft(data, Path(args.data), args.teacher)
    if args.limit:
        data["train"] = data["train"][: args.limit]
        data["val"] = data["val"][: max(50, args.limit // 8)]
    t0 = time.time()
    enc = {k: [encode(tok, ex, args.max_len) for ex in v] for k, v in data.items()}
    if main_proc:
        n_tok = sum(len(x["input_ids"]) for x in enc["train"])
        print(f"encoded in {time.time() - t0:.0f}s; train {len(enc['train'])} ex, {n_tok / 1e6:.1f}M tokens; "
              f"val {len(enc['val'])}; test {len(enc['test'])}", flush=True)

    model = JobFit(args.model).to(device)
    model.enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    core = model.module if ddp else model
    if args.optim == "adamw8bit":  # 8-bit optimizer state: ModernBERT-large OOMs on a 16 GB T4 with fp32 AdamW
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    per_epoch = math.ceil(len(batches(enc["train"], args.tokens_per_batch, True, 0)) / world / args.accum)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * per_epoch * args.epochs), per_epoch * args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    pad_id = tok.pad_token_id

    best, log = -1.0, []
    for epoch in range(args.epochs):
        all_b = batches(enc["train"], args.tokens_per_batch, True, args.seed + epoch)
        mine = all_b[rank::world][: len(all_b) // world]  # equal step counts on every rank
        t_ep, running = time.time(), 0.0
        for step, b in enumerate(mine):
            ids, mask, labels, lines, soft = collate([enc["train"][i] for i in b], pad_id)
            labels = {h: v.to(device) for h, v in labels.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out_ = model(ids.to(device), mask.to(device))
            loss = loss_fn(out_, labels, lines, args.w_line, soft, args.alpha) / args.accum
            scaler.scale(loss).backward()
            running += loss.item() * args.accum
            if (step + 1) % args.accum == 0 or step == len(mine) - 1:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if main_proc and (step + 1) % 50 == 0:
                el = time.time() - t_ep
                print(f"epoch {epoch + 1} step {step + 1}/{len(mine)} loss {running / 50:.3f} "
                      f"{el / (step + 1):.2f}s/step eta {el / (step + 1) * (len(mine) - step - 1) / 60:.0f}m", flush=True)
                running = 0.0
        if main_proc:
            res, _ = evaluate(core, enc["val"], args, device, pad_id)
            ep_min = (time.time() - t_ep) / 60
            log.append({"epoch": epoch + 1, "minutes": round(ep_min, 1), "val": res})
            print(f"epoch {epoch + 1} done in {ep_min:.1f} min; val {json.dumps(res)}", flush=True)
            if res["verdict"]["macro_f1"] > best:
                best = res["verdict"]["macro_f1"]
                torch.save(core.state_dict(), out / "best.pt")
        if ddp:
            dist.barrier()

    if main_proc:
        core.load_state_dict(torch.load(out / "best.pt", map_location=device))
        _, raw = evaluate(core, enc["val"], args, device, pad_id)
        temps = {h: round(fit_temperature(lg, y), 4) for h, (lg, y) in raw.items()}
        test, _ = evaluate(core, enc["test"], args, device, pad_id, temps)
        v1 = [i for i, ex in enumerate(data["test"]) if ex["cv_id"] <= V1_MAX_CV and ex["kind"] in V1_KINDS]
        test_v1, _ = evaluate(core, [enc["test"][i] for i in v1], args, device, pad_id, temps)
        metrics = {"model": args.model, "args": vars(args), "epochs": log, "temperatures": temps, "test": test,
                   "test_v1": {**test_v1, "n": len(v1)},
                   "params_m": round(sum(p.numel() for p in core.parameters()) / 1e6, 1)}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        tok.save_pretrained(out / "tokenizer")
        print("TEST", json.dumps(test), flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
