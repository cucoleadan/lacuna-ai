"""Trained run -> ONNX for the browser (ONNX Runtime Web / Transformers.js, WebGPU).

  python scripts/export_onnx.py d68-v3-s0 --out export/lacuna-celer
  python scripts/export_onnx.py mblarge-v3-e6 --out export/lacuna-altus

Writes model.onnx (fp32), model_fp16.onnx, model_q4f16.onnx (4-bit MatMul weights, fp16 elsewhere), the tokenizer,
and lacuna.json (labels, per-head temperatures, max length) that the extension reads. Inputs: input_ids,
attention_mask (int64, [batch, seq]). Outputs (fp32): verdict, location, seniority, must_met logits [batch, k];
line_met [batch, seq, 4] and line_must [batch, seq] per token, averaged over each job line's tokens in JS (see train.py).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import train as T  # noqa: E402

OUTPUTS = ["verdict", "location", "seniority", "must_met", "line_met", "line_must"]


class Wrapped(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        heads, met, must = self.m(input_ids, attention_mask)
        return tuple(heads[h] for h in OUTPUTS[:4]) + (met, must)


def export(model, path):
    ids = torch.randint(5, 1000, (1, 256))
    torch.onnx.export(Wrapped(model), (ids, torch.ones_like(ids)), str(path), dynamo=False, opset_version=17,
                      input_names=["input_ids", "attention_mask"], output_names=OUTPUTS,
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                    "line_met": {0: "batch", 1: "seq"}, "line_must": {0: "batch", 1: "seq"},
                                    **{h: {0: "batch"} for h in OUTPUTS[:4]}})


def to_fp16(src, dst):
    """fp32 graph -> fp16, rotary embedding kept fp32 (positions up to 4,096 lose precision in fp16). The converter leaves
    some fp32 tensors feeding fp16 ops (rotary outputs, bias-free LayerNorm constants); cast those to fp16."""
    import onnx
    from onnx import TensorProto, helper, shape_inference
    from onnxconverter_common import float16
    m = onnx.load(str(src))
    keep = {n.name for n in m.graph.node if "rotary_emb" in n.name}
    h = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True, node_block_list=list(keep))
    inf = shape_inference.infer_shapes(h)
    ty = {v.name: v.type.tensor_type.elem_type for v in list(inf.graph.value_info) + list(inf.graph.input)}
    ty.update({i.name: i.data_type for i in h.graph.initializer})
    ty.update({n.output[0]: n.attribute[0].t.data_type for n in h.graph.node if n.op_type == "Constant"})
    nodes = []
    for n in h.graph.node:
        ts = [ty.get(x) for x in n.input if x]
        if n.name not in keep and TensorProto.FLOAT16 in ts and TensorProto.FLOAT in ts and n.op_type != "Cast":
            for k, x in enumerate(n.input):
                if x and ty.get(x) == TensorProto.FLOAT:
                    nodes.append(helper.make_node("Cast", [x], [f"{x}__fp16_{len(nodes)}"], to=TensorProto.FLOAT16,
                                                  name=f"to_fp16_{len(nodes)}"))
                    n.input[k] = nodes[-1].output[0]
        nodes.append(n)
    del h.graph.node[:]
    h.graph.node.extend(nodes)
    onnx.save(h, str(dst))


def quantize_q4(src, dst):
    """4-bit MatMul weights (block 32) on the fp16 export: q4f16."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    m = onnx.load(str(src))
    heads = [n.name for n in m.graph.node if "/line_" in n.name]  # tiny per-line heads stay fp16
    q = MatMulNBitsQuantizer(m, block_size=32, is_symmetric=True, nodes_to_exclude=heads)
    q.process()
    onnx.save(q.model.model, str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-q4", action="store_true")
    ap.add_argument("--src", default="", help="run output dir (default data/runs/<run>/out)")
    ap.add_argument("--labels", default=str(ROOT / "data" / "dataset" / "labels.json"))
    a = ap.parse_args()
    src, out = Path(a.src) if a.src else ROOT / "data" / "runs" / a.run / "out", Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((src / "metrics.json").read_text())
    model = T.LacunaModel(m["model"])
    model.load_state_dict(torch.load(src / "best.pt", map_location="cpu"))
    model.eval()
    export(model, out / "model.onnx")
    to_fp16(out / "model.onnx", out / "model_fp16.onnx")
    if not a.skip_q4:
        quantize_q4(out / "model_fp16.onnx", out / "model_q4f16.onnx")
    shutil.copytree(src / "tokenizer", out, dirs_exist_ok=True)
    labels = json.loads(Path(a.labels).read_text())
    (out / "lacuna.json").write_text(json.dumps({
        "base_model": m["model"], "max_len": m["args"]["max_len"], "temperatures": m["temperatures"],
        "labels": {**labels, "must_met": ["no", "yes"]}, "outputs": OUTPUTS}, indent=1))
    for f in sorted(out.glob("*.onnx")):
        print(f"{f.name}: {f.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
