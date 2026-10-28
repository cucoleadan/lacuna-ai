# jobfit: experiment log

Every run that produced a number: what ran, on what data, with what settings, and the result. Paper tables and figures are built from this file and the files it points to. `DECISIONS.md` says *why*; this file says *what happened*.

Conventions:
- Check each run's log line `encoded ...; train N ex` against the dataset version it was meant to use: Kaggle can serve the previous dataset version for a few minutes after an upload reports `ready`.
- Dates are UTC days. Costs are list price unless noted.
- Raw outputs live in `data/` (gitignored: contains posting text). Publishable copies of metrics go in `results/<run>/metrics.json` (no posting or CV text).
- Each training run records: model id, dataset version, seed, all CLI args, GPU hours, per-epoch validation and final test metrics.

## Data runs

| Date | Run | Input | Output | Cost | Notes |
|---|---|---|---|---|---|
| 2026-10-22 | token count | one real CV + 200 postings | CV ~1.7k tokens, pair median ~2.55k, p90 ~3.25k, max < 4,096 | $0 | 3 tokenizers within ~5% after whitespace collapse |
| 2026-10-22 | pilots 1-6 | 20 CVs × 4 pairs each | positive share 1% → 34% | ~$0.70 | see DECISIONS.md, "pilot" entries |
| 2026-10-22 | full run | 2,000 CVs × 4 pairs | 1,973 CVs, 7,891 labelled pairs; verdict skip 5,581 / apply 1,411 / stretch 899; seniority above only 325 | CVs $2.59, labels $10.64, total spend to date $14.33 | `data/full_run.sh`; first pass stopped at 4,382 labels when the 1 h API token expired (401s), resumed after adding token refresh. ~2.76k input / 434 output tokens per pair |
| 2026-10-22 | dataset v1 | 7,891 pairs | split by CV: train 6,315 / val 788 / test 788; 99% of requirements (71,681/72,305) matched to a posting line | $0 | `scripts/build_dataset.py` |
| 2026-10-22 | dataset v2 | v1 + 244 new CVs (976 standard pairs) + 3,833 targeted pairs (`over`, `stretch`) | 2,217 CVs, 12,699 pairs; train 10,171 / val 1,270 / test 1,258 (v1 split frozen); seniority `above` 325 → 1,033, verdict `stretch` 899 → 1,913 | CVs $0.34, labels $6.37; project total $21.03 | `data/v2_run.sh`; 6 CVs and 1 label lost to API errors |

## Browser speed

| Date | Model | Precision | Hardware | 2,048 tok | 3,072 tok | 4,096 tok |
|---|---|---|---|---|---|---|
| 2026-10-22 | ModernBERT-base | fp16 | RTX 3060 6 GB, Chrome 155, Windows, WebGPU | 0.57 s | 0.95 s | 1.43 s |
| 2026-10-22 | ModernBERT-base | q4f16 | same | 0.56 s | 0.93 s | 1.39 s |
| 2026-10-22 | mmBERT-base | fp16 | same | failed: "memory access out of bounds" (256k vocabulary) | | |

## Training runs

| Date | Run | Model | Data | Epochs | GPU | Time | Result |
|---|---|---|---|---|---|---|---|
| 2026-10-22 | timing-base | ModernBERT-base | 475 real CV × posting pairs, **random labels** | 1 | Kaggle 2×T4 | 1.3 min/epoch (0.9M tokens), 1.16 s/step, ~3 min total | fits at 4,096 tokens; metrics meaningless (random labels); weights discarded |
| 2026-10-22 | ettin32 | Ettin-32M (31.9M params) | v1, lr 1e-4, seed 0 | 3 | Kaggle 2×T4 | ~4-5 min/epoch, 18 min total | test: verdict acc 0.789 / macro-F1 0.536; location 0.829 / 0.824; seniority 0.748 / 0.508; must-haves met 0.769 / 0.759; requirement lines 0.879 / 0.621. ECE ≤ 0.04 after temperature scaling. Val still rising at epoch 3 → likely undertrained. `results/ettin32/` |
| 2026-10-22 | ettin68 | Ettin-68M (68.2M params) | v1, lr 1e-4, seed 0 | 3 | Kaggle 2×T4 | ~10-12 min/epoch, 38 min total | test: verdict acc 0.808 / macro-F1 0.589; location 0.846 / 0.839; seniority 0.751 / 0.511; must-haves met 0.787 / 0.779; requirement lines 0.891 / 0.661. ECE ≤ 0.041. Beats 32M on every head; seniority flat. Val verdict F1 still rising at epoch 3. `results/ettin68/` |
| 2026-10-22 | mbbase | ModernBERT-base (149M params) | v1, lr 5e-5, seed 0 | 3 | Kaggle 2×T4 | ~18-21 min/epoch, 68 min total | test: verdict acc 0.797 / macro-F1 0.550; location 0.826 / 0.819; seniority 0.728 / 0.495; must-haves met 0.770 / 0.763; requirement lines 0.882 / 0.630. ECE ≤ 0.053. **Below Ettin-68M on every head**: slower start (val location F1 0.48 after epoch 1), likely undertrained at 3 epochs / lr 5e-5. `results/mbbase/` |
| 2026-10-22 | mblarge | ModernBERT-large (395M) | v1, lr 3e-5, 8,192 tokens/GPU/step | - | Kaggle 2×T4 | failed after 3 min | CUDA out of memory on the first step (14.5 GB used of 15 GB). Rerun on v2 as `mblarge-v2` with 4,096 tokens/step × accum 8 (same effective batch): OOM again at the first optimizer step (fp32 AdamW state ≈ 6.3 GB/GPU). Third try: `--optim adamw8bit` (bitsandbytes) + `expandable_segments`; first launch got the previous dataset version (stale train.py, 'unrecognized arguments'), relaunched a few minutes later. Fourth launch trained at 1.12 s/step (~61 min/epoch) but stopped at step ~680 with DDP "parameters that were not used in producing loss": one-example batches sometimes have no requirement lines, so the line heads got no gradient. Fixed with zero-weight terms in the loss (numerically identical); relaunched |
| 2026-10-22 | ettin68-v1e8 (launched as `ettin68-v2e8`) | Ettin-68M | **v1** (not v2: Kaggle served the previous dataset version; log shows 6,315 train / 788 test), lr 1e-4, seed 0 | 8 | Kaggle 2×T4 | ~13 min/epoch, 105 min total | test (v1 test, 788 ex): verdict acc 0.802 / macro-F1 0.626; location 0.883 / 0.875; seniority 0.735 / 0.576; must-haves met 0.792 / 0.778; requirement lines 0.894 / 0.681. vs `ettin68` (same data, 3 epochs): verdict +0.037, seniority +0.065, location +0.036 → **an epochs-only ablation**. Val verdict F1 plateaus from epoch 5 (best epoch 7). Found 2026-10-23 when the v1 re-score matched this run's test numbers exactly. `results/ettin68-v1e8/` |
| 2026-10-22 | mblarge-v2 | ModernBERT-large (395M params) | **v2**, lr 3e-5, 4,096 tokens/step × accum 8, 8-bit AdamW, seed 0 | 3 | Kaggle 2×T4 | ~63-72 min/epoch, 221 min total | test (v2 test, 1,258 ex): verdict acc 0.818 / macro-F1 0.674; location 0.907 / 0.906; seniority 0.785 / 0.708; must-haves met 0.816 / 0.803; requirement lines 0.900 / 0.698. ECE ≤ 0.045. **Best on every head**, still rising at epoch 3 (val verdict 0.56 → 0.64 → 0.69). Kaggle status ERROR after the test: rank 1 hit the 10-min NCCL timeout while rank 0 evaluated; metrics and weights were already saved. Fixed with a 1 h timeout. `results/mblarge-v2/` |
| 2026-10-23 | e68-v2-e3-lr5e5 | Ettin-68M | v2, lr 5e-5, seed 0 | 3 | Kaggle 2×T4 | 53 min training | val (best epoch 3) F1: verdict 0.647, seniority 0.677, location 0.894, must 0.830, lines 0.651. no test_v1 (launched before train.py reported it). Train 10,171 ex (v2 confirmed in log). `results/e68-v2-e3-lr5e5/` |
| 2026-10-23 | e68-v2-e3-lr1e4 | Ettin-68M | v2, lr 1e-4, seed 0 | 3 | Kaggle 2×T4 | 55 min training | val (best epoch 3) F1: verdict 0.661, seniority 0.700, location 0.893, must 0.849, lines 0.679. test_v1 F1: verdict 0.633, seniority 0.577, location 0.877, must 0.816, lines 0.677. Train 10,171 ex (v2 confirmed in log). `results/e68-v2-e3-lr1e4/` |
| 2026-10-23 | e68-v2-e3-lr2e4 | Ettin-68M | v2, lr 2e-4, seed 0 | 3 | Kaggle 2×T4 | 57 min training | val (best epoch 3) F1: verdict 0.665, seniority 0.682, location 0.899, must 0.848, lines 0.692. test_v1 F1: verdict 0.634, seniority 0.585, location 0.866, must 0.812, lines 0.684. Train 10,171 ex (v2 confirmed in log). `results/e68-v2-e3-lr2e4/` |
| 2026-10-23 | e68-v2-e6-s0 | Ettin-68M | v2, lr 1e-4, seed 0 | 6 | Kaggle 2×T4 | 102 min training | val (best epoch 4) F1: verdict 0.689, seniority 0.640, location 0.896, must 0.852, lines 0.699. test_v1 F1: verdict 0.638, seniority 0.577, location 0.878, must 0.825, lines 0.695. Train 10,171 ex (v2 confirmed in log). `results/e68-v2-e6-s0/` |
| 2026-10-23 | e68-v2-e6-s1 | Ettin-68M | v2, lr 1e-4, seed 1 | 6 | Kaggle 2×T4 | 106 min training | val (best epoch 5) F1: verdict 0.682, seniority 0.697, location 0.906, must 0.855, lines 0.688. test_v1 F1: verdict 0.671, seniority 0.626, location 0.876, must 0.818, lines 0.689. Train 10,171 ex (v2 confirmed in log). Seed 0 vs 1 differ by 0.033 verdict and 0.049 seniority on test_v1: seed noise is as large as most effects measured so far. `results/e68-v2-e6-s1/` |
| 2026-10-23 | e68-v2-e6-s2 | Ettin-68M | v2, lr 1e-4, seed 2 | 6 | Kaggle 2×T4 | 108 min training | val (best epoch 6) F1: verdict 0.690, seniority 0.707, location 0.908, must 0.836, lines 0.693. test_v1 F1: verdict 0.632, seniority 0.591, location 0.891, must 0.810, lines 0.694. Train 10,171 ex (v2 confirmed in log). Seeds 0-2 on test_v1, mean ± sd: verdict 0.647 ± 0.021, seniority 0.598 ± 0.025, location 0.882 ± 0.008, must 0.818 ± 0.008, lines 0.692 ± 0.003. `results/e68-v2-e6-s2/` |
| 2026-10-23 | ettin32-v2-e6 | Ettin-32M | v2, lr 1e-4 | 6 | Kaggle 2×T4 | 45 min training | val (best epoch 4) F1: verdict 0.657, seniority 0.633, location 0.879, must 0.830, lines 0.653. test_v1 F1: verdict 0.608, seniority 0.566, location 0.853, must 0.789, lines 0.658 (v1, 3 epochs: 0.536 / 0.508). Train 10,171 ex (v2 confirmed in log). `results/ettin32-v2-e6/` |
| 2026-10-23 | mblarge-v2-e6 | ModernBERT-large | v2, lr 3e-5, 8-bit AdamW, 4096 tok × accum 8 | 6 | Kaggle 2×T4 | 410 min training | val (best epoch 5) F1: verdict 0.696, seniority 0.761, location 0.899, must 0.856, lines 0.708. v2 test F1: verdict 0.716, seniority 0.731 (3 epochs: 0.674 / 0.708). test_v1 (re-scored locally on CPU, 81 min, 2026-10-25) F1: verdict 0.733, seniority 0.683, location 0.886, must 0.809, lines 0.708. Train 10,171 ex (v2 confirmed in log). `results/mblarge-v2-e6/` |
| 2026-10-23 | mbbase-v2-e6 | ModernBERT-base | v2, lr 5e-5 | 6 | Kaggle 2×T4 | 205 min training | val (best epoch 6) F1: verdict 0.676, seniority 0.699, location 0.897, must 0.837, lines 0.672. test_v1 F1: verdict 0.648, seniority 0.618, location 0.861, must 0.802, lines 0.672: inside the Ettin-68M 3-seed range at 2× the parameters and training time. Train 10,171 ex (v2 confirmed in log). `results/mbbase-v2-e6/` |
| 2026-10-25 | e68-v3-e6-s0 | Ettin-68M | v3, lr 1e-4, seed 0 | 6 | Kaggle 2×T4 | 114 min training | val (best epoch 4) F1: verdict 0.680, seniority 0.698, location 0.905, must 0.849, lines 0.682. test_v1 F1: verdict 0.659, seniority 0.608, location 0.876, must 0.821, lines 0.684. Train 10,171 ex, 19.2M tokens (v2: 19.3M; v3 CVs are 1% shorter, so v3 confirmed). `results/e68-v3-e6-s0/` |
| 2026-10-25 | e68-v3-e6-s1 | Ettin-68M | v3, lr 1e-4, seed 1 | 6 | Kaggle 2×T4 | 111 min training | val (best epoch 6) F1: verdict 0.671, seniority 0.702, location 0.911, must 0.839, lines 0.690. test_v1 F1: verdict 0.652, seniority 0.622, location 0.875, must 0.785, lines 0.689. Train 10,171 ex, 19.2M tokens (v2: 19.3M; v3 CVs are 1% shorter, so v3 confirmed). `results/e68-v3-e6-s1/` |
| 2026-10-25 | e68-v3-e6-s2 | Ettin-68M | v3, lr 1e-4, seed 2 | 6 | Kaggle 2×T4 | 116 min training | val (best epoch 5) F1: verdict 0.683, seniority 0.701, location 0.912, must 0.839, lines 0.685. test_v1 F1: verdict 0.632, seniority 0.619, location 0.882, must 0.811, lines 0.688. Train 10,171 ex, 19.2M tokens (v2: 19.3M; v3 CVs are 1% shorter, so v3 confirmed). Seeds 0-2, mean ± sd: verdict 0.648 ± 0.014, seniority 0.616 ± 0.007, location 0.878 ± 0.004, must 0.805 ± 0.018, lines 0.687 ± 0.003 (v2: 0.647 / 0.598 / 0.882 / 0.818 / 0.692). `results/e68-v3-e6-s2/` |
| 2026-10-26 | teach-v3 | ModernBERT-large (mblarge-v3-e6) | scores v3 train (10,171) + 13,970 unlabelled train-CV pairs (`scripts/distil_pairs.py`) | - | Kaggle 2×T4 | 39 min | Calibrated probabilities per head and per job line, `train/teach.py`. No API cost. |
| 2026-10-26 | d68-v3-s0 | Ettin-68M, distilled | v3 + teach-v3 soft targets, alpha 0.5, lr 1e-4, seed 0 | 3 | Kaggle 2×T4 | 119 min training | val (best epoch 3) F1: verdict 0.714, seniority 0.747, location 0.910, must 0.860, lines 0.701. test_v1 F1: verdict 0.672, seniority 0.678, location 0.891, must 0.817, lines 0.707. Train 24,141 ex, 45.8M tokens. `results/d68-v3-s0/` |
| 2026-10-26 | d68-v3-s1 | Ettin-68M, distilled | as s0, seed 1 | 3 | Kaggle 2×T4 | 127 min training | val (best epoch 3) F1: verdict 0.699, seniority 0.746, location 0.919, must 0.850, lines 0.700. test_v1 F1: verdict 0.646, seniority 0.648, location 0.887, must 0.817, lines 0.703. `results/d68-v3-s1/` |
| 2026-10-26 | d68-v3-s2 | Ettin-68M, distilled | as s0, seed 2 | 3 | Kaggle 2×T4 | 126 min training | val (best epoch 2) F1: verdict 0.711, seniority 0.742, location 0.910, must 0.856, lines 0.703. test_v1 F1: verdict 0.657, seniority 0.663, location 0.888, must 0.816, lines 0.698. Seeds 0-2, mean ± sd: verdict 0.658 ± 0.014, seniority 0.663 ± 0.015, location 0.889 ± 0.002, must 0.816 ± 0.001, lines 0.703 ± 0.005 (undistilled v3: 0.648 / 0.616 / 0.878 / 0.805 / 0.687). `results/d68-v3-s2/` |
| 2026-10-25 | mblarge-v3-e6 | ModernBERT-large | v3, lr 3e-5, 8-bit AdamW, 4096 tok × accum 8 | 6 | Kaggle 2×T4 | 420 min training | val (best epoch 5) F1: verdict 0.711, seniority 0.745, location 0.904, must 0.872, lines 0.713. test_v1 F1: verdict 0.644, seniority 0.674, location 0.887, must 0.820, lines 0.709 (v2 run: 0.733 / 0.683). Higher val verdict than the v2 run but 0.089 lower test_v1 verdict: one seed each, so large's verdict head is at least this noisy. Train 10,171 ex, 19.2M tokens (v2: 19.3M; v3 CVs are 1% shorter, so v3 confirmed). `results/mblarge-v3-e6/` |

### Findings so far (shared v1 test set, 788 examples, macro-F1; one seed unless noted)

Seed noise (Ettin-68M, v2, 6 epochs, seeds 0-2): sd 0.021 verdict, 0.025 seniority, 0.008 location and must-haves, 0.003 lines. Single-run differences below ~0.04 on verdict or seniority are not evidence.

| Model | Data | Epochs | Verdict | Seniority | Location | Must-haves | Req. lines |
|---|---|---|---|---|---|---|---|
| Ettin-68M | v1 | 3 | 0.589 | 0.511 | 0.839 | 0.779 | 0.661 |
| Ettin-68M | v1 | 8 | 0.626 | 0.576 | 0.875 | 0.778 | 0.681 |
| Ettin-68M | v2 | 3 | 0.633 | 0.577 | 0.877 | 0.816 | 0.677 |
| Ettin-32M | v2 | 6 | 0.608 | 0.566 | 0.853 | 0.789 | 0.658 |
| Ettin-68M | v2 | 6 (3 seeds, mean) | 0.647 | 0.598 | 0.882 | 0.818 | 0.692 |
| ModernBERT-base | v2 | 6 | 0.648 | 0.618 | 0.861 | 0.802 | 0.672 |
| Ettin-68M | v3 | 6 (3 seeds, mean) | 0.648 | 0.616 | 0.878 | 0.805 | 0.687 |
| ModernBERT-large | v2 | 3 | 0.659 | 0.663 | 0.885 | 0.806 | 0.695 |
| ModernBERT-large | v2 | 6 | 0.733 | 0.683 | 0.886 | 0.809 | 0.708 |
| ModernBERT-large | v3 | 6 | 0.644 | 0.674 | 0.887 | 0.820 | 0.709 |
| Ettin-68M, distilled from large v3 | v3 + 14k teacher-scored | 3 (3 seeds, mean) | 0.658 | 0.663 | 0.889 | 0.816 | 0.703 |

- **Data (clean ablation, rows 1 vs 3):** targeted v2 pairs at fixed model/epochs/LR: verdict +0.044, seniority +0.066, must-haves +0.037.
- **Epochs:** on v1, 3 → 8 epochs helped (+0.037 verdict); on v2, 3 → 6 adds +0.005 verdict (within expected seed noise; seeds pending).
- **LR (Ettin-68M, v2, 3 epochs, val verdict F1):** 5e-5 0.647, 1e-4 0.661, 2e-4 0.665: flat between 1e-4 and 2e-4; 1e-4 kept.
- **Size:** ModernBERT-large leads mainly on seniority (+0.086 over Ettin-68M v2, one seed each). At 6 epochs on its own v2 test it reaches verdict 0.716, seniority 0.731 (shared-test score pending). ModernBERT-base ≈ Ettin-68M (within seed noise) at 2× the cost, so Ettin-68M is the small-tier candidate and large is the distillation teacher.
- **Canonical CV text (v3 vs v2, Ettin-68M, 3 seeds):** no change on synthetic test (verdict +0.001, seniority +0.018, must −0.013, all within ~1 sd). Expected: synthetic CVs were already clean. v3's purpose is real, messy CVs (author's real-CV check), and train/inference parity with the browser.
- **Large is noisy on verdict:** v2 vs v3 single runs differ by 0.089 test_v1 verdict while val moved the other way (+0.015). Do not rank large runs on one test number; pick the teacher on validation.
- **Seeds:** the 3-seed spread is as large as the epochs effect; the v1→v2 data gain (+0.044 verdict, +0.066 seniority) is about 2 sd and holds.
- **Distillation (Ettin-68M, 3 seeds each, v3):** seniority +0.047 (about 3 sd), lines +0.016, must +0.011, location +0.011; verdict +0.010 (within noise). Student validation verdict (0.708 mean) matches the teacher's (0.711); on test_v1 the student reaches the teacher on seniority, must-haves and lines. Seed spread on the synthetic test is unchanged (verdict sd 0.014).
- ModernBERT-large v2 `test_v1` re-scored locally on CPU with `scripts/eval_v1.py` (±0.002 vs GPU, checked on Ettin-32M).

## Planned

- **Next:** real-CV check of the distilled seeds (does the seed spread on real CVs shrink?); human eval set; ONNX export of the distilled student.

- Size sweep on the full dataset: Ettin-32M, Ettin-68M, ModernBERT-base, ModernBERT-large; then Ettin-1B or a generative LLM depending on the curve (PLAN.md step 6c).
- Human evaluation set: 2-3 annotators label the same ~150 pairs; report inter-annotator agreement (Cohen's / Fleiss' κ) and model-vs-human, teacher-vs-human.
