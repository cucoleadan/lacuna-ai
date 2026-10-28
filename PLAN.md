# jobfit: plan

## Goal

Give it a CV and a job posting; get back, in one forward pass with calibrated probabilities:

- **Verdict:** Apply / Stretch / Skip ("would this CV likely get an interview?")
- **Location eligible:** yes / no / unclear (CV location vs on-site, hybrid or region-restricted remote). Added 2026-10-22: location limits (on-site elsewhere, region-restricted remote) are among the most common deal-breakers in practice.
- **Must-haves met:** yes/no
- **Seniority:** below / at / above the role
- **Top gaps:** the 2-3 job requirements the CV is weakest on (each requirement line scored against the CV)

It has to work for any English CV, not one person's, and run in the browser.

Model: ModernBERT-base (149M, 8k context) with small heads on top, trained and served at **4,096 tokens** (full CV + full posting, no truncation). Ship the q4f16 ONNX build if it holds accuracy. ModernBERT-large (395M) only if base falls short.

## Measurements (2026-10-22)

**Token counts** (one real 2-page CV + 200 random English postings, whitespace collapsed):

| | Tokens |
|---|---|
| A 2-page CV | ~1,700 |
| Posting, median / p90 | ~900 / ~1,600 (boilerplate stripping saves only 8-10%) |
| CV + posting, median / p90 / max | ~2,550 / ~3,250 / under 4,096 |

2,048 fits under 6% of pairs, 3,072 about 85%, 4,096 all of them.

**Browser speed** (`web/speedtest.html`, one forward pass, Chrome 155 on Windows, RTX 3060 6 GB over WebGPU):

| Tokens | ModernBERT-base fp16 (300 MB) | q4f16 (140 MB) |
|---|---|---|
| 2,048 | 0.57 s | 0.56 s |
| 3,072 | 0.95 s | 0.93 s |
| 4,096 | 1.43 s | 1.39 s |

Load from cache 0.7-1.7 s. One pass gives all heads, so this is the per-page time: inside the 2-3 s target. Older M1 / weak laptop GPUs may be 1.5-2× slower.

- mmBERT-base (Laya-multilingual's base) failed on WebGPU with "memory access out of bounds", most likely its 256k-token vocabulary (~400 MB embedding table) exceeding WebGPU buffer limits. Fixable with CPU-side embedding lookup (as some Laya browser ports do); multilingual is v2.
- On Windows, Chrome picked the integrated AMD GPU over the RTX 3060 until Chrome was set to High performance in Windows Graphics settings. The extension must request `powerPreference: "high-performance"`, and the docs should mention the Windows setting.

## Data we have

From the author's local job-search archive (read only; another process may be writing to it):

| Source | What | Use here |
|---|---|---|
| `pages.db` → `postings` | 136,095 scraped postings, 108,216 English, about 4.8k chars each, Markdown. Ashby 57k, Greenhouse 34k, Personio 18k, SmartRecruiters 18k, Workable 3k, Recruitee 3k, Lever 2k | Job side of training pairs |

No CVs. We generate them (below).

## Data rules

- Scraped postings, and anything that contains their text, stay local or in a **private** Kaggle dataset. Never committed, never published. `.gitignore` enforces this for `data/`.
- Strip recruiter names, emails and phone numbers from postings before labelling.
- No scraped or real CVs in training. Real CVs (with consent) are test-only and never committed.
- Publishable: code, rubric, prompts, synthetic CVs, synthetic demo postings, model weights, data card, eval results.

## Labeller

DeepSeek V4.1 Flash (open weights, MIT), served through a commercial OpenAI-compatible API. Decided 2026-10-22; replaces GLM 5.3 Flash on Fireworks. Endpoint and account settings live in the gitignored `local.json`.

- Model licence: MIT.
- Price (provider's list price, 2026-10-22): $0.30 / 1M input, $1.20 / 1M output; cached input is cheaper but needs requests of 4,096+ tokens, so our ~3k-token pairs get none.
- Not Claude: Anthropic's Usage Policy bans training models on Claude outputs without authorization. Not Gemini or Google AI Pro via `agy` either (ToS, quotas).
- Settings: thinking off, strict JSON output, CV first in the prompt so repeated CVs hit the cached-input rate.
- Thinking is **off** (zero reasoning tokens). With thinking on and low effort, a one-line date question used the full 1,500-token budget without answering, so it is never enabled for labelling.
- Thinking off slips on arithmetic, so the labeller never computes years: it extracts each CV's roles with start/end dates as JSON once per CV, and Python computes total years and seniority, which go into the pair prompt.
- Hard cap: the labelling script sums the token usage each response reports and stops at $30; a provider-side budget alert backs it up.

## Pipeline

1. **Snapshot + clean postings.** Copy `pages.db` with `.backup` when no harvest is running. Keep English, 800-20k chars, dedupe by sha256 and near-duplicates (reposts). Strip personal data. Tag field and seniority.
2. **Synthetic CVs (~5k).** Sample a believable career path of 2-4 postings; the labeller writes the CV of someone who held them. Vary format, length and quality (vague bullets, typos, PDF-extraction mess, career changers, gaps, students, weak CVs).
3. **Pairing (~8k pairs).** Per CV: ~40% near matches (embedding retrieval), ~40% hard negatives, ~20% random. Hard negatives are split so every head gets its own cases: seniority gaps (2+ years under, at level, over-qualified), a missing key skill or gate, and location limits (on-site elsewhere, region-restricted remote).
4. **Rubric + labelling.** Candidate-side rubric: deal-breakers vs nice-to-haves; meeting ~60-70% of the list still counts as Apply. Pilot 50 pairs, check cost and output. Labeller check before the full run against real human judgments, plus ~100 hand-labelled synthetic pairs.
5. **Split by CV.** Test CVs never appear in training.
6. **Train a size sweep** on Kaggle free GPUs (2×T4), same data and script for every size. First a 500-pair timing run to replace the estimates.

   | Model | Params | Est. per run | Role |
   |---|---|---|---|
   | Ettin-32M | 32M | ~10-15 min | fast tier (weak GPUs, CPU fallback) |
   | Ettin-68M | 68M | ~20-30 min | fast tier candidate |
   | ModernBERT-base | 149M | ~1-2 h | quality tier (main model) |
   | ModernBERT-large | 395M | ~4-6 h | upper bound and distillation teacher |

   One sweep ≈ 7-9 GPU hours of the 30 h/week.
6c. **Bigger, server-side (conditional on the sweep).** If large clearly beats base, train Ettin-1B (~10 GPU h). If the curve flattens, try a small generative LLM instead (~2-4B, LoRA, ~10-20 GPU h at 4k on T4s) that also writes the reason and missing requirements, like the teacher. Hosting only beats DeepSeek's ~$0.0015 per job above ~300-600 jobs/hour sustained (GPU at $0.3-1/h); serverless scale-to-zero narrows that.
6b. **Distil.** The large model labels tens of thousands of extra CV × posting pairs (82k postings, free, no API cost); the small models train on DeepSeek labels plus those. Small models usually gain most. Ceiling stays DeepSeek's label quality.
7. **Evaluate.** Each head on its own (seniority also on pairs where location is `yes`, so location cannot hide weak seniority), agreement with the labeller's labels, agreement with human hand labels, calibration (ECE), results on real CVs, plus failure examples.
8. **Ship.**
   - Two tiers on Hugging Face (ONNX q4f16): fast (Ettin, ~30-70 MB) and quality (ModernBERT-base, ~140 MB), with a model card including the size / accuracy / speed curve. The extension picks the tier from the hardware.
   - Browser demo with Transformers.js: paste CV and job posting, built-in synthetic examples.
   - CV input goes through the same `normalize_cv` as training (port `scripts/cvtext.py` to JS; HTML via the DOM's `innerText`).
   - **Display rule for location (no retraining):** show "Not eligible (location)" and a `skip` only when calibrated P(location = no) ≥ ~0.8; below that show "Location unclear, check the posting" and cap the verdict at `stretch`. Headers are often stricter than real hiring (e.g. "US / EMEA – Remote" lists, EU countries, "or fully remote"). Tune the threshold on the author's private real-CV check and the human set.
   - v2: Chrome extension that scores the job page you are on.
   - Write-up.

## Paper track

Target: workshop paper (e.g. NLP4HR at an ACL-family conference) and an arXiv preprint.

- **Tracking:** `DECISIONS.md` (why), `EXPERIMENTS.md` (every run and number), publishable metrics in `results/`.
- **Contributions to argue:** target-anchored synthetic CV generation with controlled coverage; "labeller judges facts, Python applies thresholds" for consistent rubric labels; multi-head on-device model (verdict, location, seniority, per-line gaps) in ~1 s in a browser; size / accuracy / speed curve; cost (< $30 API, free GPUs).
- **Evaluation needed for a paper:** 2-3 annotators on the same ~150 real pairs (inter-annotator κ); teacher-vs-human and student-vs-human agreement; per-head metrics and calibration; ablations (without target anchoring, without domestic override, without line head, distillation on/off).
- **Rigour gaps to close before writing (2026-10-23):**
  1. *Seeds:* every reported number so far is one seed. Run Ettin-68M (v2, 6 epochs) with seeds 0, 1, 2 and report mean ± spread; treat differences smaller than the spread as noise.
  2. *Confounds:* re-score every checkpoint on the frozen v1 test examples (`scripts/eval_v1.py`, local, no GPU quota) so v1 and v2 models share one test set; Ettin-68M on v2 at 3 epochs separates "more data" from "more epochs"; ModernBERT-large alone uses 8-bit AdamW (state it).
  3. *Learning rate:* one LR per model was picked up front. LR check on Ettin-68M (v2, 3 epochs: 5e-5, 1e-4, 2e-4); if the spread is large, repeat for the model we ship.
  4. *Human ground truth:* the ~150-pair human set above; scores against DeepSeek labels only show agreement with the teacher.
  5. *Selection on validation only:* choose models, epochs and LRs by validation scores; the test set is read once per final model.
- **Run queue, done 2026-10-23 (all on v2, results in EXPERIMENTS.md):** LR check, Ettin-68M 3 seeds, Ettin-32M, ModernBERT-base and ModernBERT-large at 6 epochs. Gaps 1-3 and 5 are closed except large's shared-test score (local CPU re-score pending). Outcome: Ettin-68M is the small-tier model (ModernBERT-base is no better at 2× the cost); ModernBERT-large is the teacher.
- **Next (from 2026-10-25):**
  1. *v3 = canonical CV text* (running): Ettin-68M × 3 seeds on `normalize_cv` CVs vs the v2 mean, then the author's real-CV check (`student_check.py --normalize`).
  2. *Distil:* large labels unlabelled CV × posting pairs (free, Kaggle); Ettin-68M trains on DeepSeek + large's soft labels.
  3. *Ship:* ONNX q4f16 export, browser demo (JS `normalize_cv`, location display rule), model card.
  4. *Human set* (~150 pairs, 2-3 annotators) for the paper.
- **Publishing path:** arXiv preprint (cs.CL, cross-list cs.LG) once results, demo and model are public; Substack write-up the same week; submit to the next NLP4HR (or similar) workshop, which usually allows prior arXiv posting (check each call).
- **arXiv endorsement:** first-time submitters in cs.CL need an endorser. When the draft and demo exist: shortlist authors of closely related work (NLP4HR, LLM-labelled data / distillation for classification, ModernBERT / Ettin downstream use), checked via "Which authors of this paper are endorsers?" on their arXiv pages; prefer PhD students, postdocs, early-career researchers and people in the author's network. One personal email each: why them, abstract, links to PDF / model / demo, the endorsement code, "a two-minute click, not a review".
- **Limitations to state:** synthetic training CVs; LLM teacher is the ceiling; English only; scraped postings cannot be redistributed (release posting URLs, code, synthetic CVs, prompts, labels, weights).

## Budget (hard cap $30)

| Item | Estimate |
|---|---|
| Synthetic CVs (~5k) | ~$3 |
| CV fact extraction (dates, languages, location) | ~$1-2 |
| Labelling (~8k pairs, ~2.6k tokens each, measured) | ~$8 |
| Training, hosting | $0 (Kaggle 2×T4, 30 GPU h/week, phone-verified account; Hugging Face) |

Leaves room to scale to ~20k pairs (~$17-20) inside the cap. The 50-pair pilot confirms real cost before any full run.

## Open questions

- Which job fields matter most (to weight the sample)?
- Training at 4,096 tokens on 2×T4: check memory and time per run in the first Kaggle session (gradient checkpointing, small batches).
- Speed on an M-series Mac and a weaker laptop GPU (same `web/speedtest.html`).
- When to make this repo public.
