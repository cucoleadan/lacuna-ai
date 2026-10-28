# jobfit: decision log

Every choice that shapes the model, its data or its labels, with the reason and what was rejected. The model card is written from this file. Newest decisions go at the bottom; superseded ones stay, marked as such.

## 2026-10-20: project shape

- **Task:** CV + job posting in, calibrated "would this CV likely get an interview?" out, for any English CV. Runs in the browser so the CV never leaves the device.
- **Data rules:** scraped postings (and anything containing their text) stay local or in a private Kaggle dataset, never published. No scraped or real CVs in training; real CVs are test-only, with consent. Synthetic CVs, prompts, code, weights and eval results are publishable.
- **Labeller (superseded 2026-10-22):** GLM 5.3 Flash on Fireworks.

## 2026-10-22: labeller

- **Decision:** DeepSeek V4.1 Flash through a commercial OpenAI-compatible API, thinking off, hard cap $30.
- **Why:** MIT licence (outputs can train another model); $0.30 / $1.20 per 1M tokens, $0.03 cached; no quota problems; the strongest cheap open model available on a platform already in use.
- **Rejected:**
  - Claude: Anthropic's Usage Policy bans training models on its outputs without authorization.
  - Gemini, including Google AI Pro through the `agy` CLI: terms restrict training on outputs; Google staff say personal AI Pro is for human-driven use and bulk automation risks account suspension; 5-hour/weekly quotas.
  - Free models on opencode Zen / OpenRouter: unpublished, shifting quotas (~1k requests/day), models rotate (mixed teachers give inconsistent labels), several train on submitted data.
  - An open model on Kaggle's free GPUs: free, but a ~30B model is a much weaker teacher (the student cannot beat its teacher) and it competes with training for the 30 GPU h/week.
  - Qwen3 235B / gpt-oss-120b in batch mode: cheaper (~$5-7) but weaker; kept as fallback.
- **Thinking off:** with thinking on, a one-line date question used the full 1,500-token budget without answering. With it off, the model slips on multi-step arithmetic, so it never computes: it extracts dated roles per CV, and Python computes total years and level (`prompts/cv_facts.md`).

## 2026-10-22: model and serving

- **Decision:** ModernBERT-base (149M) with small heads, trained and served at 4,096 tokens, shipped as q4f16 ONNX (~140 MB) on WebGPU (Transformers.js), browser demo first, Chrome extension second.
- **Why 4,096:** measured on one real CV + 200 postings: a 2-page CV is ~1,700 tokens, CV + posting median ~2,550, p90 ~3,250, all under 4,096. 2,048 would truncate 94% of pairs.
- **Why base, q4f16:** RTX 3060 in Chrome: 0.95 s at 3,072 tokens, 1.4 s at 4,096, inside the 2-3 s per page target; q4f16 matches fp16 speed at half the download.
- **Rejected:**
  - Fine-tuning Muse Glimmer 30B on Fireworks (free SFT promo): trained LoRAs can only be served on a dedicated GPU (~$7-10/h), not in a browser.
  - Laya / mmBERT-base (multilingual): mmBERT failed on WebGPU ("memory access out of bounds", most likely its 256k-token vocabulary). Multilingual is v2.
  - Jev (TypeSafe): closed hosted API.
- **Windows note:** Chrome used the integrated AMD GPU over the RTX 3060 until set to High performance in Windows Graphics settings; the app requests `powerPreference: "high-performance"`.

## 2026-10-22: training

- **Decision:** Kaggle free GPUs (2×T4, 30 GPU h/week, phone-verified account). Fallbacks: Colab free T4, a rented GPU (a few dollars).

## 2026-10-22: outputs and rubric

Full rubric: `prompts/label_pair.md`. Principle: copy a realistic first screen for any candidate, not one person's taste.

- **Outputs:** verdict (apply / stretch / skip), location eligible (yes / no / unclear), candidate vs role seniority (below / at / above), must-haves met, per-requirement met (yes / partial / no) for top gaps.
- **Location added** because location limits (on-site elsewhere, region-restricted remote) are among the most common deal-breakers in practice.
- **Apply** at two thirds of must-haves (partials count half): requirement lists are wish lists and first screens pass candidates missing a few.
- **Seniority:** 2 years short is stretch, 3+ years or 2 levels is skip; recruiters flex 1-2 years, rarely more.
- **Evidence:** a skill used in a role or project is met; skills-section-only or adjacent is partial.
- **Location unclear** does not block apply; explicit limits (often at the end of a posting) are enforced. Regions include all their countries; time-zone overlap alone is not a limit.
- **Hard gates** (skip on their own): required working-level spoken language, legally required licence or degree, security clearance.
- **Over-qualified** (2+ levels above) is stretch, not skip.

## 2026-10-22: the labeller judges facts, Python applies thresholds

- **Decision:** DeepSeek outputs only per-requirement judgments (must/nice, gate, met), location eligibility, the posting's required years and role level, and the main gap. Python computes seniority, must-have score and the verdict from those (`prompts/label_pair.md`, "Python, after labelling").
- **Why:** in the 3-job smoke test DeepSeek ignored its own thresholds twice (7 of 9 must-haves met → said stretch, rule says apply; 6 of 10 → said skip, rule says stretch). Inconsistent labels teach the student noise. Deterministic thresholds make every label follow the published rubric, and thresholds can change later without relabelling.
- **Founder titles:** founder, co-founder, CTO and CEO count as `lead`, not principal: a small-company founder title does not screen as principal, and mapping it to principal made a 10-year founder "over-qualified" for every mid role.

## 2026-10-22: relevant years, and the posting header

- **Relevant years:** each CV role gets a function (engineering, data, design, product, marketing, sales, operations, other) and each job a `role_function`; seniority compares the posting's required years with years **in that function**, and the level reached in it. Recruiters count "7+ years of software engineering" against engineering years, not a whole career.
  - Found by the first labeller check: counting total years (including earlier roles in other functions) made "7-8+ years" and Staff roles look "at level" for career changers.
- **Posting header:** the job text given to the labeller starts with Title, Company, Location and Workplace from the ATS fields. Some stored posting bodies omit the location line, which left the labeller at "unclear" for postings like "Remote, US" or "Canada (Remote)".

## 2026-10-22: labeller cost

- **Labeller accepted** after a check against real human apply/skip judgments (details kept private).
- **Cost:** ~$0.0015 per pair at list price (~3.4k tokens in, ~420 out): 8k pairs ≈ $12, the $30 cap ≈ 20k pairs.
- **No cache discount in practice:** the provider's implicit caching for open models gives 90% off cached tokens, but only for requests of at least 4,096 tokens (Preview), and hits are not guaranteed (provider docs, via Context7). Our pairs are ~3.3-3.9k, so `prompt_tokens_details` came back null on every call. Budget at full price; do not pad prompts to reach 4k.

## 2026-10-22: every head is trained, not just the verdict

- **Decision:** every pair carries a label for every head (location, seniority, must-haves, per-requirement, verdict). The verdict's "any blocker means skip" order does not stop the other heads from learning: a location-blocked pair still teaches seniority.
- **Data:** hard negatives are split per head (seniority gaps under / at / over, missing skill or gate, location limits), so seniority is not starved by location-dominated postings.
- **Evaluation:** each head is reported separately; seniority also on pairs with location `yes`.
- **Product:** the extension shows the per-head reasons ("Location ✓ · Seniority: 2 yrs short · Missing: Kubernetes"), not only the verdict.

## 2026-10-22: training data pipeline (`scripts/pipeline.py`)

- **Postings:** snapshot of `pages.db` (no harvest running), English, 800-20k chars, deduped by sha256 and by company + title (reposts): 82,118 postings. Each gets a header (Title, Company, Location, Workplace) and personal data scrubbed (emails, phone numbers).
- **Tags from titles:** function (engineering 23k, operations 9k, sales 8k, marketing 4k, data 3k, product 2k, design 1.6k, other 31k) and level. "Other" is mostly non-office work (nursing, trades, warehouse); kept as realistic mismatches, not used for careers.
- **Level rule fix:** "Manager" alone is not lead (Product, Account, Customer Success Manager are mid); Senior/Engineering Manager, Staff and Lead are. "Member of Technical Staff" is mid. The first version tagged 21k postings as lead.
- **Scale: ~2k CVs × 4 jobs = ~8k pairs**, down from 5k CVs, to stay inside $30 at the measured ~$0.0015 per pair plus ~$0.002 per CV.
- **Synthetic CV facts are ground truth:** Python samples the career (roles taken from real postings, dates, location, languages, education, style) and the model only writes the text, so years, functions and levels are known exactly; `cv_facts.md` extraction is only for real CVs.
- **Persona mix:** 7 career functions; years bands 0 (students: one internship) 8%, 1-3 25%, 3-6 30%, 6-10 25%, 10-20 12%; 27 locations, Europe-heavy, plus US, Canada, India, LatAm, Africa, APAC; 12% currently between jobs; ~10% career changers; 15% open to relocation; format, quality (strong / average / weak) and length varied.
- **Four jobs per CV (revised after the pilot):** near (TF-IDF-similar to the candidate's latest role, location plausible, within one level), seniority (similar, location plausible, 2+ levels away), location (similar but not open to the candidate's location), random (any posting).
  - First version picked any posting with the same function: in the 80-pair pilot 79 labels were `skip` and 73 had location `no` (a Bucharest designer paired with an on-site Auckland role). Labels were right; the sampling was not. A model trained on that learns "always skip".
  - Location plausibility is a cheap pre-filter on the posting's Location + Workplace fields (country, city, EU/EMEA/LATAM/APAC, worldwide, or remote with no place named); DeepSeek still labels the truth from the full posting.
- **Spend ledger:** `data/spend.json` sums every run; each stage stops at $30 total.

## 2026-10-22: relevant roles are judged by the labeller (replaces function tags)

- **Decision:** the pair prompt lists the candidate's roles with dates; DeepSeek returns `relevant_roles` (which roles count toward this job's experience requirement), and Python sums their dates and takes the highest title level among them.
- **Why:** the first pilot re-run showed relevant years of 0 for most pairs: function tags from title regexes ("data") did not match DeepSeek's `role_function` ("engineering" for an ML engineer job), so every such pair became `below`. Relevance is a judgment the model does well; date arithmetic stays in Python.
- **Near-match similarity** uses the title (weighted ×3) plus the first 1,500 characters of a posting; whole-posting TF-IDF matched on benefits and EEO boilerplate ("Patient Activation Specialist" as "near" for a software engineer).

## 2026-10-22: CVs are built around a target job

- **Decision:** each synthetic CV starts from a target posting. The persona gets a location the target accepts (85%), years matching the target's level (70%; otherwise any band), a career from TF-IDF-similar postings, and a coverage level the CV writer aims for: strong (~90% of the target's requirements, 40%), partial (~60%, 40%), weak (~30%, 20%). Pairs per CV: the target, a similar posting 2+ levels away, a similar posting in a location they cannot work in, a random posting.
- **Why:** random careers plus a search for matching jobs gave 78-79 `skip` out of 80 in three pilot runs: most postings are US-only or on-site, and a random career rarely meets a random posting's must-haves. Starting from the target guarantees real positives, and the coverage level spreads labels across apply / stretch / skip on purpose. DeepSeek still labels the truth; this only controls which pairs get asked about.

## 2026-10-22: pilot 4 tweaks

Pilot 4 (target-based CVs) moved the target pair to 7 apply / 3 stretch / 10 skip, with usable spread per head (location 50 no / 30 yes, seniority 47 below / 30 at / 3 above). Then:
- **Persona location from explicit matches:** the target's header must name the persona's country or city, one of its regions, or worldwide; a bare "Remote" is only a fallback (7 of 20 "Remote" targets were US-only in the body).
- **Over-qualified needs years too:** `above` requires 2+ levels and, when the posting states years, at least 5 more than asked (a 3.8-year candidate with "Senior" titles was flagged above a 3-year ask).
- **Random pair replaced by a second near pair** (similar, location plausible, within one level): random pairs were easy skips that teach little; near pairs add positives.

## 2026-10-22: pilot 5 fixes

Pilot 5: the target pair was still `skip` 16/20. Reasons, one by one:
- **Domestic relocation (rubric change):** on-site or hybrid in another city of the candidate's own country is `unclear`, not `no`; another country stays `no`. People relocate within their country and need no visa; treating New York → California on-site as ineligible made most US on-site jobs unreachable.
- **Years from the posting:** 70% of personas take their years from the target's stated "N+ years" (N to N+3) instead of the title's level band; "Firmware Engineer" (no level word) still asked for 5+ years.
- **CV writer invents jobs:** a sample CV added an employer and dates not in the sampled career, breaking the ground-truth facts. The prompt now says to list exactly the given roles and no other employers, with education ending before the first role.

## 2026-10-22: domestic override, full run

- **Domestic override (Python):** DeepSeek kept marking same-country on-site jobs `no` despite the rule (New York candidate, on-site Sunnyvale). If the posting's own Location field names the candidate's country, `no` becomes `unclear`. Synthetic personas live in their own country, so they are authorized there; for real CVs it applies only when the CV states the country. Pilot 6 with the override: 14 apply / 13 stretch / 53 skip (34% positive); location 32 yes / 22 unclear / 26 no; seniority 42 at / 36 below; must-haves 40 / 40.
- **Company names:** ATS slugs ("candidhealth") are turned into readable names before the CV writer sees them.
- **Full run started** (`data/full_run.sh`, nohup, resumable): 2,000 CVs in 4 batches of 500, 4 pairs each (~8,000 pairs), est. ~$13 (pilot: $0.0012 per CV, $0.00134 per pair). Spend before the run: $1.10.

## 2026-10-22: size sweep and distillation

- **Decision:** train four sizes on the same data and script: Ettin-32M, Ettin-68M, ModernBERT-base (149M), ModernBERT-large (395M). Ship two tiers: fast (best Ettin) and quality (base); large is the upper bound and the distillation teacher, not shipped (too slow in a browser at 4k tokens).
- **Why:** uses the free Kaggle quota fully (one sweep ≈ 7-9 GPU h of 30 h/week); gives a measured size / accuracy / speed curve for the model card; a fast tier covers laptops without a strong GPU.
- **Ettin:** Johns Hopkins encoder family (17M-1B) trained on the ModernBERT recipe with 8k context, so all sizes share one script and tokenizer setup; Ettin-32M already runs in browsers (Pawn, ONNX Runtime Web).
- **Distillation:** the trained large model labels extra CV × posting pairs at no API cost; small models train on DeepSeek labels plus these. Ceiling remains DeepSeek's label quality.
- **Rejected:** Ettin-1B and larger (too big and slow for a browser at 4k tokens).

## 2026-10-22: training setup (`train/train.py`, `scripts/build_dataset.py`, `scripts/kaggle_run.py`)

- **One script for every size:** Ettin 17M-400M and ModernBERT base/large are all the ModernBERT architecture (checked on Hugging Face), so one training script and one browser runtime cover the sweep.
- **Input:** "CV [SEP] job", up to 4,096 tokens, truncating the longer side first.
- **Heads on one pass:** verdict, location, seniority (3-way each), must-haves met (2-way) on mean-pooled tokens; per job line: met (none / yes / partial / no) and must flag, applied per token and averaged over each line's tokens. A linear head commutes with mean pooling, so the browser can average token logits per line: one pass gives "top gaps" (must-have lines labelled no).
- **Line labels:** each DeepSeek requirement is matched to the posting line it came from by word overlap (≥ 50% of its content words); unmatched lines are "none".
- **Split by CV** 80/10/10, so test CVs never appear in training.
- **Training:** fp16 autocast, gradient checkpointing, length-bucketed batches with a token budget, AdamW, linear warmup/decay, DDP over both T4s via torchrun. Best checkpoint by validation verdict macro-F1.
- **Calibration:** one temperature per head fitted on validation; test reports accuracy, macro-F1 and ECE per head.
- **Kaggle:** a private dataset (data + train.py) and private script kernels launched from the command line with `--accelerator NvidiaTeslaT4` (the 2×T4 shape). Weekly quota 30 GPU h, resets Saturdays 00:00 UTC.
- **Timing run first:** ~600 synthetic CVs × real postings with random labels, only to measure speed and memory at 4k tokens; that model is discarded.

## 2026-10-22: dataset v2, targeted pairs for the rare classes (`pipeline.py extra`, `data/v2_run.sh`)

- **Why:** on v1, Ettin-32M → 68M (2× params) gained +0.05 verdict macro-F1 and nothing on seniority. The weak classes are rare in v1: seniority `above` (over-qualified candidate) 325 of 7,891, verdict `stretch` 899. That points at data, not model size.
- **Targeted, not uniform:** uniform extra pairs would be mostly easy `skip`s. Two new pair kinds per CV: `over` (a similar job 2+ levels below the CV's level; senior+ CVs only) and `stretch` (a similar job exactly one level above). Same labeller and rules as v1; only the sampling changes.
- **More distinct CVs:** 250 new CVs with the standard 4 pairs, so the model sees more people, not only more pairs per person.
- **Frozen split:** v1 CVs (ids ≤ 1,973) keep their exact v1 split (verified byte-identical); new CVs split by id (`id % 10`: 0-7 train, 8 val, 9 test). New pairs follow their CV's split, so v1 and v2 models can be compared on the v1 test examples.
- **Budget:** capped at $24 of the $30 project cap, leaving a reserve for the human-evaluation set and fixes.

## 2026-10-22: 8-bit AdamW for ModernBERT-large (`train.py --optim adamw8bit`)

- ModernBERT-large (395M) runs out of memory on a 16 GB T4 at 4,096 tokens even with one sequence per step: fp32 weights + grads + AdamW state ≈ 6.3 GB per GPU before activations. 8-bit AdamW (bitsandbytes) cuts the optimizer state by ~2.4 GB. Only large uses it; smaller models keep fp32 AdamW, so their results are unaffected. Noted as a confound when comparing large with the others.

## 2026-10-23: 6 epochs as the default

- Ettin-68M for 8 epochs (on v1 data, see EXPERIMENTS.md `ettin68-v1e8`): validation macro-F1 rises through epoch 4-5 (verdict 0.44 → 0.66, seniority 0.38 → 0.61), then stays within ±0.02 (noise) for epochs 5-8, with no drop (no overfitting). 6 epochs keeps a margin over the observed plateau without paying for flat epochs. Caveat: the linear decay schedule spans the whole run, so a 6-epoch run is not identical to epoch 6 of an 8-epoch run; the best checkpoint by validation verdict macro-F1 is kept either way.

## 2026-10-23: rigour runs and one shared test set

- **Shared test set:** every run now also reports `test_v1`, the frozen v1 test examples (788 pairs: v1 CVs, v1 pair kinds), so models trained on any dataset version are compared on the same examples. Checkpoints trained before this are re-scored locally with `scripts/eval_v1.py` using their own validation-fitted temperatures; the script is checked by reproducing a v1 model's reported test numbers.
- **Run queue:** see PLAN.md, "Rigour gaps". Seeds and the LR check use Ettin-68M because it is cheap (~13 min/epoch); the 3-epoch LR runs double as the data-vs-epochs ablation (v2 at 3 epochs vs v1 at 3 epochs).
- **Selection rule from here:** choose by validation scores; read test once per final model.

## 2026-10-23: one canonical CV text (`scripts/cvtext.py`, dataset v3)

- **Why:** on a real CV (the author's, HTML) the models rated jobs the author applied to as apply/stretch far less often than the teacher (best: ModernBERT-large 63% vs DeepSeek 93%, 30 jobs). The HTML-to-text step broke a line at every tag (165 one-field lines), a layout no synthetic CV has; the synthetic "PDF-extracted" CVs came out clean, so the model never saw conversion clutter.
- **Choice:** normalise instead of augment. `html_to_text` is block-aware like a browser's `innerText`; `normalize_cv` maps any CV text to one layout (ALL-CAPS headings, "- " bullets, wrapped lines joined, page numbers and blank-line noise dropped, email/phone scrubbed). The same rules run on training CVs (dataset v3) and on the user's CV in the browser, so the model only sees one layout. Rule-based so the browser port is exact.
- **v3 = v2 with normalised CVs** (same pairs, labels and split). `build_dataset.py --raw-cv` still reproduces v2 byte-for-byte.
- Caveat: a quick re-flow of the author's CV moved real-CV scores by 10-13 points but did not close the gap, so layout is only part of it; the per-head diagnosis is in the private notes.

## 2026-10-26: distillation teacher = ModernBERT-large trained on v3

Two 6-epoch large runs (v2 and v3 CV text) disagree: v2 scores higher on the shared test (verdict 0.733 vs 0.644), v3 higher on validation (0.711 vs 0.696). Following the rule "select on validation only", and confirmed by the author's private real-CV check (where the v3 run agrees with the author's own past decisions far more often), the v3 run is the teacher. The test gap is read as single-seed noise on large's verdict head. The student keeps v3 text too, so teacher, student and browser all see `normalize_cv` output.

## 2026-10-28: shipped models and precision

- **Fast tier (default): distilled Ettin-68M (`d68-v3-s0`), q4f16, 76 MB.** Chosen on validation verdict among the three seeds. 4-bit weights cost nothing measurable: on 100 validation pairs the argmax matches PyTorch on 94-99% per head, F1 moves ±0.04 both ways (noise).
- **Quality tier (opt-in): ModernBERT-large (`mblarge-v3-e6`), fp16, 790 MB.** q4f16 (297 MB) was tested and rejected: verdict F1 0.720 → 0.669 and seniority 0.789 → 0.751 on the same 100 pairs, which is most of large's edge over the small model. A Quality tier that is barely better than Fast has no reason to exist, so it ships uncompressed; fp16 matches PyTorch on 100% of argmaxes.
- **Off-the-shelf similarity is not enough:** on the author's private real-CV check, TF-IDF and two general embedding models (bge-m3, gte-modernbert-base, untrained) rank the author's applied jobs well below both jobfit models, with the biggest gap on location, which similarity cannot read.
- **No paper:** the project ships the model and the Chrome extension.
