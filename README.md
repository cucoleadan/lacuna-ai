<img alt="Lacuna: mind the gap, before you apply." src="assets/header-dark.png">

Lacuna tells a job seeker how a job posting fits their CV, and where it doesn't: a fit score with an apply / stretch / skip verdict, whether they can work where the job requires, the seniority it asks for, and the posting's must-have requirements their CV does not cover, line by line. It runs in the browser on WebGPU, so the CV never leaves the person's device. *Lacuna* is Latin for "gap".

- **Models** on Hugging Face: [Lacuna Celer](https://huggingface.co/lacuna-ai/lacuna-celer) (68M, 76 MB, the default) and [Lacuna Altus](https://huggingface.co/lacuna-ai/lacuna-altus) (395M, 790 MB)
- **Chrome extension:** [download the latest release](https://github.com/cucoleadan/lacuna-ai/releases/latest) (unzip, then *Load unpacked* in `chrome://extensions`); source in [`extension/`](extension/). A side panel that scores the job page you are on and keeps a ranked list of the jobs you've checked
- [DECISIONS.md](DECISIONS.md): every design choice, why, and what was rejected
- [EXPERIMENTS.md](EXPERIMENTS.md): every run and number
- [PLAN.md](PLAN.md): the original plan and measurements

## How it works

1. **Teacher labels.** DeepSeek V4.1 Flash (open weights, MIT) reads a CV and a job posting and judges facts: each requirement met / partial / not met, location eligibility, the posting's required years and level, which CV roles are relevant. Python applies the rubric thresholds, so every label follows the same published rules (`prompts/label_pair.md`).
2. **Synthetic CVs around a target job.** Each training CV is written for a fictional person whose career is sampled from real postings, anchored on a target job with a controlled level of fit (`prompts/synth_cv.md`, `scripts/pipeline.py`). 2,217 CVs, 12,699 labelled pairs.
3. **Models.** ModernBERT / Ettin encoders learn every head in one pass over "CV [SEP] job" at 4,096 tokens (`train/train.py`), on free Kaggle GPUs. ModernBERT-large (Altus) is the teacher; Ettin-68M (Celer) is distilled from it on 13,970 extra unlabelled pairs (`scripts/distil_pairs.py`, `train/teach.py`), which brings it level with the teacher on held-out data.
4. **Browser.** ONNX on WebGPU through ONNX Runtime Web (`scripts/export_onnx.py`, checked against PyTorch by `scripts/check_onnx.py`). The extension's JavaScript port of the CV normaliser and the encoder is tested against the Python pipeline (`extension/test/parity.mjs`).

## Results

Macro-F1 on 1,258 held-out pairs (test CVs never appear in training):

| Model | Verdict | Seniority | Location | Must-haves | Req. lines |
|---|---|---|---|---|---|
| Lacuna Celer (distilled Ettin-68M) | 0.686 | 0.718 | 0.903 | 0.810 | 0.709 |
| Lacuna Altus (ModernBERT-large) | 0.671 | 0.731 | 0.901 | 0.806 | 0.715 |
| Ettin-68M, same data, no distillation | 0.654 | 0.667 | 0.891 | 0.804 | 0.686 |

Details, seeds and every other run are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Running it

```
cp local.example.json local.json      # your OpenAI-compatible endpoint for DeepSeek V4.1 Flash, paths, Kaggle user
python scripts/pipeline.py postings   # needs a postings sqlite db (not distributed)
python scripts/pipeline.py cvs --n 20 && python scripts/pipeline.py pairs && python scripts/pipeline.py label
python scripts/build_dataset.py
python scripts/kaggle_run.py upload && python scripts/kaggle_run.py push --model jhu-clsp/ettin-encoder-68m --name e68
python scripts/export_onnx.py <run> --out export/<name>

cd extension && npm install && npm run build   # then load extension/dist in chrome://extensions
```

Scraped job postings are not redistributed; code, prompts, model weights and evaluation results are.

## Limits

Lacuna is a triage aid for job seekers, used on their own CV. It is not for screening or ranking candidates. Labels come from an LLM, not from recruiters; training CVs are synthetic; it works best on English postings; it treats US-only remote roles as closed to people outside the US.

MIT licensed.
