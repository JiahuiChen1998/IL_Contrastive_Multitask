# Ionic-liquid property prediction

Supporting data and code for **Efficient ionic-liquid property prediction through contrastive learning and task-specific loss allocation**, by Jiahui Chen, Jiawen Li, Yuxin Qiu, and Zhen Song.

Repository: [JiahuiChen1998/IL_Contrastive_Multitask](https://github.com/JiahuiChen1998/IL_Contrastive_Multitask).

This release follows the manuscript snapshot dated 14 October 2026. It contains the frozen experimental dataset, original model code, portable experiment configurations, saved validation/test predictions, and numerical source tables for Figures 1–5. It excludes manuscript drafts, rendered artwork, figure-layout utilities, obsolete pilots, local environments, logs, and queue state.

## Start here

Python 3.12 on Linux or WSL is the reference environment. CPU is sufficient for checking the saved results. Install an appropriate PyTorch build separately; the original fitting environment used PyTorch 2.11.0 with CUDA 12.8.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch
python -m pip install -e '.[test]'
python -m scripts.verify_release
python -m scripts.verify_release --predictions --output reproduced/verification
python -m scripts.summarize_results
python -m pytest -q
```

The prediction audit checks row identities, target values, recorded conditions, and four independently recomputed metrics for every released property/split prediction set. It performs no fitting or model inference. The first verification command also checks release checksums and the frozen complete-IL split.

## Contents

| Path | Contents |
|---|---|
| `data/benchmarks/il_v1/` | 74,887 observations, 5,458 complete IL identities, row/structure split manifests, task schema and checksums |
| `data/processed/` | Preserved importer outputs, rejected records and duplicate records |
| `il_contrastive/` | Graph features, encoder, condition heads, training, pretraining and prediction code |
| `scripts/` | Scientific trainers and their dependencies; portable reproduction and verification commands |
| `configs/experiments/` | Settings and jobs for the eight graph studies |
| `configs/reference_plans/` | Historical plans, with local paths normalized for publication |
| `configs/fit_registry.json` | 1,108 distinct fitted-artifact directories and their study memberships |
| `runs/` | Metrics, model configurations and losslessly compressed saved predictions; seven final pretrained encoders |
| `results/figure_1/` … `results/figure_5/` | Numerical source tables mapped to the current manuscript figures |
| `results/studies/` | Completed study summaries, selection records, intervals and audit records |
| `provenance/` | Original-source hashes, public-release hashes, manuscript snapshot hash and large-asset inventory |
| `verification/` | Verification results produced when this release was assembled |

The larger head-family studies provide contextual graph candidates for the chemical-support comparison. They reuse fitted artifacts across studies; membership counts must not be added to obtain a number of independent fits. Structure-only equation/MLP labels sometimes alias the same fitted model. The registry consolidates those paths.

## Reproduce an experiment

List job identifiers and inspect one job before execution:

```bash
python -m scripts.reproduce --study pretraining_full_coverage_v1 --list
python -m scripts.reproduce --study pretraining_full_coverage_v1 --job melting_point/42/1m
python -m scripts.reproduce --study property_specialists_1m_v1 --job melting_point/42/priority90
```

Add `--run` to execute one selected job. New results go to `reproduced/`; released results are preserved. The seven included final encoders support downstream refitting without repeating pretraining. See [reproduction instructions](docs/REPRODUCING.md) for weighted training, tabular algorithms, complete pretraining, and checkpoint inference.

## Large assets

The following large files are distributed separately through the [v1.0.0 release](https://github.com/JiahuiChen1998/IL_Contrastive_Multitask/releases/tag/v1.0.0):

- `graph_checkpoints_v1.zip`: 778 trained graph checkpoints, matching `configs/optional_checkpoints.json`.
- `pretrain_train_set.txt.gz` and `pretrain_val_set.txt.gz`: exact, compressed copies of the supplied pretraining input files.

They are unnecessary for checking saved predictions. Full graph inference needs the checkpoint archive; fresh pretraining needs the input files and a rebuilt SQLite corpus. Third-party TabPFN and LimiX weights are obtained separately from their providers, using the recorded checksums. See [data documentation](docs/DATA.md) and [upload instructions](docs/UPLOAD.md).

## Interpretation and manuscript alignment

Primary MAE uses **ln-viscosity** and native targets for the other properties, including CO₂ mole fraction. Standardized MSE uses training-task variance. Seed SD, paired-seed SD, and whole-IL bootstrap intervals represent different uncertainties. Validation choices and descriptive test minima are recorded separately.

[Manuscript alignment notes](docs/MANUSCRIPT_ALIGNMENT.md) identify differences between the supplied draft and the saved settings, including uniform allocation, application priorities, corpus deduplication wording, and source units. The published data and configurations preserve the executed experiments.

## Attribution and license

Original project code is released under [MIT](LICENSE), as requested by the authors. External model weights and supplied pretraining inputs retain their own provenance and terms; MIT does not relicense them.

The original property workbooks are from [ILTransR](https://github.com/GuzhongChen/ILTransR), pinned to commit `ff5e55cfb8162b0706fc2d88ca5cd384705686b1`. Use `python -m scripts.fetch_data` if you need those workbooks; hashes are verified before use. Citation metadata is in [CITATION.cff](CITATION.cff).
