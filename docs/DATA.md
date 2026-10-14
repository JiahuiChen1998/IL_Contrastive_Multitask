# Data and artifact dictionary

The authoritative labeled dataset is `data/benchmarks/il_v1/observations.csv.gz`. Its compressed-file SHA-256 is `c9c84284f2dfb8d90638fcae90fdf2e469e67b1e6f21105c253976f55763572e`.

| Field | Meaning |
|---|---|
| `smiles` | Canonical complete ionic-liquid structure, preserving fragment multiplicity and formal charge |
| `task` | Property identifier from `task_schema.json` |
| `value` | Original property response in the frozen source convention |
| `temperature_K` | Condition temperature in kelvin; missing where the schema has no temperature input |
| `pressure_input` | Condition pressure in the schema's input convention; CO₂ bar values have already been multiplied by 0.1 to give MPa |
| `source_file`, `source_sheet`, `source_row` | Identity of the original observation |
| `split` | Frozen `train`, `validation`, or `test` assignment |
| `prediction_transformed` | Model prediction after reversing training standardization, but before reversing the task log transform |
| `prediction` | Prediction in the native source target convention |

`row_splits.csv.gz` additionally supplies stable row IDs. `structure_splits.csv` assigns 4,376/546/536 complete ILs to train/validation/test. Every property and condition record for a complete IL stays in one partition; individual ions can occur in several partitions.

The curated dataset contains 74,887 observations for 5,458 ILs. The importer removed 514 invalid/incomplete/nonneutral records and 1,193 exact duplicates. Different measured values at identical conditions remain as replicates. The curation manifest and preserved `data/processed/rejected.csv` and `duplicates.csv` document those decisions.

Viscosity and CO₂ use natural-log training targets. Primary MAE uses log space for viscosity and native space otherwise. Standardization uses the population SD of training-task labels only. `task_schema.json` is authoritative for target names, input conditions, source units and transforms. In particular, heat-capacity target/pressure units remain unverified, and surface-tension values retain the native convention consistent with N/m.

Each released fit directory has `metrics.json`, optional task/execution/replay metadata, and gzip-compressed validation/test prediction CSVs. Compression preserves the exact original CSV bytes. The release audit interprets legacy float32/16 short-decimal serialization only when all four archived metrics independently agree. This interpretation does not alter saved files.

The corpus preparation manifest records 30,214,506 unique accepted training structures and 305,245 validation structures after canonical deduplication and exclusion of all complete benchmark structures. Of the training structures, 980,778 satisfy the neutral-ionic criterion. The rest comprise neutral molecules, isolated ions and other structures. The seven nested sizes use SHA-256 ordering and separate pretraining initialization; each receives three passes. This changes both unique structures and optimizer updates.

To reconstruct the corpus, restore the separately prepared exact input files and run `scripts.prepare_pretraining`. Their decompressed SHA-256 values are:

| Input | SHA-256 |
|---|---|
| `pretrain_train_set.txt` | `003745414f825a8aec939ea6d3be88f6b0cbb81b92de00d88306054e3211599a` |
| `pretrain_val_set.txt` | `af05fa7db62b26eb91ed9332b4dcf190962efd02b687bf4a656be2cc5c487670` |

The 11.54-GB SQLite file is a rebuildable local index and is omitted from Git. The unused `CID-SMILES.gz` input is also omitted. The authors confirmed on 14 October 2026 that the TXT files were provided by the ILBERT authors in the same research group. The associated paper is Qiu et al., Digital Discovery (2025), DOI [10.1039/D5DD00035A](https://doi.org/10.1039/D5DD00035A). The supporting release preserves those exact supplied files and does not assign them the code's MIT license.
