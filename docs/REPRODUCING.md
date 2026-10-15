# Reproducing the manuscript

Run all commands from the repository checkout with the environment activated. Linux/WSL is required for the original trainers' `fcntl`, resource recording and process locks. Saved-result verification needs CPU only; production graph fitting used one CUDA worker and four CPU threads.

## Saved-result verification

```bash
python -m scripts.verify_release --predictions --output reproduced/verification
python -m scripts.summarize_results --output reproduced/summaries
```

The first command checks release hashes, frozen data/splits, the seven pretrained-encoder hashes, every prediction identity/condition/target, and four archived metrics. The second rebuilds seed-level summaries and validation-candidate tables from saved metrics. It writes 11,106 study-membership/property/split records and 2,230 aggregate rows, including reused artifact memberships. `results/studies/` retains the original study-specific selection rules, adjusted tests and intervals; `results/figure_N/` contains the numerical figure source tables.

## Graph experiments

The portable adapter prints the command or weighted configuration unless `--run` is supplied. Use `--list` to discover exact job IDs. A job always writes a fresh directory under `reproduced/` and refuses to overwrite an existing result.

```bash
# Pretraining-scale transfer: one single-property fit with a supplied encoder
python -m scripts.reproduce --study pretraining_full_coverage_v1 --job melting_point/42/1m --run

# Single-property loss allocation: one eleven-head specialist
python -m scripts.reproduce --study property_specialists_1m_v1 --job melting_point/42/priority90 --run

# Application-aware allocation: one shared eleven-property model
python -m scripts.reproduce --study application_weighted_1m_v1 --job heat_transfer/42/weighted_plain --run

# List the other graph-study job IDs
python -m scripts.reproduce --study multitask_heads_30m_v1 --list
python -m scripts.reproduce --study property_specialists_30m_v1 --list
python -m scripts.reproduce --study heat_transfer_attention_30m_v1 --list
python -m scripts.reproduce --study co2_absorbent_weighted_30m_v1 --list
```

The public Python package is `il_contrastive/`, renamed from `il_mixer/` on 15 October 2026. The accompanying source edits update imports, module commands, package discovery and source-file paths only. Model equations, loss functions, optimization settings and checkpoint formats are unchanged. `provenance/source_files.json` retains the original research-source paths and hashes alongside the current public paths and hashes; `provenance/namespace_migration.json` records the namespace-only changes. Existing experiment configurations, fitted artifacts and numerical results retain their historical hashes. The old plans contain preparation-time dependency guards for the original queue and are retained as historical evidence in `configs/reference_plans/`.

`scripts.reproduce` replaces that historical queue guard with a public guard checking the frozen-data hash, pretrained-encoder hash, current published scientific-source hashes and released configuration. It retains the original loss, sampling, selection and optimization code. For weighted jobs at both scales it calls the unchanged 1M specialist/application trainer, which adds disk-checkpoint replay at the original 5e-6 training-SD tolerance. This is a portable numerical-finalization adaptation of the older 30.2M trainer. It does not change the 30.2M weights, priorities, initialization or epoch cap. The original older trainers are also included for inspection. New fits are expected to show backend/version numerical variation rather than bit-identical predictions.

The portable adapter does not resume the historical queues or regenerate their dependency snapshots. Each explicitly launched public job records its own configuration. Original reference-plan hashes refer to the pre-publication files; path-normalized copies have separate public hashes. Existing source-table provenance hashes retain their historical meaning.

## Fresh pretraining

Put the two separately supplied `.txt.gz` assets in the repository root, then restore the exact input filenames:

```bash
python -m scripts.restore_assets --pretraining-inputs .
python -m scripts.prepare_pretraining --output data/pretraining/corpus_v1.sqlite
python -m scripts.reproduce --study pretraining_full_coverage_v1 --ssl --job 1m
```

The first command checks compressed and decompressed hashes. Preparation streams the inputs, canonicalizes them, removes exact canonical duplicates, rejects unsupported structures, and excludes all complete benchmark structures. Use a fresh output path; it never overwrites an existing corpus. Preparing all structures requires substantial CPU time and approximately 12 GB for the SQLite index, plus input storage.

Add `--run` to the SSL command to train the selected scale. Seven scale labels are `100k`, `500k`, `1m`, `5m`, `10m`, `20m`, and `30.2m`. Each uses three complete passes, the same held-out contrastive sample, and a final third-pass checkpoint. A newly fitted encoder is saved below `reproduced/`; the supplied historical encoders remain the default downstream initialization. To investigate newly pretrained encoders, explicitly edit a copy of the execution configuration and record its new checksum in a separately versioned experiment.

## Tabular refitting

Install `python -m pip install -e '.[tabular]'`. Obtain third-party weights separately and verify them against `configs/external_dependencies.json`. Restore LimiX source at the pinned commit:

```bash
git clone https://github.com/limix-ldm-ai/LimiX.git external/LimiX
git -C external/LimiX checkout 61ede97e37d3eaae9e4500c0bba02965a2370eba
python -m pip install einops hyperopt tqdm pyyaml huggingface-hub
```

The pinned repository's environment file describes its optional platform-specific acceleration dependencies. Put `LimiX-16M.ckpt` in `models/tabular/limix/` and the exact V2 regressor in `models/tabular/tabpfn/tabpfn-v2-regressor.ckpt`. Run `python -m scripts.verify_external_assets` to check both checkpoint hashes and the LimiX source commit before refitting. The official [TabPFN model repository](https://huggingface.co/Prior-Labs/TabPFN-v2-reg) and pinned LimiX README provide acquisition instructions and provider terms. Do not substitute a newer default checkpoint when reproducing this benchmark.

Example raw descriptor refits:

```bash
python -m scripts.run_tabular_all --data data/benchmarks/il_v1/observations.csv.gz --output reproduced/tabular_extratrees --models extratrees --modes raw --seeds 42,43,44,45,46 --device cpu --run
python -m scripts.run_tabular_all --data data/benchmarks/il_v1/observations.csv.gz --output reproduced/tabular_limix --models limix --modes raw --seeds 42,43,44,45,46 --device cuda --run
python -m scripts.run_tabular_all --data data/benchmarks/il_v1/observations.csv.gz --output reproduced/tabular_tabpfn_small --models tabpfn --tasks melting_point,glass_transition,decomposition_temperature,toxicity,heat_capacity,refractive_index,surface_tension,thermal_conductivity,co2_solubility --modes raw --seeds 42,43,44,45,46 --device cuda --run
```

Both representations (`descriptors`, `descriptors_fp`) are defaults; all eleven tasks are default unless `--tasks` is provided. Extra Trees has 500 trees, minimum leaf size 2, and four CPU threads. TabPFN uses V2/four estimators. LimiX uses four quantile/power configurations with retrieval disabled and preprocessing fitted only on the training context.

Density and viscosity exceed TabPFN's normal training-row limit. Use the supplied full-context override, once per task/representation/seed:

```bash
python -m scripts.tabpfn_memory_job --task density --representation descriptors --seed 42 --output reproduced/tabpfn_density_descriptors_seed42 --run
```

This enables `ignore_pretraining_limits=True` and memory-saving mode, with four estimators and no subsampling. Repeat for the two tasks, two representations and seeds 42–46. `results/studies/tabular_final_v2/artifact_registry.json` records the exact configuration and original checkpoint hashes for all 330 final tabular records. Tabular learners did not save independently serialized fitted models; reproduce inference by refitting from frozen training contexts and the pinned provider checkpoint. Saved held-out predictions are included for result checking.

## Trained graph inference

Restore the optional checkpoint archive using `python -m scripts.restore_assets --checkpoints /path/to/graph_checkpoints_v1.zip`. Checkpoint paths and individual SHA-256 values are in `configs/optional_checkpoints.json`.

```bash
python -m scripts.predict --checkpoint runs/property_specialists_1m_v1/fits/melting_point/seed_42/priority90/best.pt --input examples/requests.csv --output reproduced/example_predictions.csv
```

The public wrapper calls the original predictor and selects the historical heat-model class when its checkpoint configuration requires it. Ordinary checkpoints also work with `python -m il_contrastive.predict`. The example request CSV uses the long-form `smiles,task,temperature_K,pressure_input` schema. Use checkpoints fitted for the requested tasks. Viscosity requires temperature; density and CO₂ use MPa pressure; heat capacity retains its unverified source pressure convention. Prediction does not combine different specialist checkpoints into a single shared model.
