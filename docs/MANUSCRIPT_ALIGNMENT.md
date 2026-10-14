# Alignment with the supplied manuscript

The repository was prepared against `IL_property_prediction_manuscript_revising.docx` supplied on 14 October 2026. Its SHA-256 is in `provenance/manuscript.json`. The document itself is omitted. The following draft wording differs from the saved scientific records; no manuscript text was changed during packaging.

| Draft location/wording | Executed setting or source record |
|---|---|
| Section 2.1: corpus obtained “by duplicating canonical structures” | Preparation **deduplicates** canonical structures and excludes complete benchmark structures. Accepted training corpus = 30,214,506 unique structures. |
| Section 2.3: uniform weighting “11/100 per property” | Exactly **1/11 per property**, totaling 1.0. It is approximately 9.09%, not 11%. |
| Section 2.3: six heat priorities and “six auxiliaries” | **30.2M:** six priorities (MP, Tg, Td, viscosity, Cp, conductivity), five auxiliaries; weights 15%/2%. **1M:** five priorities (Tg auxiliary), six auxiliaries; weights 18%/1.6667%. |
| Section 2.3: application settings not fully distinguished by scale | Application epoch caps are **200 at 1M** and **100 at 30.2M**. Specialist caps are 200 at both scales. Figure 5 records this distinction. |
| Table 1: Cp assigned a definite J mol⁻¹ K⁻¹ unit | Frozen HCAP and pressure units remain **unverified workbook conventions**. No guessed conversion is applied. |
| Table 1: surface tension labeled mN m⁻¹ | Frozen values retain the native convention **consistent with N/m**; values/MAEs were not multiplied by 1,000. Explicitly label native units or perform/document a display-only conversion. |

`data/benchmarks/il_v1/manifest.json` contains the original three historical fitting seeds 42–44. This metadata was frozen before the later five-seed studies. The experiments in `configs/experiments/` explicitly use seeds 42–46; they do not change the split. Do not read the historical manifest's seed list as the seed count of the final studies.

Figure/source mapping:

| Current figure | Numerical source directories |
|---|---|
| 1: molecular/experimental space, correlations, coverage | `results/figure_1/`; frozen data/schema |
| 2: pretraining scale and computation | `results/figure_2/`; `results/studies/pretraining_full_coverage_v1/` |
| 3: tabular/graph and chemical support | `results/figure_3/`; final tabular and head-family study tables |
| 4: uniform sharing, overlap and specialist allocations | `results/figure_4/`; both specialist studies |
| 5: application-aware optimization | `results/figure_5/`; both 30.2M applications and the joint 1M study |

Figures 3 and 4 use different reference normalizations. Figure 3 uses the historical smaller three-seed scratch GNN; Figure 4 uses the same property's five-seed 1M single-task mean. The scale-only architecture-matched comparison is Figure 2. Equal-IL versus measurement-weighted results and descriptive test minima versus validation choices remain separate in the source tables.
