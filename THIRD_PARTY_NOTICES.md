# Third-party attribution

The root MIT license applies to original contributions. It does not replace the following notices or provider terms.

| Material | Provenance | Retained notice |
|---|---|---|
| Edge-augmented graph architecture and equations adapted locally | `ByteDance-Seed/bamboo_mixer`, copyright 2024 Bytedance Ltd. and/or affiliates | `licenses/LICENSE-BAMBOO`, Apache-2.0; root `NOTICE` |
| Property workbooks and the curated dataset derived from them | `GuzhongChen/ILTransR`, commit `ff5e55cfb8162b0706fc2d88ca5cd384705686b1` | `licenses/LICENSE-ILTransR`, upstream Apache-2.0 license; raw source manifest |
| LimiX source, obtained externally for refitting | `limix-ldm-ai/LimiX`, commit `61ede97e37d3eaae9e4500c0bba02965a2370eba` | `licenses/LICENSE-LimiX`, upstream Apache-2.0 license |
| TabPFN V2 regression weights | [official model repository](https://huggingface.co/Prior-Labs/TabPFN-v2-reg) | Obtain from provider; checkpoint terms remain those supplied by the provider |
| LimiX-16M weights | Official provider referenced by the pinned LimiX source | Obtain from provider; check recorded SHA-256 before use |
| Supplied pretraining TXT inputs | Provided by the ILBERT authors in the same research group; author-confirmed on 14 October 2026; exact files identified by SHA-256 | Original data provenance is preserved; these inputs are not relicensed by the code's MIT license |

The pretraining-input archives are prepared separately for the authors' data archive and attributed to the ILBERT authors. The associated work is Qiu et al., Digital Discovery (2025), DOI [10.1039/D5DD00035A](https://doi.org/10.1039/D5DD00035A). Exact historical files are supplied; no claim is made that they match a particular publicly hosted archive byte for byte. The experimental dataset retains source workbook, sheet, and row identifiers. Third-party library licenses are supplied by their installed distributions.

The supporting source implements an architecture adaptation; it does not claim to reproduce published Bamboo Mixer or ILTransR model results. The locally fitted graph checkpoints are separate models.
