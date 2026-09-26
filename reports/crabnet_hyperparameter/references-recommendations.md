# References update — recommendations for IMMI manuscript

Companion to `references.bib`. Generated from an Edison `LITERATURE_HIGH`
literature search on **2026-05-06** for the upcoming submission to
*Integrating Materials and Manufacturing Innovation* (IMMI, Springer Nature)
as a **Data Descriptor** article (Technical Article sub-class; ≤8000 words;
public dataset with persistent DOI required — Zenodo 7694268 satisfies this).

The recommendations below are **for the bibliography only**. The manuscript
prose itself is unchanged in this PR per the instruction *"Don't update the
manuscript yet, just update a .bib file and provide recommendations for
where to insert."*

---

## 1. Updates to existing references

Six existing entries cite preprint versions that have since been published in
peer-reviewed venues. The corresponding BibTeX keys in `references.bib`
already point to the journal versions. Update in-text citations as follows:

| Old ref # in DIB manuscript | Old citation                                          | New BibTeX key                       | Updated citation                                                                                                       |
| --------------------------- | ----------------------------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| (2)                         | Kandasamy et al., arXiv:1903.06694, 2020              | `Existing02_Kandasamy2020_Dragonfly` | JMLR **21**(81), 1–40 (2020)                                                                                           |
| (6)                         | Eriksson & Jankowiak, arXiv:2103.00349, 2021          | `Existing06_Eriksson2021_SAASBO`     | UAI 2021, PMLR **161**                                                                                                 |
| (7) and (9)                 | Two ChemRxiv versions of "Compactness Matters"        | `Existing07_09_Baird2023_Compactness`| Computational Materials Science **224**, 112134 (2023). **Consolidate refs (7) and (9) into a single citation.**       |
| (8)                         | Baird & Sparks, ChemRxiv-2023-fjjk7                   | `Existing08_Baird2023_HardSphere`    | Data in Brief **50**, 109487 (2023)                                                                                    |
| (12)                        | Wang et al., arXiv:2204.05838, 2022                   | `Existing12_Wang2022_ActiveLearning` | Oxford Open Materials Science **2**(1), itac006 (2022)                                                                 |

After applying these updates the reference list contracts from 15 entries
to 14 (refs 7 and 9 merge), then expands again with the additions in §2.

## 2. New references to add

Twelve new entries were identified, covering all eight topic areas requested
(A multi-fidelity BO • B multi-objective BO • C high-dimensional BO •
D constrained / mixed-variable BO • E benchmarking platforms • F CrabNet /
composition models • G heteroskedastic surrogates • H self-driving labs).

| BibTeX key                  | Topic | Complements / replaces existing ref(s) | Where to cite in the manuscript                                                                                                                         |
| --------------------------- | ----- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SabanzaGil2025_MFBO`       | A     | complements (1)                        | *Objective* / introduction, alongside (1) when motivating multi-fidelity BO for materials; also in any methods discussion of MF surrogate construction. |
| `Gantzler2023_MFBO_COF`     | A     | complements (1), (8)                   | *Objective* / related work, as a concrete materials-science MFBO case study (covalent organic frameworks for Xe/Kr separations).                        |
| `Ament2023_LogEI`           | B, G  | complements (3), (4)                   | *Methods* / related work on multi-objective acquisition functions (qNEHVI, qParEGO); cite LogEI / qLogNEHVI as the numerically-robust reformulation that addresses noisy observations — also relevant to the heteroskedastic-noise framing. |
| `Hvarfner2024_VanillaBO`    | C     | complements (5), (6)                   | *Discussion* alongside (5), (6); challenges the assumption that specialized HD methods are needed for the 23-dimensional CrabNet search space.          |
| `Papenmeier2023_Bounce`     | C, D  | complements (5), (6), (7)              | *Related work* on high-dimensional BO over mixed/combinatorial spaces — directly relevant given the mixed numerical/categorical CrabNet hyperparameters. |
| `Hickman2025_Anubis`        | D     | complements (7), (9)                   | *Related work* on constrained BO with **unknown** feasibility constraints — matches the failure-mode of CrabNet runs where some hyperparameter combinations fail at training time. |
| `Choudhary2024_JARVIS`      | E     | complements (10), (11)                 | *Introduction* / benchmarks paragraph alongside Matbench (10) and MODNet (11) when surveying community-driven materials benchmarking platforms.        |
| `Fitzner2025_BayBE`         | E     | new addition                           | *Value of the data* / methods section, when noting downstream use of the dataset in the Acceleration Consortium BayBE benchmarking notebooks.           |
| `Hickman2025_Atlas`         | E     | complements (15)                       | *Introduction* / benchmarks paragraph alongside Olympus (15) when surveying SDL optimization frameworks (Atlas covers mixed, multi-objective, constrained, and multi-fidelity). |
| `Madani2025_TransformerGraph` | F   | complements (5)                        | *Background* on CrabNet; cite as a recent CrabNet-inspired hybrid Transformer-Graph successor that extends attention-based composition models with structural information. |
| `Tom2024_SDL_Review`        | H     | new addition                           | *Introduction* when motivating the broader self-driving-laboratory context (definitive Chemical Reviews 2024 review; Baird and Sparks are co-authors). |
| `Abolhasani2023_SDL`        | H     | new addition                           | *Introduction*, alongside `Tom2024_SDL_Review`, as a concise complementary review on the rise of SDLs in chemical and materials sciences.              |

## 3. Items NOT to cite as formal references

Per @sgbaird's note, the following downstream uses of the surrogate may be
mentioned as URL footnotes / data-availability links but should **not** be
cited as bibliography entries (no peer-reviewed manuscript exists for them):

* Static surrogate deployment used in the AC HuggingFace Space:
  <https://huggingface.co/spaces/AccelerationConsortium/crabnet-hyperparameter>
  (training script: `train_surrogate.py` in the same Space; trained
  separately, not currently mirrored on Zenodo)
* Downstream BayBE & Ax benchmarking notebooks:
  <https://github.com/AccelerationConsortium/baybe-multi-task-bo>
* Acceleration Consortium Kaggle benchmarking competition dataset:
  <https://www.kaggle.com/datasets/acceleration-consortium/crabnet-optimization-challenge-dataset>

The peer-reviewed BayBE methods paper (`Fitzner2025_BayBE`) is the cite-able
proxy for the BayBE benchmarking work above.

## 4. Provenance

* Edison job: `LITERATURE_HIGH`, task id recorded in
  `agent_state` of the response payload, run on 2026-05-06.
* Query input: the 15 existing references plus the eight target topic areas
  (A–H) listed above; full prompt is preserved in the job submission.
* Reasoning, citation IDs and supporting page-level evidence for every
  recommendation are in the Edison response (not committed to the repo to
  keep the diff small — re-runnable from `EDISON_API_KEY`).
