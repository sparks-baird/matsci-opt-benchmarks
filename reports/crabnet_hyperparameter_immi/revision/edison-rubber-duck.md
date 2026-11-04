<!-- markdownlint-disable -->
# Edison Scientific rubber-duck: critique of the revision plan

An independent `LITERATURE_HIGH` query (Edison Scientific, task
`966ce63d-ba0d-4ed5-9f92-09f5de658596`) was used to stress-test the planned
responses (themes T1–T13 / responses A–J) and to surface supporting or
contradicting peer-reviewed literature. The full transcript (question + answer +
reference list with DOIs) is in
[`edison-rubber-duck-transcript.md`](./edison-rubber-duck-transcript.md).

> [!NOTE]
> Edison is an AI literature agent. Treat verdicts as a well-sourced second
> opinion and **verify each DOI/claim** before citing. Page numbers in the
> transcript (e.g. "pages 5-7") are Edison's internal source locators, not
> manuscript pages.

## 1. Per-response verdicts

| Resp. | Theme | Verdict | Headline fix |
|-------|-------|---------|--------------|
| A — synthetic sum-to-one constraint | T1 | **Defensible** | Call it a *synthetic linear equality (simplex) constraint*; reproduce only the geometry; explicitly **not** physics/thermodynamics; list affected params + transform. |
| B — correlations + pairwise Pareto | T8 | **Defensible** | Add Pearson **and** Spearman matrices, all six pairwise fronts, **plus the four-objective non-dominated set**; report non-dominated counts, repeat-uncertainty, raw-vs-surrogate agreement. |
| C — Sobol marginals | T1/R1.3 | **Defensible (caveats)** | Say Sobol *targets* uniform coverage with lower discrepancy, **not** that it equals i.i.d. uniform; show empirical marginals/CDFs before & after the constraint; add a 23-D finite-sample caveat. |
| D — benchmark scope | T2 | **Defensible** | Use "**materials-relevant HPO / black-box optimization benchmark**" throughout; compare explicitly to YAHPO Gym. |
| E — SHAP wording | T5 | **Very strong** | Adopt the exact corrected wording (Section 3); name model/output/explainer/background. |
| F — multi-fidelity | T4 | **FIX** | `train_frac` + epochs are legitimate fidelity axes; **reclassify "repeats" as noise characterization**; runtime is a *cost correlate*, not a fidelity coordinate; demonstrate ≥1 cost-aware MF run. |
| G — percentile-rank noise | T9 | **Weak** | Do not call ranks a "distribution-free noise model"; call it a *nonparametric descriptive encoding*; add an ablation vs a Gaussian(μ,σ) baseline (calibration / held-out likelihood). |
| H — practical utility | T10 | **Weak** | Adoption ≠ validity. Run a minimal *repeated* comparison (random/Sobol vs ≥1 BO method, fixed budgets/seeds, regret/hypervolume vs cost) and validate surrogate fidelity via optimizer-ranking agreement. |
| I — FAIR + dictionary | T7 | **Defensible** | "Compliance is not binary" — map evidence to sub-principles F1–F4/A1–A2/I1–I3/R1–R1.3; add checksums, versioned DOI, machine-readable schema; say "**designed to support FAIR**" unless formally assessed. |
| J — CrabNet/Matbench/Ax primers | T3 | **Defensible** | Cite original CrabNet (Wang 2021) + Baird 2022 (CrabNet–Ax–SAASBO) + Matbench; state the exact Matbench version/fold/count; clarify Ax is an optimizer, not a materials model. |

## 2. The three critical flags (most likely to sink the paper)

1. **Multi-fidelity (F).** Naming *repeats* as a fidelity axis is not standard —
   repeats improve *precision*, they do not change approximation fidelity.
   Defensible version: name `train_frac` and epochs as fidelity axes,
   demonstrate their cost–accuracy trade-off, and run at least one cost-aware
   multi-fidelity optimization on the surrogate. Otherwise **soften** the claim.
2. **Percentile-rank noise (G).** No published precedent validates within-group
   percentile ranks as a heteroskedastic-noise *model*. With only 1–5 repeats
   percentiles are coarse and singletons carry no variance. Reframe as a
   descriptive encoding **and** add a quantitative ablation vs a Gaussian
   baseline (this is the least defensible claim as written).
3. **Practical utility (H).** A single illustrative BO trajectory is anecdotal;
   established benchmarks (YAHPO Gym, Olympus) validate a surrogate by showing
   optimizer *rankings* on the surrogate match those on real evaluations. Run a
   small repeated random-vs-BO comparison with predefined budgets/seeds.

These map to author-questions **Q9** (multi-fidelity/repeats), **Q4** (noise
ablation), and **Q5** (optimizer demo) respectively — Edison's guidance is that
the *minimum viable* versions (small ablation, small BO demo) are worth doing
rather than deferring entirely, because G and H are otherwise the weakest points.

## 3. Ready-to-adapt wording

- **SHAP (E).** "For the specified background distribution and SHAP variant,
  SHAP decomposes the surrogate model's prediction for a given input into a
  baseline value (the expected prediction over the background dataset) plus
  individual feature attributions; each attribution quantifies how much that
  feature shifts the prediction away from the baseline. It does **not** measure
  the gap between true experimental values and the model's expected output."
- **Scope (D).** "a high-dimensional, multi-objective **surrogate benchmark for
  hyperparameter optimization**, grounded in a real materials-science ML-training
  task (Matbench `matbench_expt_gap`) rather than a synthetic function — a
  complementary testbed that is higher-dimensional and noisier than standard
  synthetic functions, and cheaper/more reproducible than real ML-training runs.
  It is **not** a materials-property-design benchmark."
- **Constraint (A).** "a deliberately synthetic linear equality (sum-to-one)
  constraint that reproduces the *simplex geometry* of composition/mixture
  constraints, included to exercise constrained optimizers; it is not a physics,
  thermodynamic, phase-stability, or processing model."
- **FAIR (I).** "designed to support the FAIR principles" (not "FAIR-compliant")
  unless a formal assessment against F1–F4/A1–A2/I1–I3/R1–R1.3 is included.

## 4. Recommended references (verify before citing)

Peer-reviewed unless marked *(preprint)*. "In bib?" = already present in
`../references.bib`.

| Topic (theme) | Reference | Year | Venue | In bib? |
|---------------|-----------|------|-------|---------|
| MF-BO materials (T4) | Sabanza-Gil et al., 10.1038/... / arXiv:2410.00544 | 2025 | Nat. Comput. Sci. | yes (`SabanzaGil2025_MFBO`) |
| MF-BO materials (T4) | Fare et al., 10.1038/s41524-022-00947-9 | 2022 | npj Comput. Mater. | no |
| MF-BO materials (T4) | Tran et al., 10.1063/5.0015672 | 2020 | J. Chem. Phys. | no |
| MF-BO taxonomy (T4) | Do & Zhang, arXiv:2311.13050 *(preprint)* | 2023 | arXiv | no |
| MF/MO HPO benchmark (T2/T4/T10) | **Pfisterer et al., YAHPO Gym** | 2022 | AutoML/PMLR | no ⭐ closest precedent |
| MF-HPO (T4) | Moosbauer et al., 10.1109/TEVC.2022.3211336 | 2022 | IEEE TEVC | no |
| Tabular HPO benchmark (T2) | Klein & Hutter, arXiv:1905.04970 *(preprint)* | 2019 | arXiv | no |
| High-dim MOBO (T8) | Daulton et al. (MORBO), arXiv:2109.10964 | 2022 | NeurIPS | no |
| High-dim BO comparison (T10) | Santoni et al., 10.1145/3670683 | 2024 | ACM TELO | no |
| Heteroskedastic BO (T9) | **Griffiths et al., 10.1088/2632-2153/ac298c** | 2022 | MLST | no ⭐ noise ablation |
| Heterogeneous MF errors (T9) | Foumani et al., arXiv:2309.02771 *(preprint)* | 2023 | arXiv | no |
| Constrained MOBO SDL (T8/T10) | Low et al., 10.1038/s41524-024-01274-x | 2024 | npj Comput. Mater. | no |
| Simplex constraints (T1) | Hickman et al., arXiv:2203.17241 *(preprint)* | 2022 | arXiv | no |
| Simplex/formulation (T1) | Verret et al., chemrxiv-2025-hx7pz *(preprint)* | 2025 | ChemRxiv | no |
| SHAP (T5) | **Chen et al., 10.1038/s42256-023-00657-x** | 2023 | Nat. Mach. Intell. | no ⭐ |
| SHAP (T5) | Lundberg & Lee | 2017 | NeurIPS | no |
| Sobol discrepancy (T1/R1.3) | Morokoff & Caflisch, 10.1137/0915077 | 1994 | SIAM J. Sci. Comput. | no |
| Sobol high-dim (R1.3) | Kucherenko, High-Dim Sobol' | 2008 | proc. | no |
| FAIR (T7) | **Wilkinson et al., 10.1038/sdata.2016.18** | 2016 | Sci. Data | no ⭐ |
| Olympus benchmark (T10) | Häse et al., 10.1088/2632-2153/abedc8 | 2021 | MLST | no |
| Olympus enhanced (T10) | Hickman et al. *(preprint)* | 2023 | ChemRxiv | no |

⭐ = highest-value additions for the flagged weak points (YAHPO Gym for
scope/benchmarking, Griffiths for the noise ablation, Chen for SHAP, Wilkinson
for FAIR). These are **not** auto-added to `references.bib`; import the subset
matching the responses the authors choose (Q4/Q5/Q7/Q9).

## 5. Net effect on the plan and questions

- Plan **T4** updated: repeats → noise characterization (not fidelity); keep
  `train_frac`/epochs as the fidelity axes; runtime = cost correlate.
- Plan **T2** wording → "materials-relevant HPO / black-box optimization
  benchmark"; add YAHPO Gym as the named precedent.
- Plan **T7** → "designed to support FAIR" + sub-principle mapping + checksums.
- Plan **T8** → add the four-objective non-dominated set + Pearson & Spearman
  matrices + explicit raw-vs-surrogate front agreement (the surrogate front
  currently matches raw ρ to within ±0.02 — report this as validation).
- Plan **T9/T10** → do the *minimum viable* ablation and BO demo rather than
  deferring both; these are the two weakest points.
- Author-questions **Q9** annotated to reflect the repeats reclassification.
