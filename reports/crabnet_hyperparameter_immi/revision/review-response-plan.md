# IMMI revision plan — CrabNet hyperparameter benchmark Data Descriptor

This document maps every Editor / Reviewer comment to a concrete planned
action, the manuscript location that will change, and (where a scientific or
strategic choice is required) a pointer to the corresponding question in
[`author-questions.md`](./author-questions.md). It is a **planning artefact**:
no manuscript prose is rewritten here. The authors drive the edits after
answering the open questions; the machinery in this folder then produces the
tracked-changes PDF and the point-by-point response.

> [!IMPORTANT]
> **Version provenance (read first).** The manuscript the reviewers saw is
> **not identical** to `../manuscript.tex` in this branch. Two reviewer quotes
> do not appear anywhere in the in-repo LaTeX:
>
> - *"applying a contrived constraint that the sum of all parameters must equal
>   one"* — this sentence is in the **original Data in Brief** source
>   (`../../crabnet_hyperparameter/Datainbrief.docx`), **not** in
>   `../manuscript.tex` (the Methods section here only says continuous
>   parameters "were given physically meaningful upper and lower bounds").
> - **Figure 6 (SHAP)** and the sentence *"SHAP analysis ... analyzes the
>   difference between the true experimental output and the expected output"* —
>   present in neither the DIB source nor `../manuscript.tex`; both stop at
>   Figures 1–5.
>
> Conclusion: the submitted PDF was a **hybrid** (DIB constraint prose + an
> added SHAP Figure 6) that lives outside this repo. See
> [`README.md`](./README.md) and **Q1** in `author-questions.md` — we need
> Xavier's exact submitted `.tex`/`.pdf` for a faithful tracked-changes diff.
> Until then the frozen baseline `manuscript_v0_submitted.tex` is the
> **pre-revision repo state**, not the literal submission.

---

## A. Cross-cutting themes

Most individual comments collapse into a smaller set of themes. Addressing a
theme once resolves several comments.

| ID | Theme | Raised by | One-line resolution |
|----|-------|-----------|---------------------|
| T1 | Sum-to-one constraint: ambiguous + "contrived" | Ed, R1#1, R1#3, R2#3 | Reframe honestly as a *deliberate synthetic linear-equality constraint* to exercise constrained optimizers; state precisely which variables it touches and how; **do not** claim it reproduces alloy thermodynamics. Show marginal distributions before/after (R1#3). |
| T2 | Scope: HPO benchmark vs materials-design benchmark | Ed, R1#4, R2#6 | State explicitly and early that this is a **hyperparameter-optimization benchmark** with materials-relevant *structure* (heteroskedastic noise, multi-fidelity, mixed variables, constraints), **not** a materials-design/inverse-design benchmark. |
| T3 | Tool primers + researcher workflow | Ed, R3, R2 (related work) | Add short paragraphs on **CrabNet**, **Matbench** (`matbench_expt_gap`), and **Ax**; add a "how a typical researcher uses these today, and what this benchmark lets them do better" paragraph. |
| T4 | "Multi-fidelity" not substantiated | R2#2 | Define the fidelity axes actually present — training-set fraction (`train_frac`), epochs, and repeat count — and their **cost** (runtime spans ~4 orders of magnitude); show cheap↔expensive correlation, or soften the claim to "supports multi-fidelity study via cost-controlled axes." |
| T5 | SHAP wording is technically wrong | R1#5 | Correct the caption/sentence: SHAP attributes a **model prediction** relative to a reference/expected prediction; it does **not** measure error against ground-truth experiment. (Only relevant if the SHAP figure is retained — see Q6.) |
| T6 | Figures: resolution, log ticks, consolidation | Ed | Consolidate Figs 1–5 into one multi-panel figure; supply Fig 6 (SHAP) as vector/print-res; fix Fig 4 log-y tick labels; **add the new Pareto figure(s)** already generated. |
| T7 | FAIR compliance + data dictionary | Ed | Add a column-by-column data dictionary (all 35 CSV columns + `.pkl`/`.json` contents) and an explicit FAIR statement (Findable=Zenodo DOI; Accessible=open CC-BY; Interoperable=CSV/JSON/pickle + documented schema; Reusable=license + provenance + surrogate). |
| T8 | Are there genuine objective trade-offs / Pareto fronts? | R1#2 | **Already computed** — see Section C. MAE↔RMSE are near-redundant (ρ≈0.97); accuracy trades off against runtime (ρ≈−0.6); model size is ~independent of accuracy. New Pareto panels visualise this. |
| T9 | Noise model (percentile ranks) unvalidated | R2#4 | Either (a) add a quantitative comparison of the rank-resampling noise model vs a Gaussian baseline (calibration / held-out log-likelihood), or (b) scope it explicitly as a *design choice* with qualitative justification + future-work. Decision → Q4. |
| T10 | No optimizer demonstration on the benchmark | R2#5 | Either (a) run ≥1 optimizer (Ax/BayBE) vs random baseline **on the surrogate** and show it discriminates, or (b) position as future work and cite the existing downstream uses (HF Space, BayBE/Ax notebooks, Kaggle competition). Decision → Q5. |
| T11 | Novelty / limitations / critical self-assessment | R2 (overall), R2#7 | Add a **Limitations** paragraph and sharpen "what optimization challenges this enables that existing resources (Olympus, particle-packing) do not." |
| T12 | Funding Information field blank in submission record | Ed | Submission-system action: enter **NSF DMR-1651668** in the Funding Information field. Manuscript text already acknowledges it. → Q11 (confirm grant list complete). |
| T13 | Internal data-accuracy discrepancies (found during this pass) | (authors) | Fix numbers that the released data does not support — see Section D. These are latent reviewer risks even though not explicitly raised. |

---

## B. Point-by-point matrix

Status legend: **[auto]** = can be drafted programmatically from data/plan;
**[author]** = needs an author decision (see linked Qn); **[system]** =
submission-system action outside the manuscript.

### Editor

| # | Comment (abridged) | Theme | Planned action | Location | Status |
|---|--------------------|-------|----------------|----------|--------|
| E1 | Data dictionary + FAIR compliance | T7 | Add data-dictionary table + FAIR paragraph; auto-generate the column list from `sobol_regression.csv`. | Data Records | [auto] |
| E2 | Resolve sum-to-one ambiguity | T1 | Reframe + define precisely + marginal-distribution figure. | Methods (search space) | [author] Q2, Q3 |
| E3 | Establish materials-community value; make scope explicit | T2, T11 | Scope sentence in Abstract + Background; value paragraph. | Abstract, Background & Summary | [author] Q2 |
| E4 | Add primers on CrabNet, Matbench, Ax + researcher workflow | T3 | Three short primer paragraphs + one workflow paragraph. | Background & Summary (new subsection) | [author] Q7 (depth) |
| E5 | Fig 6 resolution; Fig 4 log ticks; consolidate Figs 1–5 | T6 | Multi-panel consolidation; vector Fig 6; fix Fig 4 ticks; add Pareto fig. | Technical Validation | [author] Q8 |
| E6 | Complete Funding Information field | T12 | Enter NSF DMR-1651668 in Editorial Manager. | (submission record) | [system] Q11 |

### Reviewer 1

| # | Comment (abridged) | Theme | Planned action | Location | Status |
|---|--------------------|-------|----------------|----------|--------|
| R1.1 | Sum-to-one constraint artificial / physical motivation overstated | T1 | Remove any alloy-physics claim; describe as synthetic equality constraint whose *purpose* is to make the benchmark exercise constrained optimizers. | Methods | [author] Q2 |
| R1.2 | Are there real Pareto fronts / correlated objectives? | T8 | Add Pareto figure + Spearman table (Section C); text interpreting redundancy of MAE/RMSE and the accuracy–runtime trade-off. | Technical Validation | [auto] |
| R1.3 | Is Sobol ≈ uniform? Show marginals + constraint effect | T1 | Add per-parameter marginal histograms (raw Sobol vs post-constraint); state Sobol gives low-discrepancy ≈uniform marginals and quantify how the constraint deforms them. | Methods / Tech Validation | [auto] + Q3 |
| R1.4 | This is an ML-HPO benchmark, not materials-design | T2 | Explicit scope statement. | Abstract, Background | [author] Q2 |
| R1.5 | SHAP description technically inaccurate | T5 | Correct wording (attribution vs error). | (SHAP fig caption) | [author] Q6 |

### Reviewer 2 (recommend reject — needs the most work)

| # | Comment (abridged) | Theme | Planned action | Location | Status |
|---|--------------------|-------|----------------|----------|--------|
| R2.a | Limited novelty; reads as technical report | T11 | Sharpen contribution framing; Limitations + "what this enables" paragraphs; lean on Data Descriptor article type (a dataset paper is legitimately descriptive). | Background, new Limitations | [author] Q2, Q10 |
| R2.b | SHAP superficial / intuitive | T5, T11 | Either deepen SHAP interpretation or de-emphasise it; tie to actionable guidance. | Tech Validation | [author] Q6 |
| R2.c | "Multi-fidelity" not substantiated | T4 | Define + demonstrate cost-controlled fidelity axes; or soften. | Methods, Tech Validation | [author] Q9 |
| R2.d | Sum-to-one "contrived"; doesn't reproduce physics | T1 | Same as T1 reframe; concede it is synthetic and justify by optimizer-stressing purpose. | Methods | [author] Q2 |
| R2.e | Noise model not validated vs simpler models | T9 | Ablation (rank-resampling vs Gaussian) or scoped design-choice justification. | Methods / new Tech Validation subsec | [author] Q4 |
| R2.f | No optimizer demo vs Olympus / particle-packing | T10 | Optimizer-vs-random demo on surrogate, or future-work + downstream-use citations. | new Usage/Validation subsec | [author] Q5 |
| R2.g | Narrow scope (one architecture, one task) | T2, T11 | Acknowledge in Limitations; frame as first entry in an extensible benchmark family. | Limitations | [author] Q10 |
| R2.h | Weak related-work / self-criticism | T11 | Expand related work (already have 26 refs incl. 12 new); add critical Limitations. | Background, Limitations | [auto] + Q7 |

### Reviewer 3 (positive — light touch)

| # | Comment (abridged) | Theme | Planned action | Location | Status |
|---|--------------------|-------|----------------|----------|--------|
| R3.1 | What is CrabNet and why care | T3 | CrabNet primer paragraph. | Background & Summary | [author] Q7 |
| R3.2 | What is Matbench | T3 | Matbench + `matbench_expt_gap` primer. | Background & Summary | [author] Q7 |
| R3.3 | The Ax platform | T3 | Ax primer (role: Sobol generation + intended BO consumer). | Background / Methods | [author] Q7 |
| R3.4 | How researchers use these + how the benchmark improves their work | T3, T2 | Workflow paragraph (status quo → with this benchmark). | Background & Summary | [author] Q7 |

---

## C. New analysis already produced (answers R1.2 directly)

`scripts/crabnet_hyperparameter/plot_pareto_fronts.py` regenerates two
2×3-panel figures (one per objective pair) with Pareto fronts and Spearman ρ:

- `../figures/pareto_rawdata.{png,pdf}` — repeat-averaged raw objectives
  (41,550 unique sets from 173,219 runs).
- `../figures/pareto_surrogate.{png,pdf}` — released RandomForest surrogate
  evaluated at the median noise percentile (rank = 0.5).

**Spearman rank correlations (repeat-averaged raw data; surrogate in
parentheses):**

| Objective pair | Spearman ρ | Pareto-front size | Interpretation |
|----------------|-----------|-------------------|----------------|
| MAE vs RMSE | +0.97 (+0.97) | 1 (2) | Near-redundant — effectively one accuracy objective. |
| MAE vs runtime | −0.57 (−0.58) | 65 (76) | Genuine accuracy↔cost trade-off. |
| MAE vs model size | −0.05 (−0.05) | 22 (22) | Essentially independent. |
| RMSE vs runtime | −0.61 (−0.62) | 52 (53) | Genuine accuracy↔cost trade-off. |
| RMSE vs model size | −0.07 (−0.08) | 21 (21) | Essentially independent. |
| runtime vs model size | +0.37 (+0.38) | 15 (16) | Weakly coupled cost objectives. |

(Both objectives in every pair are minimized, so a **negative** ρ implies a
real trade-off / non-trivial Pareto front; the surrogate reproduces the raw
structure to within ±0.02, which is itself a technical-validation result.)

**Takeaways for the response:**
- MAE and RMSE should be presented as a single accuracy axis (or one used as a
  redundant check); this answers R1.2's "are some objectives strongly
  correlated?".
- Accuracy vs runtime is the substantive multi-objective trade-off and the
  natural target for multi-objective + multi-fidelity demonstrations (T4, T10).

---

## D. Data-accuracy discrepancies found during this pass (Theme T13)

These are **not** yet reflected in `../manuscript.tex`; flag to authors before
resubmission because reviewers can trivially check them against the public CSV.

1. **Mean repeats "≈2.6" is wrong.** `manuscript.tex` (Background and the
   Figure 1 caption) states ≈2.6 successful repeats per set. The released
   `sobol_regression.csv` has 173,219 rows over 41,550 unique sets →
   **mean 4.17**. Repeat-count histogram: {1:130, 2:1,294, 3:6,647, 4:16,835,
   5:16,644}. → **Q12**.
2. **"MAE and RMSE span more than two orders of magnitude" is unsupported by
   the released data.** In `sobol_regression.csv`, MAE ∈ [0.169, 1.214] eV
   (7.2×, 0.86 decades) and RMSE ∈ [0.440, 1.703] eV (3.9×, 0.59 decades) —
   **less than one order of magnitude**. It is **runtime** that spans ~4
   orders (2.84 s → 3.39×10⁴ s) and model size ~1.8 orders. Either the
   histograms were made from *unfiltered* data (incl. divergent runs) that is
   not what was deposited, or the claim is a leftover. → **Q12**, and relates
   to E1/T7 (figures must match the deposited records).
3. **Constraint language divergence** (see provenance box): the reviewer-quoted
   "contrived sum-to-one constraint" is not in `manuscript.tex`. The reframe
   (T1) must be written against whatever text was actually submitted. → **Q1**.
4. **SHAP Figure 6 provenance**: not in repo. Its resolution fix (E5) and
   wording fix (R1.5/T5) can only be done once the source figure/caption is
   located. → **Q1**, **Q6**.

---

## F. Edison rubber-duck refinements (independent critique)

An independent Edison Scientific literature query stress-tested the responses
above; the full verdict table, ready-to-adapt wording, and a referenced reading
list are in [`edison-rubber-duck.md`](./edison-rubber-duck.md). Net changes to
the themes above:

- **T1 (constraint):** frame as a *synthetic linear equality (simplex)*
  constraint reproducing composition **geometry only** — explicitly not
  thermodynamics; show empirical marginals before/after the constraint and say
  Sobol *targets* low-discrepancy uniform coverage (not i.i.d. uniform).
- **T2 (scope):** standardize on "materials-relevant HPO / black-box
  optimization benchmark"; name **YAHPO Gym** (Pfisterer 2022) as the precedent.
- **T4 (multi-fidelity) — FIX:** keep `train_frac` + epochs as fidelity axes but
  **reclassify "repeats" as noise characterization, not fidelity**; runtime is a
  *cost correlate*. Substantiate with ≥1 cost-aware MF run or soften the claim.
- **T5 (SHAP):** adopt the corrected decomposition wording (prediction =
  baseline + attributions; not error-vs-experiment).
- **T7 (FAIR):** say "**designed to support** FAIR", map to sub-principles
  F1–F4/A1–A2/I1–I3/R1, add checksums + versioned DOI + machine-readable schema.
- **T8 (Pareto):** add Pearson **and** Spearman matrices, the **four-objective**
  non-dominated set, and report the raw-vs-surrogate front agreement
  (ρ matches within ±0.02) as surrogate validation.
- **T9 (noise) — WEAK:** relabel percentile ranks a *nonparametric descriptive
  encoding*; add a minimal ablation vs a Gaussian(μ,σ) baseline.
- **T10 (utility) — WEAK:** run a minimal repeated random-vs-BO comparison
  (fixed budgets/seeds; regret/hypervolume vs cost) and validate the surrogate
  by optimizer-ranking agreement, not adoption counts.

The three items most likely to draw further objection are **T4, T9, T10** — do
at least the *minimum-viable* versions rather than deferring them.

## E. Deliverables in this folder

| File | Purpose |
|------|---------|
| `review-response-plan.md` | This document. |
| `author-questions.md` | Multiple-choice / multi-select questions to unblock the edits. **Answer these first.** |
| `response-to-reviewers.tex` | Placeholder (unpopulated) LaTeX point-by-point response; reviewer text quoted, author responses stubbed. |
| `manuscript_v0_submitted.tex` | Frozen pre-revision snapshot of `../manuscript.tex` (latexdiff baseline). |
| `manuscript_original_dib.tex` | Pandoc conversion of the original Data in Brief `.docx` (contains the reviewer-quoted "contrived constraint" prose) — worst-case reference baseline. |
| `make-tracked-changes.sh` | Runs `latexdiff(baseline, ../manuscript.tex)` → `manuscript_diff.pdf`. |
| `edison-rubber-duck.md` | Independent critique of this plan from an Edison Scientific literature query. |
| `README.md` | How the revision workflow fits together + the provenance caveat. |

**Suggested order of operations**

1. Answer `author-questions.md` (esp. Q1 — supply the real submitted files).
2. Drop the true submitted `.tex` in as `manuscript_v0_submitted.tex` (replacing
   the placeholder baseline) if available.
3. Make the edits to `../manuscript.tex` driven by the answers.
4. Run `bash make-tracked-changes.sh` to produce the markup PDF.
5. Populate `response-to-reviewers.tex` from this plan + the tracked-changes PDF.
