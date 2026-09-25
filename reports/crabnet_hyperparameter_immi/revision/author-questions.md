<!-- markdownlint-disable -->
# Author decisions needed to action the IMMI revision

**Edit this file in place** (check the boxes, add notes) and commit — the
answers drive the manuscript edits and the point-by-point response.

> Direct edit link:
> <https://github.com/sparks-baird/matsci-opt-benchmarks/edit/copilot/rework-crabnet-benchmark-paper/reports/crabnet_hyperparameter_immi/revision/author-questions.md>

Conventions: `[ ]` = unchecked, `[x]` = checked. Questions tagged
**(single)** expect one box; **(multi)** expect one or more. Each question ends
with a free-text **Notes** line. Question IDs match `review-response-plan.md`.
A ⭐ marks the option the plan currently recommends.

---

## Q1 — Provide the *actual* submitted manuscript (single) — **BLOCKER**

The reviewers quote text (a "contrived sum-to-one constraint") and a **SHAP
Figure 6** that exist in **neither** `../manuscript.tex` **nor** the original
Data in Brief source. We cannot produce a faithful tracked-changes diff without
the file that was actually uploaded to Editorial Manager.

- [ ] I will drop the exact submitted `.tex` into this folder (replacing the placeholder `manuscript_v0_submitted.tex`). ⭐
- [ ] I can only provide the submitted **PDF** (we will convert/reconstruct a `.tex` from it for diffing).
- [ ] I can provide the submitted Word `.docx`.
- [ ] The submission was essentially the original Data in Brief text (use `manuscript_original_dib.tex` as the baseline).
- [ ] Unavailable — accept the in-repo `../manuscript.tex` snapshot as the baseline and note the caveat in the response.

**Notes:**

---

## Q2 — Primary scope / framing of the dataset (single)

Reviewers split on what this benchmark *is* (Ed E3, R1.4, R2.a, R2.d/g).

- [ ] **A hyperparameter-optimization (HPO) benchmark** that happens to use a materials model, with materials-relevant *structure* (noise, multi-fidelity, mixed variables, a synthetic constraint). ⭐
- [ ] A materials-optimization benchmark that mimics real formulation/alloy problems.
- [ ] Both, stated as: "an HPO benchmark engineered to *emulate* the difficulty structure of materials-formulation problems."

**Notes:**

---

## Q3 — How to handle the sum-to-one constraint (multi)

The reviewer-quoted framing (mimics alloy composition) is the main target of
R1.1, R2.d, and Ed E2. Pick all edits to make (T1):

- [ ] Drop any claim that it reproduces alloy physics/thermodynamics. ⭐
- [ ] Re-describe it as a *deliberate synthetic linear-equality constraint* whose purpose is to force optimizers to respect an equality constraint (a known-hard setting). ⭐
- [ ] State precisely which variables it applies to and the exact transform (scale each numeric HP to [0,1], then project onto the sum = 1 simplex). ⭐
- [ ] Add per-parameter **marginal-distribution histograms** (raw Sobol vs post-constraint) to answer R1.3 "is Sobol ≈ uniform?". ⭐
- [ ] Remove the constraint from the *narrative* entirely and present the raw (unconstrained) hyperparameter space, treating the constraint only as an optional benchmark mode.
- [ ] Keep the constraint but move its discussion to a "benchmark modes" subsection.

**Notes (which variables exactly? is the constraint applied in the deposited data or only as an optional transform?):**

---

## Q4 — Noise-model validation (R2.e, single)

Reviewer 2 wants evidence the percentile-rank heteroskedastic model beats a
simpler one (T9).

- [ ] Add a quantitative ablation now: rank-resampling vs Gaussian(μ,σ) per cell, compared by held-out calibration / log-likelihood on the repeats. (More work; strongest rebuttal.)
- [ ] Scope it as a *design choice* with qualitative justification + name it explicit future work. ⭐
- [ ] Remove the heteroskedastic-noise emphasis and present only aggregated means.

**Notes:**

---

## Q5 — Optimizer demonstration on the benchmark (R2.f, single)

Reviewer 2 wants proof an optimizer behaves differently here vs existing
benchmarks (T10).

- [ ] Run a demo now: e.g. Ax/BayBE vs random search **on the surrogate**, report convergence over a fixed budget, contrast with Olympus / particle-packing. (Strongest, most work.)
- [ ] Add a *small* illustrative demo (single optimizer vs random) only. ⭐
- [ ] Defer to future work and instead cite existing downstream uses (HF Space, BayBE/Ax notebooks, Kaggle competition) as evidence of utility.

**Notes:**

---

## Q6 — SHAP Figure 6 (single)

R1.5 says the SHAP description is technically wrong; R2.b says it is
superficial. The figure is not in the repo (see Q1).

- [ ] Keep it, **fix the wording** (SHAP attributes a model prediction relative to an expected value; it does *not* measure error vs experiment), and supply it as vector/print-resolution. ⭐
- [ ] Keep it but deepen the interpretation into actionable HP guidance.
- [ ] Remove it (it is not essential to a Data Descriptor).

**Notes:**

---

## Q7 — Tool primers + researcher-workflow paragraph (multi)

Ed E4 + all of Reviewer 3. Which primers to add, and at what depth (T3)?

- [ ] CrabNet — what it is; why composition-only attention-based property prediction matters. ⭐
- [ ] Matbench — what it is; the `matbench_expt_gap` experimental band-gap task and why it was chosen. ⭐
- [ ] Ax — its role here (Sobol generation; intended BO consumer of the benchmark). ⭐
- [ ] A "status quo → with this benchmark" workflow paragraph for a typical researcher. ⭐
- [ ] Keep primers to ~2–3 sentences each (Data Descriptors are concise).
- [ ] Allow a fuller ~1 paragraph each.

**Notes:**

---

## Q8 — Figure consolidation and fixes (multi)

Editor E5 + our new Pareto figure (T6).

- [ ] Consolidate current Figs 1–5 (histograms) into **one multi-panel** figure. ⭐
- [ ] Add the new **Pareto-front** panel figure (`figures/pareto_rawdata.pdf`) to Technical Validation. ⭐
- [ ] Also add the **surrogate** Pareto figure (`figures/pareto_surrogate.pdf`), or keep it supplementary only.
- [ ] Fix Figure 4 log-y tick labels (explicit decade ticks). ⭐
- [ ] Supply Figure 6 (SHAP) as vector art (depends on Q6/Q1). ⭐

**Notes (raw Pareto in main text, surrogate in SI? or both in main text?):**

---

## Q9 — Which fidelity axes to claim as "multi-fidelity" (multi)

Reviewer 2 disputes the multi-fidelity claim (T4). Select the axes actually
present in the data that we will name and (optionally) demonstrate:

- [ ] Training-set fraction (`train_frac`) — lower = cheaper/lower-fidelity. ⭐ (Edison: legitimate fidelity axis.)
- [ ] Number of epochs (`epochs`/`epochs_step`). ⭐ (Edison: legitimate fidelity axis.)
- [ ] Number of repeats per configuration (statistical fidelity). ⚠️ Edison: **reclassify as noise characterization, not fidelity** — repeats add precision, not a lower-fidelity approximation.
- [ ] Runtime as the *cost* variable (spans ~4 orders of magnitude) linking the above to a multi-fidelity acquisition setting. (Edison: a *cost correlate*, not a fidelity coordinate.)
- [ ] If none can be convincingly demonstrated, **soften** to "supports cost-controlled study" rather than "multi-fidelity".

**Notes:** Edison-recommended fidelity/benchmark citations (YAHPO Gym,
Moosbauer 2022, Do & Zhang 2023, Sabanza-Gil 2025) are staged in
[`edison-rubber-duck.md`](./edison-rubber-duck.md) §4.

---

## Q10 — Limitations / scope-narrowing paragraph (multi)

R2.a/g/h want critical self-assessment (T11).

- [ ] Add an explicit **Limitations** paragraph (single architecture, single task, 47% incomplete runs, synthetic constraint, surrogate approximation error). ⭐
- [ ] Frame the dataset as the **first entry in an extensible benchmark family** (more architectures/tasks to follow). ⭐
- [ ] Expand related-work discussion using the 12 new references already in `references.bib`. ⭐

**Notes:**

---

## Q11 — Funding + authorship (multi)

- [ ] Confirm the Funding Information field will list **NSF DMR-1651668** (Ed E6). ⭐
- [ ] The grant list above is complete (no other funders to add).
- [ ] Add **Xavier Zaitzeff (@XZaitzeff)** as a co-author (he prepared/submitted the manuscript). *Currently the author list is Baird, Parikh, Sparks.*
- [ ] Keep the author list unchanged; acknowledge Xavier in Acknowledgments instead.

**Notes:**

---

## Q12 — Correct the data-accuracy discrepancies (multi)

Found against the public `sobol_regression.csv` (Section D of the plan). These
are checkable by reviewers, so should be fixed regardless.

- [ ] Fix "≈2.6 successful repeats" → **mean 4.17** (173,219 runs / 41,550 sets); update the Figure 1 caption too. ⭐
- [ ] Fix "MAE and RMSE span more than two orders of magnitude" — the deposited data shows <1 decade for MAE/RMSE; **runtime** spans ~4 decades and model size ~1.8. Re-attribute the claim. ⭐
- [ ] Confirm whether the histograms (Figs 1–5) were made from *filtered* or *unfiltered* data; ensure the figures match the deposited CSV (FAIR consistency, Ed E1).

**Notes (source of the 2.6 number? was there an unfiltered figure dataset?):**

---

## Q13 — Re-run modeling + new Zenodo deposit? (single)

Per the PR note: "we'd probably want to do another Zenodo upload if we change
something about the modeling." Some options above (Q4 ablation, Q5 optimizer
demo) add *analysis* without changing the deposited data; others might.

- [ ] No re-run — all planned changes are analysis/prose on the existing deposit (record 7694268). ⭐
- [ ] Re-run/extend modeling and mint a **new Zenodo version** (update DOIs in the manuscript).
- [ ] Undecided — depends on Q4/Q5 outcomes.

**Notes:**

---

## Q14 — Venue (single)

Reviewer 2 suggested resubmitting elsewhere; the Editor invited a revision.

- [ ] Revise and resubmit to **IMMI** (address all comments). ⭐
- [ ] Revise but move to a different data journal (e.g., Scientific Data, Data in Brief).

**Notes:**

---

### Anything else for the agent to action

**Notes:**
