<!-- markdownlint-disable -->
# Edison Scientific rubber-duck — verbatim transcript

Verbatim question + response from an Edison Scientific `LITERATURE_HIGH`
query, retained for provenance. Synthesis and caveats are in
[`edison-rubber-duck.md`](./edison-rubber-duck.md). Edison is an AI
literature agent; verify every DOI/claim before citing.

---

TASK 966ce63d-ba0d-4ed5-9f92-09f5de658596

Question:
I am preparing a point-by-point response and revision for a Data Descriptor
manuscript submitted to the Springer Nature journal Integrating Materials and
Manufacturing Innovation (IMMI). Please act as a critical scientific reviewer
and "rubber duck": stress-test my planned responses for scientific soundness,
flag anything indefensible or overstated, and cite peer-reviewed literature that
would strengthen (or contradict) each point.

DATASET / MANUSCRIPT UNDER REVIEW
- A benchmark DATASET: 173,219 successful CrabNet (composition-only,
  attention-based band-gap regression neural network) training runs over a
  23-dimensional hyperparameter space sampled with a quasi-random Sobol design,
  trained on the Matbench experimental band gap task. Recorded objectives per
  run: MAE, RMSE, GPU runtime, and on-disk model size.
- Repeated identical parameter sets (1-5 repeats) capture heteroskedastic noise;
  within-group percentile ranks are stored instead of assuming Gaussian noise.
- A RandomForest surrogate is released so users can benchmark high-dimensional,
  multi-objective, multi-fidelity, noisy Bayesian optimization (BO) cheaply.
- A "contrived" linear equality constraint (some normalized numerical
  hyperparameters are made to sum to one) is included to emulate composition /
  formulation constraints common in materials optimization.

KEY REVIEWER CRITICISMS
1. The sum-to-one constraint is artificial and its physical motivation is
   overstated; it does not reproduce alloy thermodynamics (multiple reviewers).
2. Unclear whether MAE/RMSE/runtime/model-size exhibit genuine Pareto trade-offs
   or are strongly correlated.
3. Is Sobol sampling equivalent to uniform random sampling? Show marginal
   distributions and how the constraint distorts them.
4. It is a hyperparameter-optimization benchmark built on a materials task, NOT a
   materials-design benchmark; make this explicit.
5. The claim "SHAP analysis analyzes the difference between the true experimental
   output and the expected output of the model" is inaccurate.
6. The "multi-fidelity" claim is not substantiated/defined/demonstrated.
7. The percentile-rank heteroskedastic-noise treatment is not validated against
   simpler noise models (no ablation).
8. Practical utility is not demonstrated: no BO algorithm comparison versus
   established benchmarks (e.g., Olympus, the group's particle-packing benchmark).
9. Need a data dictionary and demonstrated FAIR compliance.
10. Need context on what CrabNet, Matbench, and the Ax platform are, how a typical
    researcher uses them today, and what this benchmark lets them do better.

MY PLANNED RESPONSES (please critique each)
A. Reframe the sum-to-one constraint honestly as a deliberately synthetic linear
   EQUALITY constraint that emulates the *mathematical structure* of composition
   / mixture constraints (simplex-constrained inputs) encountered in formulation
   and alloy design, explicitly NOT a physics/thermodynamics model; keep the word
   "contrived" but clarify it means "synthetic by construction," and specify
   exactly which parameters are normalized.
B. Add a multi-panel Pareto-front analysis for all six objective pairs, computed
   both from the released surrogate and from repeat-averaged raw data, plus a
   correlation matrix, to show genuine trade-offs (e.g., accuracy vs runtime/size)
   and where objectives are correlated.
C. Add marginal-distribution plots and argue Sobol low-discrepancy sampling
   yields approximately uniform marginals with lower discrepancy than i.i.d.
   uniform, and show how the equality constraint reshapes affected marginals.
D. State scope explicitly: this is a high-dimensional multi-objective
   multi-fidelity *optimization* benchmark whose objective happens to be training
   a materials ML model; it is not a materials-property-design benchmark.
E. Correct the SHAP description: SHAP attributes a model prediction to input
   features relative to a baseline/expected prediction; it does NOT measure the
   gap between true experimental values and model expectation. Either fix the
   sentence and provide the figure at print resolution, or remove SHAP.
F. Substantiate "multi-fidelity" by explicitly naming the fidelity axes actually
   present: fraction of training data (train_frac), training epochs, and number
   of repeats, with GPU runtime as the fidelity cost; cite multi-fidelity BO work.
G. Justify percentile-rank noise as a non-parametric, distribution-free model of
   heteroskedastic noise; add at least a qualitative comparison to a Gaussian
   model and mark a full ablation as future work.
H. For practical utility, point to existing downstream use (BayBE, Ax, and a
   Kaggle competition on this surrogate) and add a small illustrative BO run on
   the surrogate; position a full cross-benchmark study as future work.
I. Add a column-level data dictionary and an explicit FAIR statement (Zenodo DOI
   = Findable/Accessible; CSV+JSON = Interoperable; license + provenance +
   metadata = Reusable).
J. Add short primer paragraphs on CrabNet, Matbench, and Ax, each ending with how
   a typical researcher uses the tool today and what this benchmark improves.

QUESTIONS FOR YOU
1. For each planned response A-J, is it scientifically defensible? Where is it
   weak, overclaiming, or likely to draw further reviewer objection?
2. Is naming train_frac / epochs / repeats as fidelity axes (response F) a
   legitimate use of the term "multi-fidelity" in the BO literature? Provide
   supporting citations or contradicting evidence.
3. Is the claim that Sobol sequences give approximately uniform, lower-discrepancy
   marginals versus i.i.d. uniform sampling correct and citable (response C)?
4. Is my SHAP correction (response E) accurate and precisely worded?
5. What is the strongest defensible framing for a synthetic hyperparameter-sweep
   dataset as a materials-relevant OPTIMIZATION benchmark (responses A, D, H)?
6. Recommend specific peer-reviewed references (2019-2025) for: multi-fidelity BO
   in materials, high-dimensional BO, noise-aware / heteroskedastic surrogate
   modeling, Pareto/multi-objective BO, and using ML-training hyperparameter
   response surfaces as optimization benchmarks.
Please give a per-item verdict (defensible / weak / fix) with brief reasoning and
citations.


## Critical Review of Planned Point-by-Point Responses: Per-Item Verdicts, Stress Tests, and Supporting Literature

### 1. Per-Item Verdict Summary

The following table synthesizes the scientific defensibility of each planned response (A–J), identifies the strongest likely reviewer objections, and recommends concrete fixes.

| Response | Verdict | Strength of Evidence | Key Risk/Weakness | Recommended Fix |
|---|---|---|---|---|
| **A — Synthetic sum-to-one constraint** | **Defensible** | **Strong.** Nonnegative component fractions summing to one define the standard simplex geometry used for mixtures and compositions; known constraints are legitimate optimization-domain restrictions (verret2025enhancingflameretardancy pages 5-7, hickman2022bayesianoptimizationwith pages 2-3). | The constrained hyperparameters have no compositional, alloy-thermodynamic, or other physical interpretation. Calling the construction an alloy or formulation model would remain indefensible. | Call it a **synthetic linear equality constraint** that reproduces only the geometry of a simplex-constrained search space. State explicitly that it is not a physics, thermodynamics, phase-stability, or processing model; list the affected parameters, bounds, normalization, and sampling transformation. |
| **B — Correlations and pairwise Pareto fronts** | **Defensible** | **Moderate to strong.** Pareto-front and hypervolume analyses are standard for multiple competing objectives, including noisy materials optimization (pfisterer2022yahpogymanefficient pages 4-6, pfisterer2022yahpogymanefficient pages 23-26). | Pairwise plots alone do not characterize the four-objective Pareto set. Strongly correlated objectives may produce nearly degenerate fronts, and surrogate-derived fronts may contain artifacts or differ from observed fronts. | Add Pearson and Spearman matrices, all six pairwise fronts, and the four-objective nondominated set. Report nondominated counts or fractions, objective directions, tie handling, uncertainty from repeats, and agreement between raw and surrogate fronts. Do not assert genuine trade-offs unless the results demonstrate them. |
| **C — Sobol sampling and marginals** | **Defensible** | **Strong, with caveats.** Sobol points are deterministic low-discrepancy designs with asymptotically uniform coordinate projections. Their discrepancy can approach logarithmic factors times N⁻¹, versus random-sampling behavior near N⁻¹ᐟ² (cheng2013computationalinvestigationof pages 1-2, morokoff1994quasirandomsequencesand pages 1-3, morokoff1994quasirandomsequencesand pages 6-9). | Sobol sampling is not equivalent to i.i.d. uniform sampling: points are dependent and deterministic. Finite prefixes and higher-dimensional projections can still be poor, and constraint enforcement or renormalization may destroy uniform marginals (morokoff1994quasirandomsequencesand pages 1-3, morokoff1994quasirandomsequencesand pages 23-25). | Say that Sobol sampling targets uniform coverage with lower discrepancy, not that it is uniform random sampling. Plot empirical CDFs or histograms before and after the constraint; quantify discrepancy or distributional deviation; show relevant two-dimensional projections; document scrambling, seed, skipped points, sample size, and transformation. Add a high-dimensional finite-sample caveat. |
| **D — Benchmark scope** | **Defensible** | **Strong.** Surrogate response surfaces derived from ML training are established HPO benchmarks; YAHPO Gym explicitly supports multi-objective and multi-fidelity HPO (pfisterer2022yahpogymanefficient pages 1-3, pfisterer2022yahpogymanefficient pages 4-6, pfisterer2022yahpogymanefficient pages 3-4). | The term materials benchmark without qualification could imply that decision variables are compositions, structures, or processing conditions. | Use **materials-relevant HPO or black-box optimization benchmark** throughout. State that inputs are CrabNet hyperparameters, outputs characterize ML training, and the materials connection is the Matbench band-gap task—not direct materials design. Compare explicitly with YAHPO Gym and other HPO surrogate benchmarks. |
| **E — SHAP interpretation** | **Defensible** | **Very strong.** SHAP assigns each feature a Shapley-value contribution to a model prediction relative to a baseline or reference value; the contributions sum to the prediction minus that reference (chen2023algorithmstoestimate pages 13-14, chen2023algorithmstoestimate pages 6-8, chen2023algorithmstoestimate pages 2-4). | Expected prediction is not universally defined: it depends on the background distribution and whether marginal, conditional, or fixed-baseline SHAP is used. SHAP does not establish causality or explain experimental residuals. | Write: **For the specified background distribution and SHAP variant, SHAP decomposes the surrogate prediction into a baseline value plus feature attributions.** Name the explained model, output, explainer, background data, and treatment of correlated features; otherwise remove the figure. |
| **F — Multi-fidelity claim** | **Fix** | **Mixed.** Training-set fraction and training epochs are established resource or fidelity variables in multi-fidelity HPO (moosbauer2022automatedbenchmarkdrivendesign pages 5-7, pfisterer2022yahpogymanefficient pages 4-6, moosbauer2022automatedbenchmarkdrivendesign pages 9-10). Repetition is less canonical and usually improves precision by reducing stochastic noise rather than changing approximation fidelity (do2311multifidelitybayesianoptimization pages 15-18, sabanzagil2025bestpracticesfor pages 9-12). | Merely recording epochs or `train_frac` does not prove a usable multi-fidelity benchmark. Their levels must vary, change cost, remain informative about the full-fidelity target, and be queryable by an optimizer. GPU runtime is a measured cost, not necessarily a fidelity coordinate. | Treat `train_frac` and epochs as **candidate resource or fidelity axes** only after showing levels, costs, rank correlations with full-fidelity performance, and full-fidelity definitions. Reclassify repeats as replication or noise characterization unless users can adaptively purchase additional replicates and receive an aggregate of increasing precision. Demonstrate at least one cost-aware multi-fidelity optimization run. |
| **G — Percentile-rank noise treatment** | **Weak** | **Weak for the specific method.** Literature supports input-dependent heteroskedastic modeling and comparisons with homoskedastic baselines, but does not validate within-group percentile ranks as a general noise model (griffiths2022achievingrobustnessto pages 20-24, griffiths2022achievingrobustnessto pages 1-4, griffiths2022achievingrobustnessto pages 4-7). | Calling percentile ranks a distribution-free model of heteroskedastic noise overstates what ranks provide. With only one to five repeats, percentiles are coarse; singleton groups contain no variance information; ranks discard magnitude and do not by themselves define a predictive noise distribution. | Describe ranks as a **nonparametric descriptive encoding of within-configuration outcomes**, not a validated noise model. Add a quantitative ablation against raw replicates, group means plus empirical variance, and homoskedastic or heteroskedastic Gaussian baselines. Evaluate calibration, held-out likelihood or prediction intervals, and downstream optimizer rankings. |
| **H — Practical utility** | **Weak** | **Moderate precedent, weak manuscript-level demonstration.** Established benchmarks evaluate optimizers quantitatively and test whether surrogate benchmarks preserve real-problem rankings (pfisterer2022yahpogymanefficient pages 21-23, pfisterer2022yahpogymanefficient pages 23-26, pfisterer2022yahpogymanefficient pages 32-33). | BayBE or Ax compatibility and Kaggle use show adoption, not benchmark validity. A single illustrative trajectory may look anecdotal, and a surrogate can distort optimizer rankings. | Run a minimal repeated comparison—at least random or Sobol search versus one standard BO method and, if claiming multi-fidelity, one resource-aware method. Predefine budgets and seeds; report regret or hypervolume versus cost with uncertainty. Validate surrogate fidelity using held-out error, rank correlation, and preferably optimizer-ranking agreement. |
| **I — FAIR statement and dictionary** | **Defensible** | **Strong framework, but compliance is not binary.** FAIR requires persistent identifiers and rich searchable metadata, standardized retrieval, formal and shared representations, licensing, provenance, and community standards (d.2016thefairguiding pages 3-4, d.2016thefairguiding pages 4-5, mons2017cloudyincreasinglyfair; pages 3-5). | A Zenodo DOI alone does not establish Findability and Accessibility, and CSV or JSON alone does not establish semantic Interoperability. FAIR-compliant may overclaim without machine-readable metadata, vocabularies, versioning, checksums, provenance, and schema definitions. | Map evidence to individual subprinciples: F1–F4, A1–A2, I1–I3, and R1–R1.3. Add a column-level schema with units, types, allowed values, missingness, objective direction, identifiers, constraint status, and provenance; include a versioned DOI, license, checksums, code and environment versions, and citation instructions. Say **designed to support FAIR principles** unless formally assessed. |
| **J — CrabNet, Matbench, and Ax primers** | **Defensible** | **Strong.** CrabNet is a composition-only self-attention model; `matbench_expt_gap` is an experimental band-gap regression task; Ax has been used to optimize CrabNet’s 23-dimensional hyperparameter space (baird2203highdimensionalbayesianoptimization pages 1-2, baird2022highdimensionalbayesianoptimization pages 1-2, wang2021compositionallyrestrictedattentionbased pages 1-2). | Primers can blur the distinction between predicting materials properties, tuning the predictor, and designing materials. Dataset counts may differ across Matbench or processed versions and must match the exact release used. | Keep each primer concise and end with current use, role in this dataset, and benchmark benefit. Cite the original CrabNet and Matbench sources plus the peer-reviewed CrabNet–Ax or SAASBO study. Specify the exact Matbench version, fold protocol, sample count after preprocessing, and clarify that Ax is an optimization platform rather than a materials model. |


*Table: Scientific defensibility, evidentiary strength, likely reviewer objections, and concrete revisions for each planned response. The table distinguishes supportable claims from those requiring reclassification or quantitative validation.*

### 2. Detailed Answers to the Six Specific Questions

#### Question 1: Scientific Defensibility of Each Response

Responses **A, B, C, D, E, I, and J** are scientifically defensible with the fixes noted above. Response **F** (multi-fidelity) is partially defensible and requires reclassification of the "repeats" axis. Responses **G** (percentile-rank noise) and **H** (practical utility) are the weakest and most likely to draw further objection unless substantial additional work is included in the revision.

The primary risk across all responses is **overclaiming**: using materials-science language for what is fundamentally a hyperparameter optimization benchmark, claiming multi-fidelity without demonstrating cost-aware optimizer behavior, and calling percentile ranks a "model" of heteroskedastic noise without validation. The strongest revision strategy is to be maximally honest about scope and limitations, which paradoxically strengthens the paper by making its genuine contributions (large-scale hyperparameter sweep, multiple recorded objectives, realistic noise, and a fast surrogate) unassailable.

#### Question 2: Is Naming train_frac / epochs / repeats as Fidelity Axes Legitimate?

**train_frac and epochs: Yes.** Training-set fraction is explicitly identified as a fidelity variable in multi-fidelity HPO by Moosbauer et al. (2022), who formalize it through the FABOLAS framework where the resource parameter r ∈ (0, 1] represents training-set fraction (moosbauer2022automatedbenchmarkdrivendesign pages 5-7). Do and Zhang (2023) explicitly model "the number of training data points and the number of optimization iterations" as two fidelity variables whose combination determines a fidelity level (do2311multifidelitybayesianoptimization pages 25-27). YAHPO Gym defines multi-fidelity benchmarks through ordinal budget parameters such as "fraction of data or an epoch" (pfisterer2022yahpogymanefficient pages 4-6, pfisterer2022yahpogymanefficient pages 3-4). Sabanza-Gil et al. (2025) provide the most authoritative materials-context definition, requiring that multi-fidelity information sources differ in "reliability or accuracy" and have "different associated costs" (sabanzagil2025bestpracticesfor pages 1-3, sabanzagil2025bestpracticesfor pages 9-12).

**Repeats: Weak / Not standard.** Multiple sources note that the defining criterion for a fidelity axis is whether it "indexes different-cost or different-accuracy evaluations of the same underlying objective, with a meaningful highest-fidelity setting" (do2311multifidelitybayesianoptimization pages 25-27). Repeats reduce stochastic noise but do not change the approximation accuracy of the underlying model. Do and Zhang (2023) state that repeats "may be modeled as a fidelity variable only if increasing repeats predictably changes evaluation quality or cost and can be ordered" but note this is "less canonical than training-set fraction, epochs, iterations, sample size, or dataset resolution" (do2311multifidelitybayesianoptimization pages 15-18). Sabanza-Gil et al. (2025) distinguish between fidelity (accuracy of the information source) and noise reduction (sabanzagil2025bestpracticesfor pages 9-12). **Recommendation:** Reclassify repeats as a noise-characterization mechanism, not a fidelity axis. The legitimate fidelity axes are train_frac and epochs; GPU runtime is a measured cost correlate, not itself a fidelity coordinate.

#### Question 3: Sobol Sequences and Approximately Uniform Marginals

**The claim is correct and citable, with important caveats.** Sobol sequences are low-discrepancy quasi-random sequences designed to fill the unit hypercube as uniformly as possible. Their star discrepancy bounds are of order O((log N)^d / N), substantially better than the approximately O(N^{−1/2}) expected discrepancy of i.i.d. uniform random samples for sufficiently large N (kucherenko2008highdimensionalsobols pages 1-5, morokoff1994quasirandomsequencesand pages 1-3, morokoff1994quasirandomsequencesand pages 6-9). Coordinate projections of Sobol sequences are constructed as permutations of the van der Corput sequence and are therefore designed to be uniformly distributed over [0, 1] in the limit (morokoff1994quasirandomsequencesand pages 6-9, morokoff1994quasirandomsequencesand pages 23-25). Kucherenko (2008) confirms that "projecting a d-dimensional LDS onto an s-dimensional coordinate subspace produces an s-dimensional LDS" (kucherenko2008highdimensionalsobols pages 1-5).

**Critical caveats for 23 dimensions:** (1) Sobol sampling is deterministic—points are dependent, not i.i.d.—so the phrase "approximately uniform marginals" must not imply independence (kucherenko2008highdimensionalsobols pages 5-10). (2) The (log N)^d factor grows rapidly with dimension; for d = 23 and moderate N, the theoretical advantage over random sampling may be modest (kucherenko2008highdimensionalsobols pages 1-5, morokoff1994quasirandomsequencesand pages 1-3). (3) Two-dimensional projections can still show clumping or empty regions for finite prefixes (morokoff1994quasirandomsequencesand pages 23-25). (4) Applying the sum-to-one constraint via renormalization or rejection will distort the originally uniform marginals on the constrained parameters, and this distortion must be shown explicitly. **Cite** Morokoff and Caflisch (1994) and Kucherenko (2008) for the theoretical claims, and present empirical marginal histograms or CDFs for the actual sample.

#### Question 4: Is the SHAP Correction Accurate?

**Yes, the correction is accurate and can be stated precisely.** SHAP values decompose a model's prediction for a specific input into a sum of feature-level contributions relative to a baseline or expected prediction. Formally, for a model f with d features, the explanation satisfies the efficiency property: f(x) = φ₀ + Σᵢ φᵢ, where φ₀ is the value of the empty coalition (typically E[f(X)] over a background dataset) and φᵢ is the Shapley-weighted average marginal contribution of feature i (nohara2022explanationofmachine pages 2-3, chen2023algorithmstoestimate pages 13-14, chen2023algorithmstoestimate pages 2-4). SHAP does **not** measure the gap between true experimental values and the model's expected output; it operates entirely within the model's own predictions (chen2023algorithmstoestimate pages 6-8, chen2023algorithmstoestimate pages 4-6). The baseline depends on the SHAP variant used—marginal, conditional, or fixed-baseline—and different choices produce different numerical attributions (chen2023algorithmstoestimate pages 6-8, chen2023algorithmstoestimate pages 4-6).

**Suggested corrected wording:** "SHAP decomposes the surrogate model's prediction for a given input into a baseline value (the expected prediction over the background dataset) plus individual feature attributions. Each attribution quantifies how much that feature's value shifts the prediction away from the baseline, according to Shapley-value weighting over all possible feature coalitions." Cite Chen et al. (2023) in *Nature Machine Intelligence* (chen2023algorithmstoestimate pages 13-14) and the original Lundberg & Lee (2017) NeurIPS paper.

#### Question 5: Strongest Defensible Framing for the Benchmark

The strongest framing positions the dataset as a **high-dimensional, multi-objective surrogate benchmark for hyperparameter optimization**, grounded in a real materials-science ML training task rather than a synthetic function. This framing is directly supported by the YAHPO Gym paradigm, which uses neural-network surrogates trained on prerecorded ML-training performance data as standardized, fast, and reproducible optimization testbeds (pfisterer2022yahpogymanefficient pages 1-3, pfisterer2022yahpogymanefficient pages 4-6, pfisterer2022yahpogymanefficient pages 3-4). The key differentiators that make this benchmark materials-relevant (rather than generic HPO) are: (a) the objective function is training a materials-property-prediction model (CrabNet) on a materials benchmark (Matbench experimental band gap) (baird2203highdimensionalbayesianoptimization pages 1-2, baird2022highdimensionalbayesianoptimization pages 1-2); (b) the synthetic simplex constraint emulates the geometric structure of composition/formulation constraints encountered in materials and chemical optimization (verret2025enhancingflameretardancy pages 5-7, hickman2022bayesianoptimizationwith pages 2-3); and (c) the recorded objectives (MAE, RMSE, GPU runtime, model size) are directly relevant to deploying ML models in materials workflows.

The framing should **not** claim that optimizing CrabNet hyperparameters is equivalent to materials design, that the simplex constraint reproduces alloy thermodynamics, or that the benchmark replaces domain-specific benchmarks like Olympus. Instead, state that it offers a complementary test problem that is higher-dimensional, noisier, and more realistic than standard synthetic functions, while being cheaper and more reproducible than real ML-training experiments.

#### Question 6: Recommended Peer-Reviewed References (2019–2025)

The following table organizes recommended references by topic area, with publication details and their relevance to specific planned responses.

| Topic | Reference | Year | Journal/Venue | Key Relevance |
|---|---|---:|---|---|
| Multi-fidelity BO in materials | Sabanza-Gil et al., “Best Practices for Multi-Fidelity Bayesian Optimization in Materials and Molecular Research,” DOI: 10.48550/arXiv.2410.00544 | 2025 | *Nature Computational Science* | Defines MFBO through information sources of differing cost and reliability; emphasizes validating low-fidelity informativeness and cost advantage before claiming MF utility. (sabanzagil2025bestpracticesfor pages 1-3, sabanzagil2025bestpracticesfor pages 9-12) |
| Multi-fidelity BO in materials | Fare et al., “A Multi-Fidelity Machine Learning Approach to High Throughput Materials Screening,” DOI: 10.1038/s41524-022-00947-9 | 2022 | *npj Computational Materials* | Demonstrates cost-aware fusion of experimental and computational information sources for materials screening and optimization. (fare2022amultifidelitymachine pages 2-3) |
| Multi-fidelity BO in materials | Tran et al., “Multi-Fidelity Machine-Learning with Uncertainty Quantification and Bayesian Optimization for Materials Design: Application to Ternary Random Alloys,” DOI: 10.1063/5.0015672 | 2020 | *Journal of Chemical Physics* | Materials-specific example combining DFT and interatomic-potential fidelities for BO over ternary-alloy composition space. (fare2022amultifidelitymachine pages 2-3) |
| Multi-fidelity BO in materials | Do and Zhang, “Multi-Fidelity Bayesian Optimization in Engineering Design,” DOI: 10.48550/arXiv.2311.13050 | 2023 | arXiv survey/preprint | Useful taxonomy of fidelity variables and MF surrogates; explicitly discusses data quantity and optimization iterations as fidelity coordinates. Not peer reviewed in the retrieved record. (do2311multifidelitybayesianoptimization pages 25-27, do2311multifidelitybayesianoptimization pages 15-18) |
| Multi-fidelity HPO benchmarks | Pfisterer et al., “YAHPO Gym—An Efficient Multi-Objective Multi-Fidelity Benchmark for Hyperparameter Optimization” | 2022 | AutoML Conference / PMLR | Closest precedent: surrogate HPO benchmarks with multiple objectives and fidelity variables such as data fraction and epochs. (pfisterer2022yahpogymanefficient pages 1-3, pfisterer2022yahpogymanefficient pages 4-6, pfisterer2022yahpogymanefficient pages 3-4) |
| Multi-fidelity HPO benchmarks | Moosbauer et al., “Automated Benchmark-Driven Design and Explanation of Hyperparameter Optimizers,” DOI: 10.1109/TEVC.2022.3211336 | 2022 | *IEEE Transactions on Evolutionary Computation* | Formalizes resource-based MF-HPO and uses benchmark-driven ablations; supports training-set fraction as a canonical fidelity variable. (moosbauer2022automatedbenchmarkdrivendesign pages 5-7, moosbauer2022automatedbenchmarkdrivendesign pages 9-10) |
| Multi-fidelity HPO benchmarks | Klein and Hutter, “Tabular Benchmarks for Joint Architecture and Hyperparameter Optimization,” DOI: 10.48550/arXiv.1905.04970 | 2019 | arXiv preprint | Early precedent for reproducible, inexpensive HPO evaluation using prerecorded ML-training response surfaces; supplement with a peer-reviewed surrogate-benchmark citation if required by the journal. (pfisterer2022yahpogymanefficient pages 1-3, pfisterer2022yahpogymanefficient pages 3-4) |
| High-dimensional BO | Daulton et al., “Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces,” DOI: 10.48550/arXiv.2109.10964 | 2022 | NeurIPS / arXiv version | Introduces MORBO, directly relevant to simultaneous high-dimensional and multi-objective optimization. (pfisterer2022yahpogymanefficient pages 4-6, pfisterer2022yahpogymanefficient pages 23-26) |
| High-dimensional BO | Baird et al., “High-Dimensional Bayesian Optimization of 23 Hyperparameters …: A Case Study on CrabNet Using Ax Platform and SAASBO,” DOI: 10.1016/j.commatsci.2022.111505 | 2022 | *Computational Materials Science* | Direct antecedent for the dataset: 23-dimensional CrabNet HPO on the Matbench experimental-band-gap task using Ax and SAASBO. (baird2022highdimensionalbayesianoptimization pages 1-2, baird2022highdimensionalbayesianoptimization pages 10-14, baird2022highdimensionalbayesianoptimization pages 14-16) |
| High-dimensional BO | Santoni et al., “Comparison of High-Dimensional Bayesian Optimization Algorithms on BBOB,” DOI: 10.1145/3670683 | 2024 | *ACM Transactions on Evolutionary Learning and Optimization* | Comparative evaluation showing that high-dimensional BO performance depends strongly on landscape and budget; motivates rigorous multi-algorithm benchmarking. (pfisterer2022yahpogymanefficient pages 3-4, pfisterer2022yahpogymanefficient pages 32-33) |
| Heteroskedastic/noise-aware BO | Griffiths et al., “Achieving Robustness to Aleatoric Uncertainty with Heteroscedastic Bayesian Optimisation,” DOI: 10.1088/2632-2153/ac298c | 2022 | *Machine Learning: Science and Technology* | Models input-dependent aleatoric noise with a heteroskedastic GP and compares against homoskedastic BO, supporting the need for a quantitative noise-model ablation. (griffiths2022achievingrobustnessto pages 20-24, griffiths2022achievingrobustnessto pages 1-4, griffiths2022achievingrobustnessto pages 4-7) |
| Heteroskedastic/noise-aware BO | Foumani et al., “On the Effects of Heterogeneous Errors on Multi-Fidelity Bayesian Optimization,” DOI: 10.48550/arXiv.2309.02771 | 2023 | arXiv preprint | Examines fidelity-specific heterogeneous errors and challenges the assumption that one noise process adequately models all information sources. Not peer reviewed in the retrieved record. (do2311multifidelitybayesianoptimization pages 15-18, sabanzagil2025bestpracticesfor pages 9-12) |
| Multi-objective/Pareto BO | Low et al., “Evolution-Guided Bayesian Optimization for Constrained Multi-Objective Optimization in Self-Driving Labs,” DOI: 10.1038/s41524-024-01274-x | 2024 | *npj Computational Materials* | Materials example using qNEHVI and Pareto-front coverage under constraints; supports reporting hypervolume and feasible-front quality rather than correlations alone. (pfisterer2022yahpogymanefficient pages 4-6, pfisterer2022yahpogymanefficient pages 23-26) |
| Multi-objective/Pareto BO | Daulton et al., “Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces,” DOI: 10.48550/arXiv.2109.10964 | 2022 | NeurIPS / arXiv version | Provides a high-dimensional MOBO reference directly aligned with the benchmark’s combined dimensionality and multiple objectives. (pfisterer2022yahpogymanefficient pages 4-6, pfisterer2022yahpogymanefficient pages 23-26) |
| Composition/simplex constraints | Hickman et al., “Bayesian Optimization with Known Experimental and Design Constraints for Chemistry Applications,” DOI: 10.48550/arXiv.2203.17241 | 2022 | arXiv preprint | Frames BO over a known feasible subset and motivates enforcing design constraints during acquisition optimization; useful conceptual support but not a thermodynamics model. (hickman2022bayesianoptimizationwith pages 2-3) |
| Composition/simplex constraints | Verret et al., “Enhancing Flame Retardancy in Polypropylene Composites: A Bayesian Optimization Approach,” DOI: 10.26434/chemrxiv-2025-hx7pz | 2025 | ChemRxiv preprint | Explicitly formulates formulation fractions on a simplex and enforces the sum-to-one equality by eliminating one variable. Not peer reviewed in the retrieved record. (verret2025enhancingflameretardancy pages 5-7) |
| CrabNet and Matbench | Wang et al., “Compositionally Restricted Attention-Based Network for Materials Property Predictions,” DOI: 10.1038/s41524-021-00545-1 | 2021 | *npj Computational Materials* | Original CrabNet paper: composition-only, structure-agnostic Transformer-style attention for materials-property prediction. (wang2021compositionallyrestrictedattentionbased pages 2-3, wang2021compositionallyrestrictedattentionbased pages 1-2, wang2021compositionallyrestrictedattentionbased pages 2-2) |
| CrabNet and Matbench | Baird et al., “High-Dimensional Bayesian Optimization of 23 Hyperparameters …: A Case Study on CrabNet Using Ax Platform and SAASBO,” DOI: 10.1016/j.commatsci.2022.111505 | 2022 | *Computational Materials Science* | Defines the direct CrabNet–Matbench–Ax context and documents high-dimensional HPO on `matbench_expt_gap`. (baird2203highdimensionalbayesianoptimization pages 1-2, baird2022highdimensionalbayesianoptimization pages 1-2) |
| SHAP | Chen et al., “Algorithms to Estimate Shapley Value Feature Attributions,” DOI: 10.1038/s42256-023-00657-x | 2023 | *Nature Machine Intelligence* | Authoritative modern account of baseline, marginal, and conditional Shapley attributions; supports defining SHAP as prediction decomposition relative to a specified reference. (chen2023algorithmstoestimate pages 13-14, chen2023algorithmstoestimate pages 6-8, chen2023algorithmstoestimate pages 2-4) |
| SHAP | Lundberg and Lee, “A Unified Approach to Interpreting Model Predictions” | 2017 | NeurIPS | Original SHAP formulation unifying additive feature-attribution methods; use alongside Chen et al. for current caveats about baselines and feature dependence. (chen2023algorithmstoestimate pages 13-14, chen2023algorithmstoestimate pages 1-2) |
| Sobol sequences | Morokoff and Caflisch, “Quasi-Random Sequences and Their Discrepancies,” DOI: 10.1137/0915077 | 1994 | *SIAM Journal on Scientific Computing* | Foundational discrepancy analysis: low-discrepancy sequences target asymptotic uniformity and can improve on the approximately N⁻¹ᐟ² behavior of i.i.d. sampling, subject to dimension-dependent caveats. (morokoff1994quasirandomsequencesand pages 1-3, morokoff1994quasirandomsequencesand pages 3-6, morokoff1994quasirandomsequencesand pages 6-9) |
| Sobol sequences | Kucherenko, “High Dimensional Sobol’ Sequences and Their Application” | 2008 | Conference/proceedings record | Discusses projection quality and high-dimensional limitations; useful for qualifying claims about approximately uniform marginals in 23 dimensions. (kucherenko2008highdimensionalsobols pages 1-5, kucherenko2008highdimensionalsobols pages 5-10) |
| FAIR data | Wilkinson et al., “The FAIR Guiding Principles for Scientific Data Management and Stewardship,” DOI: 10.1038/sdata.2016.18 | 2016 | *Scientific Data* | Definitive FAIR reference: persistent identifiers and rich metadata, standardized access, shared representations and vocabularies, licensing, provenance, and community standards. (d.2016thefairguiding pages 3-4, d.2016thefairguiding pages 4-5, d.2016thefairguiding pages 1-2) |
| Olympus benchmark | Häse et al., “Olympus: A Benchmarking Framework for Noisy Optimization and Experiment Planning,” DOI: 10.1088/2632-2153/abedc8 | 2021 | *Machine Learning: Science and Technology* | Establishes realistic, experimentally derived chemistry and materials emulators for quantitative optimizer comparisons; an important comparator for demonstrating practical benchmark utility. (pfisterer2022yahpogymanefficient pages 21-23, pfisterer2022yahpogymanefficient pages 32-33) |
| Olympus benchmark | Hickman et al., “Olympus, Enhanced: Benchmarking Mixed-Parameter and Multi-Objective Optimization in Chemistry and Materials Science” | 2023 | ChemRxiv preprint | Extends Olympus to mixed parameters and multiple objectives; useful comparator for the proposed benchmark’s scope, but the retrieved version is a preprint. (pfisterer2022yahpogymanefficient pages 4-6, pfisterer2022yahpogymanefficient pages 19-21) |


*Table: References supporting—or qualifying—the manuscript’s claims about multi-fidelity, high-dimensional, noisy, constrained, and multi-objective optimization. Preprints are labeled explicitly so they are not misrepresented as peer-reviewed sources.*

### 3. Summary of Critical Flags

Three areas require the most attention in the revision:

1. **Multi-fidelity claim (Response F):** Do not list "repeats" as a fidelity axis without stronger justification. The most defensible version names train_frac and epochs as candidate fidelity dimensions, demonstrates their cost–accuracy tradeoff empirically, and runs at least one cost-aware multi-fidelity optimization experiment on the surrogate (do2311multifidelitybayesianoptimization pages 25-27, moosbauer2022automatedbenchmarkdrivendesign pages 5-7, pfisterer2022yahpogymanefficient pages 4-6).

2. **Percentile-rank noise model (Response G):** This is the least defensible claim because no published precedent validates within-group percentile ranks as a heteroskedastic noise model for BO surrogates. Reframe it as a nonparametric descriptive encoding and provide a quantitative comparison against at least a Gaussian baseline. The heteroskedastic BO literature (griffiths2022achievingrobustnessto pages 20-24, griffiths2022achievingrobustnessto pages 1-4) provides the appropriate methodological comparison point.

3. **Practical utility (Response H):** Established benchmark papers like YAHPO Gym and Olympus validate their surrogates by showing that optimizer rankings on the surrogate match those on real evaluations (pfisterer2022yahpogymanefficient pages 21-23, pfisterer2022yahpogymanefficient pages 32-33). A single illustrative BO trajectory is likely insufficient; running a minimal repeated comparison (e.g., random search vs. one BO method, with predefined budgets and seeds) would substantially strengthen the revision.

The remaining responses (A, B, C, D, E, I, J) are defensible with the specific fixes noted in the verdict table and should not draw significant further objection if the recommended revisions are implemented carefully.

References

1. (verret2025enhancingflameretardancy pages 5-7): Eric Verret, Sophie Duquesne, Anthony Collin, and Martin Stievenard. Enhancing flame retardancy in polypropylene composites: a bayesian optimization approach. ChemRxiv, Apr 2025. URL: https://doi.org/10.26434/chemrxiv-2025-hx7pz, doi:10.26434/chemrxiv-2025-hx7pz. This article has 0 citations.

2. (hickman2022bayesianoptimizationwith pages 2-3): Riley J. Hickman, Matteo Aldeghi, Florian Häse, and Alán Aspuru-Guzik. Bayesian optimization with known experimental and design constraints for chemistry applications. ArXiv, Mar 2022. URL: https://doi.org/10.48550/arxiv.2203.17241, doi:10.48550/arxiv.2203.17241. This article has 118 citations.

3. (pfisterer2022yahpogymanefficient pages 4-6): F Pfisterer, L Schneider, and J Moosbauer. Yahpo gym-an efficient multi-objective multi-fidelity benchmark for hyperparameter optimization. Unknown journal, 2022.

4. (pfisterer2022yahpogymanefficient pages 23-26): F Pfisterer, L Schneider, and J Moosbauer. Yahpo gym-an efficient multi-objective multi-fidelity benchmark for hyperparameter optimization. Unknown journal, 2022.

5. (cheng2013computationalinvestigationof pages 1-2): Jian Cheng and Marek J. Druzdzel. Computational investigation of low-discrepancy sequences in simulation algorithms for bayesian networks. Preprint, Jan 2013. URL: https://doi.org/10.48550/arxiv.1301.3841, doi:10.48550/arxiv.1301.3841. This article has 84 citations.

6. (morokoff1994quasirandomsequencesand pages 1-3): William J. Morokoff and Russel E. Caflisch. Quasi-random sequences and their discrepancies. SIAM J. Sci. Comput., 15:1251-1279, Nov 1994. URL: https://doi.org/10.1137/0915077, doi:10.1137/0915077. This article has 605 citations.

7. (morokoff1994quasirandomsequencesand pages 6-9): William J. Morokoff and Russel E. Caflisch. Quasi-random sequences and their discrepancies. SIAM J. Sci. Comput., 15:1251-1279, Nov 1994. URL: https://doi.org/10.1137/0915077, doi:10.1137/0915077. This article has 605 citations.

8. (morokoff1994quasirandomsequencesand pages 23-25): William J. Morokoff and Russel E. Caflisch. Quasi-random sequences and their discrepancies. SIAM J. Sci. Comput., 15:1251-1279, Nov 1994. URL: https://doi.org/10.1137/0915077, doi:10.1137/0915077. This article has 605 citations.

9. (pfisterer2022yahpogymanefficient pages 1-3): F Pfisterer, L Schneider, and J Moosbauer. Yahpo gym-an efficient multi-objective multi-fidelity benchmark for hyperparameter optimization. Unknown journal, 2022.

10. (pfisterer2022yahpogymanefficient pages 3-4): F Pfisterer, L Schneider, and J Moosbauer. Yahpo gym-an efficient multi-objective multi-fidelity benchmark for hyperparameter optimization. Unknown journal, 2022.

11. (chen2023algorithmstoestimate pages 13-14): Hugh Chen, Ian C. Covert, Scott M. Lundberg, and Su-In Lee. Algorithms to estimate shapley value feature attributions. May 2023. URL: https://doi.org/10.1038/s42256-023-00657-x, doi:10.1038/s42256-023-00657-x. This article has 687 citations and is from a domain leading peer-reviewed journal.

12. (chen2023algorithmstoestimate pages 6-8): Hugh Chen, Ian C. Covert, Scott M. Lundberg, and Su-In Lee. Algorithms to estimate shapley value feature attributions. May 2023. URL: https://doi.org/10.1038/s42256-023-00657-x, doi:10.1038/s42256-023-00657-x. This article has 687 citations and is from a domain leading peer-reviewed journal.

13. (chen2023algorithmstoestimate pages 2-4): Hugh Chen, Ian C. Covert, Scott M. Lundberg, and Su-In Lee. Algorithms to estimate shapley value feature attributions. May 2023. URL: https://doi.org/10.1038/s42256-023-00657-x, doi:10.1038/s42256-023-00657-x. This article has 687 citations and is from a domain leading peer-reviewed journal.

14. (moosbauer2022automatedbenchmarkdrivendesign pages 5-7): Julia Moosbauer, Martin Binder, Lennart Schneider, Florian Pfisterer, Marc Becker, Michel Lang, Lars Kotthoff, and Bernd Bischl. Automated benchmark-driven design and explanation of hyperparameter optimizers. IEEE Transactions on Evolutionary Computation, 26:1336-1350, Dec 2022. URL: https://doi.org/10.1109/tevc.2022.3211336, doi:10.1109/tevc.2022.3211336. This article has 18 citations and is from a highest quality peer-reviewed journal.

15. (moosbauer2022automatedbenchmarkdrivendesign pages 9-10): Julia Moosbauer, Martin Binder, Lennart Schneider, Florian Pfisterer, Marc Becker, Michel Lang, Lars Kotthoff, and Bernd Bischl. Automated benchmark-driven design and explanation of hyperparameter optimizers. IEEE Transactions on Evolutionary Computation, 26:1336-1350, Dec 2022. URL: https://doi.org/10.1109/tevc.2022.3211336, doi:10.1109/tevc.2022.3211336. This article has 18 citations and is from a highest quality peer-reviewed journal.

16. (do2311multifidelitybayesianoptimization pages 15-18): Bach Do and Ruda Zhang. Multi-fidelity bayesian optimization in engineering design. ArXiv, Nov 2311. URL: https://doi.org/10.48550/arxiv.2311.13050, doi:10.48550/arxiv.2311.13050. This article has 20 citations.

17. (sabanzagil2025bestpracticesfor pages 9-12): Víctor Sabanza-Gil, Riccardo Barbano, Daniel Pacheco Gutiérrez, Jeremy S. Luterbacher, José Miguel Hernández-Lobato, Philippe Schwaller, and Loïc Roch. Best practices for multi-fidelity bayesian optimization in materials and molecular research. Nature computational science, 5 7:572-581, Oct 2025. URL: https://doi.org/10.48550/arxiv.2410.00544, doi:10.48550/arxiv.2410.00544. This article has 57 citations and is from a peer-reviewed journal.

18. (griffiths2022achievingrobustnessto pages 20-24): Ryan-Rhys Griffiths, Alexander A Aldrick, Miguel Garcia-Ortegon, Vidhi Lalchand, and Alpha A Lee. Achieving robustness to aleatoric uncertainty with heteroscedastic bayesian optimisation. Machine Learning: Science and Technology, 3:015004, Nov 2022. URL: https://doi.org/10.1088/2632-2153/ac298c, doi:10.1088/2632-2153/ac298c. This article has 63 citations and is from a peer-reviewed journal.

19. (griffiths2022achievingrobustnessto pages 1-4): Ryan-Rhys Griffiths, Alexander A Aldrick, Miguel Garcia-Ortegon, Vidhi Lalchand, and Alpha A Lee. Achieving robustness to aleatoric uncertainty with heteroscedastic bayesian optimisation. Machine Learning: Science and Technology, 3:015004, Nov 2022. URL: https://doi.org/10.1088/2632-2153/ac298c, doi:10.1088/2632-2153/ac298c. This article has 63 citations and is from a peer-reviewed journal.

20. (griffiths2022achievingrobustnessto pages 4-7): Ryan-Rhys Griffiths, Alexander A Aldrick, Miguel Garcia-Ortegon, Vidhi Lalchand, and Alpha A Lee. Achieving robustness to aleatoric uncertainty with heteroscedastic bayesian optimisation. Machine Learning: Science and Technology, 3:015004, Nov 2022. URL: https://doi.org/10.1088/2632-2153/ac298c, doi:10.1088/2632-2153/ac298c. This article has 63 citations and is from a peer-reviewed journal.

21. (pfisterer2022yahpogymanefficient pages 21-23): F Pfisterer, L Schneider, and J Moosbauer. Yahpo gym-an efficient multi-objective multi-fidelity benchmark for hyperparameter optimization. Unknown journal, 2022.

22. (pfisterer2022yahpogymanefficient pages 32-33): F Pfisterer, L Schneider, and J Moosbauer. Yahpo gym-an efficient multi-objective multi-fidelity benchmark for hyperparameter optimization. Unknown journal, 2022.

23. (d.2016thefairguiding pages 3-4): Mark D. Wilkinson, Michel Dumontier, I. Jsbrand Jan Aalbersberg, Gabrielle Appleton, Myles Axton, Arie Baak, Niklas Blomberg, Jan-Willem Boiten, Luiz Bonino da Silva Santos, Philip E. Bourne, Jildau Bouwman, Anthony J. Brookes, Tim Clark, Mercè Crosas, Ingrid Dillo, Olivier Dumon, Scott Edmunds, Chris T. Evelo, Richard Finkers, Alejandra Gonzalez-Beltran, Alasdair J. G. Gray, Paul Groth, Carole Goble, Jeffrey S. Grethe, Jaap Heringa, Rob Hooft, Tobias Kuhn, Ruben Kok, Joost Kok, Scott J. Lusher, Maryann E. Martone, Albert Mons, Abel L. Packer, Bengt Persson, Philippe Rocca-Serra, Marco Roos, Rene van Schaik, Susanna-Assunta Sansone, Erik Schultes, Thierry Sengstag, Ted Slater, George Strawn, Morris A. Swertz, Mark Thompson, Johan van der Lei, Erik van Mulligen, Jan Velterop, Andra Waagmeester, Peter Wittenburg, Katherine Wolstencroft, Jun Zhao, and Barend Mons. The fair guiding principles for scientific data management and stewardship. Scientific Data, Mar 2016. URL: https://doi.org/10.1038/sdata.2016.18, doi:10.1038/sdata.2016.18. This article has 24674 citations and is from a peer-reviewed journal.

24. (d.2016thefairguiding pages 4-5): Mark D. Wilkinson, Michel Dumontier, I. Jsbrand Jan Aalbersberg, Gabrielle Appleton, Myles Axton, Arie Baak, Niklas Blomberg, Jan-Willem Boiten, Luiz Bonino da Silva Santos, Philip E. Bourne, Jildau Bouwman, Anthony J. Brookes, Tim Clark, Mercè Crosas, Ingrid Dillo, Olivier Dumon, Scott Edmunds, Chris T. Evelo, Richard Finkers, Alejandra Gonzalez-Beltran, Alasdair J. G. Gray, Paul Groth, Carole Goble, Jeffrey S. Grethe, Jaap Heringa, Rob Hooft, Tobias Kuhn, Ruben Kok, Joost Kok, Scott J. Lusher, Maryann E. Martone, Albert Mons, Abel L. Packer, Bengt Persson, Philippe Rocca-Serra, Marco Roos, Rene van Schaik, Susanna-Assunta Sansone, Erik Schultes, Thierry Sengstag, Ted Slater, George Strawn, Morris A. Swertz, Mark Thompson, Johan van der Lei, Erik van Mulligen, Jan Velterop, Andra Waagmeester, Peter Wittenburg, Katherine Wolstencroft, Jun Zhao, and Barend Mons. The fair guiding principles for scientific data management and stewardship. Scientific Data, Mar 2016. URL: https://doi.org/10.1038/sdata.2016.18, doi:10.1038/sdata.2016.18. This article has 24674 citations and is from a peer-reviewed journal.

25. (mons2017cloudyincreasinglyfair; pages 3-5): Barend Mons, Cameron Neylon, Jan Velterop, Michel Dumontier, Luiz Olavo Bonino da Silva Santos, and Mark D. Wilkinson. Cloudy, increasingly fair; revisiting the fair data guiding principles for the european open science cloud. Information Services and Use, 37:49-56, Feb 2017. URL: https://doi.org/10.3233/isu-170824, doi:10.3233/isu-170824. This article has 666 citations and is from a peer-reviewed journal.

26. (baird2203highdimensionalbayesianoptimization pages 1-2): Sterling G. Baird, Marianne Liu, and Taylor D. Sparks. High-dimensional bayesian optimization of hyperparameters for an attention-based network to predict materials property: a case study on crabnet using ax and saasbo. Text, Jan 2203. URL: https://doi.org/10.48550/arxiv.2203.12597, doi:10.48550/arxiv.2203.12597. This article has 2 citations and is from a peer-reviewed journal.

27. (baird2022highdimensionalbayesianoptimization pages 1-2): Sterling G. Baird, Marianne Liu, and Taylor D. Sparks. High-dimensional bayesian optimization of 23 hyperparameters over 100 iterations for an attention-based network to predict materials property: a case study on crabnet using ax platform and saasbo. Computational Materials Science, 211:111505, Aug 2022. URL: https://doi.org/10.1016/j.commatsci.2022.111505, doi:10.1016/j.commatsci.2022.111505. This article has 35 citations and is from a domain leading peer-reviewed journal.

28. (wang2021compositionallyrestrictedattentionbased pages 1-2): Anthony Yu-Tung Wang, Steven K. Kauwe, Ryan J. Murdock, and Taylor D. Sparks. Compositionally restricted attention-based network for materials property predictions. npj Computational Materials, May 2021. URL: https://doi.org/10.1038/s41524-021-00545-1, doi:10.1038/s41524-021-00545-1. This article has 354 citations and is from a peer-reviewed journal.

29. (do2311multifidelitybayesianoptimization pages 25-27): Bach Do and Ruda Zhang. Multi-fidelity bayesian optimization in engineering design. ArXiv, Nov 2311. URL: https://doi.org/10.48550/arxiv.2311.13050, doi:10.48550/arxiv.2311.13050. This article has 20 citations.

30. (sabanzagil2025bestpracticesfor pages 1-3): Víctor Sabanza-Gil, Riccardo Barbano, Daniel Pacheco Gutiérrez, Jeremy S. Luterbacher, José Miguel Hernández-Lobato, Philippe Schwaller, and Loïc Roch. Best practices for multi-fidelity bayesian optimization in materials and molecular research. Nature computational science, 5 7:572-581, Oct 2025. URL: https://doi.org/10.48550/arxiv.2410.00544, doi:10.48550/arxiv.2410.00544. This article has 57 citations and is from a peer-reviewed journal.

31. (kucherenko2008highdimensionalsobols pages 1-5): S Kucherenko. High dimensional sobol's sequences and their application. Unknown journal, 2008.

32. (kucherenko2008highdimensionalsobols pages 5-10): S Kucherenko. High dimensional sobol's sequences and their application. Unknown journal, 2008.

33. (nohara2022explanationofmachine pages 2-3): Yasunobu Nohara, Koutarou Matsumoto, Hidehisa Soejima, and Naoki Nakashima. Explanation of machine learning models using shapley additive explanation and application for real data in hospital. Computer Methods and Programs in Biomedicine, 214:106584, Feb 2022. URL: https://doi.org/10.1016/j.cmpb.2021.106584, doi:10.1016/j.cmpb.2021.106584. This article has 1324 citations and is from a peer-reviewed journal.

34. (chen2023algorithmstoestimate pages 4-6): Hugh Chen, Ian C. Covert, Scott M. Lundberg, and Su-In Lee. Algorithms to estimate shapley value feature attributions. May 2023. URL: https://doi.org/10.1038/s42256-023-00657-x, doi:10.1038/s42256-023-00657-x. This article has 687 citations and is from a domain leading peer-reviewed journal.

35. (fare2022amultifidelitymachine pages 2-3): Clyde Fare, Peter Fenner, Matthew Benatan, Alessandro Varsi, and Edward O. Pyzer-Knapp. A multi-fidelity machine learning approach to high throughput materials screening. npj Computational Materials, 8:1-9, Dec 2022. URL: https://doi.org/10.1038/s41524-022-00947-9, doi:10.1038/s41524-022-00947-9. This article has 89 citations and is from a peer-reviewed journal.

36. (baird2022highdimensionalbayesianoptimization pages 10-14): Sterling G. Baird, Marianne Liu, and Taylor D. Sparks. High-dimensional bayesian optimization of 23 hyperparameters over 100 iterations for an attention-based network to predict materials property: a case study on crabnet using ax platform and saasbo. Computational Materials Science, 211:111505, Aug 2022. URL: https://doi.org/10.1016/j.commatsci.2022.111505, doi:10.1016/j.commatsci.2022.111505. This article has 35 citations and is from a domain leading peer-reviewed journal.

37. (baird2022highdimensionalbayesianoptimization pages 14-16): Sterling G. Baird, Marianne Liu, and Taylor D. Sparks. High-dimensional bayesian optimization of 23 hyperparameters over 100 iterations for an attention-based network to predict materials property: a case study on crabnet using ax platform and saasbo. Computational Materials Science, 211:111505, Aug 2022. URL: https://doi.org/10.1016/j.commatsci.2022.111505, doi:10.1016/j.commatsci.2022.111505. This article has 35 citations and is from a domain leading peer-reviewed journal.

38. (wang2021compositionallyrestrictedattentionbased pages 2-3): Anthony Yu-Tung Wang, Steven K. Kauwe, Ryan J. Murdock, and Taylor D. Sparks. Compositionally restricted attention-based network for materials property predictions. npj Computational Materials, May 2021. URL: https://doi.org/10.1038/s41524-021-00545-1, doi:10.1038/s41524-021-00545-1. This article has 354 citations and is from a peer-reviewed journal.

39. (wang2021compositionallyrestrictedattentionbased pages 2-2): Anthony Yu-Tung Wang, Steven K. Kauwe, Ryan J. Murdock, and Taylor D. Sparks. Compositionally restricted attention-based network for materials property predictions. npj Computational Materials, May 2021. URL: https://doi.org/10.1038/s41524-021-00545-1, doi:10.1038/s41524-021-00545-1. This article has 354 citations and is from a peer-reviewed journal.

40. (chen2023algorithmstoestimate pages 1-2): Hugh Chen, Ian C. Covert, Scott M. Lundberg, and Su-In Lee. Algorithms to estimate shapley value feature attributions. May 2023. URL: https://doi.org/10.1038/s42256-023-00657-x, doi:10.1038/s42256-023-00657-x. This article has 687 citations and is from a domain leading peer-reviewed journal.

41. (morokoff1994quasirandomsequencesand pages 3-6): William J. Morokoff and Russel E. Caflisch. Quasi-random sequences and their discrepancies. SIAM J. Sci. Comput., 15:1251-1279, Nov 1994. URL: https://doi.org/10.1137/0915077, doi:10.1137/0915077. This article has 605 citations.

42. (d.2016thefairguiding pages 1-2): Mark D. Wilkinson, Michel Dumontier, I. Jsbrand Jan Aalbersberg, Gabrielle Appleton, Myles Axton, Arie Baak, Niklas Blomberg, Jan-Willem Boiten, Luiz Bonino da Silva Santos, Philip E. Bourne, Jildau Bouwman, Anthony J. Brookes, Tim Clark, Mercè Crosas, Ingrid Dillo, Olivier Dumon, Scott Edmunds, Chris T. Evelo, Richard Finkers, Alejandra Gonzalez-Beltran, Alasdair J. G. Gray, Paul Groth, Carole Goble, Jeffrey S. Grethe, Jaap Heringa, Rob Hooft, Tobias Kuhn, Ruben Kok, Joost Kok, Scott J. Lusher, Maryann E. Martone, Albert Mons, Abel L. Packer, Bengt Persson, Philippe Rocca-Serra, Marco Roos, Rene van Schaik, Susanna-Assunta Sansone, Erik Schultes, Thierry Sengstag, Ted Slater, George Strawn, Morris A. Swertz, Mark Thompson, Johan van der Lei, Erik van Mulligen, Jan Velterop, Andra Waagmeester, Peter Wittenburg, Katherine Wolstencroft, Jun Zhao, and Barend Mons. The fair guiding principles for scientific data management and stewardship. Scientific Data, Mar 2016. URL: https://doi.org/10.1038/sdata.2016.18, doi:10.1038/sdata.2016.18. This article has 24674 citations and is from a peer-reviewed journal.

43. (pfisterer2022yahpogymanefficient pages 19-21): F Pfisterer, L Schneider, and J Moosbauer. Yahpo gym-an efficient multi-objective multi-fidelity benchmark for hyperparameter optimization. Unknown journal, 2022.
