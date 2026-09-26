<!-- markdownlint-disable -->
# IMMI Data Descriptor — revision support

Support artifacts for responding to the *Integrating Materials and Manufacturing
Innovation* (IMMI) reviews of the CrabNet hyperparameter Data Descriptor. These
files are **scaffolding + analysis + questions** — they intentionally do **not**
rewrite `../manuscript.tex`. The authors drive the manuscript edits after
answering [`author-questions.md`](./author-questions.md).

## Start here

1. **[`author-questions.md`](./author-questions.md)** — answer these first; they
   unblock every editorial decision (multiple-choice / multi-select with ⭐
   recommendations, plus a direct GitHub edit link at the top).
2. **[`review-response-plan.md`](./review-response-plan.md)** — the master plan:
   cross-cutting themes (T1–T13), a point-by-point matrix for the Editor and all
   three reviewers, the Pareto/correlation analysis, and the data-discrepancy
   audit.
3. **[`edison-rubber-duck.md`](./edison-rubber-duck.md)** — an independent
   literature critique of the plan (verdict table, ready-to-adapt wording, and a
   referenced reading list). Verbatim transcript in
   [`edison-rubber-duck-transcript.md`](./edison-rubber-duck-transcript.md).

## Versions

| Version | Where | Notes |
|---------|-------|-------|
| Submitted to IMMI (IMMJ-S-26-00176) | `original_submission/CrabNet_Hyperparameters_IMMI_submission.docx`, transcribed to `manuscript_v0_submitted.tex` (+ `.pdf`) | The version the reviewers read: "contrived constraint" text and the SHAP Figure 6 (`../figures/submitted/`). This is the latexdiff baseline. |
| Xavier's edits (commit 79e44be) | `../../../CrabNet_Hyperparameters.docx`, text snapshot in `original_submission/xavier_edits_79e44be.md` | CrabNet / Matbench / Ax primers and constraint justification; merged into `../manuscript.tex` with his wording kept. Diff his next version against this snapshot. |
| Revision | `../manuscript.tex` (+ `.pdf`) | Current revised manuscript. |
| Original Data in Brief | `manuscript_original_dib.tex` | Earlier venue, kept for reference. |

## File index

| File | Purpose |
|------|---------|
| `author-questions.md` | Author decisions (answered). |
| `review-response-plan.md` | Master plan / point-by-point matrix. |
| `edison-rubber-duck.md`, `edison-rubber-duck-transcript.md` | Independent literature critique of the plan. |
| `edison/` | Edison query for the small-proxy to full-scale tuning reference (muP; FABOLAS). |
| `new_refs_staging.bib` | Verified BibTeX entries merged into `../references.bib`. |
| `analysis/` | JSON/CSV numbers behind the new figures (data checks, fidelity, noise ablation, marginals, optimizer comparison). |
| `response-to-reviewers.tex` (+ `.pdf`) | Point-by-point response; open items marked [TODO]. |
| `manuscript_v0_submitted.tex` (+ `.pdf`) | Submitted version, latexdiff baseline. |
| `manuscript_diff.pdf` | Tracked changes, submitted vs revision. |
| `make-tracked-changes.sh` | Rebuilds `manuscript_diff.pdf`. |

## Toolchain

The LaTeX + conversion tooling (Debian/Ubuntu):

```
sudo apt-get update
sudo apt-get install -y latexmk latexdiff pandoc libalgorithm-diff-perl \
  texlive-latex-base texlive-latex-recommended texlive-latex-extra \
  texlive-fonts-recommended texlive-science
```

## Common tasks

Compile the point-by-point response (standalone, ~4 pages):

```
latexmk -pdf response-to-reviewers.tex
```

Produce the tracked-changes (latexdiff) markup PDF — run from **this** folder;
the script builds inside the manuscript directory so the relative
`\graphicspath` and `\bibliography` resolve:

```
bash make-tracked-changes.sh                      # baseline vs ../manuscript.tex
bash make-tracked-changes.sh OLD.tex NEW.tex      # explicit pair
```

Regenerate the Pareto-front panel figures used in the Technical Validation
discussion (writes PNG@200dpi + vector PDF into `../figures/`):

```
python3 ../../../scripts/crabnet_hyperparameter/plot_pareto_fronts.py \
  --data-dir /path/to/zenodo_7694268
```

The script resolves the dataset from the repo, a local cache, or a fresh Zenodo
download (record `10.5281/zenodo.7694268`), builds the surrogate feature matrix,
repeat-averages the raw runs, and renders both the raw-data and surrogate Pareto
panels (`../figures/pareto_rawdata.*`, `../figures/pareto_surrogate.*`).

## Suggested order of operations

1. Answer `author-questions.md` (especially **Q1** — supply the real submitted
   files).
2. If available, replace `manuscript_v0_submitted.tex` with the true submitted
   `.tex`.
3. Edit `../manuscript.tex` per the answers.
4. `bash make-tracked-changes.sh` → `manuscript_diff.pdf`.
5. Populate `response-to-reviewers.tex` from the plan + the tracked-changes PDF.
