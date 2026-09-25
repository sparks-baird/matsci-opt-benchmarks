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

## Provenance caveat (important)

Three manuscript versions exist and they differ:

| Version | Where | Distinguishing content |
|---------|-------|------------------------|
| Original Data in Brief | `../../crabnet_hyperparameter/Datainbrief.{docx,pdf}` → frozen here as `manuscript_original_dib.tex` | Has the "contrived constraint" prose; Figs 1–5; no SHAP. |
| Repo Data Descriptor | `../manuscript.tex` → frozen here as `manuscript_v0_submitted.tex` | Springer Nature DD structure; dropped the constraint sentence; added "orders of magnitude" claims; no SHAP. |
| Authors' submitted PDF | **not in the repo** | Reviewers quote *both* the constraint prose *and* a SHAP "Figure 6". |

Because the literal submitted source is not in the repo, `latexdiff` here compares
against the **pre-revision repo snapshot**, not the true submission. If the authors
have the real submitted `.tex`, drop it in as `manuscript_v0_submitted.tex` before
running the tracked-changes script (see **Q1**).

## File index

| File | Purpose |
|------|---------|
| `author-questions.md` | Decisions needed from the authors. **Answer first.** |
| `review-response-plan.md` | Master plan / point-by-point matrix / analysis. |
| `edison-rubber-duck.md` | Independent literature critique (synthesis). |
| `edison-rubber-duck-transcript.md` | Verbatim Edison transcript (provenance). |
| `response-to-reviewers.tex` | Placeholder point-by-point response; reviewer text quoted verbatim, author responses stubbed with `\todo`. Compiles standalone. |
| `manuscript_v0_submitted.tex` | Frozen pre-revision `../manuscript.tex` (latexdiff baseline). |
| `manuscript_original_dib.tex` | Pandoc conversion of the original DIB `.docx` (worst-case reference baseline). |
| `make-tracked-changes.sh` | `latexdiff(baseline, ../manuscript.tex)` → `manuscript_diff.pdf`. |

`manuscript_diff.tex` / `manuscript_diff.pdf` are build products and are
git-ignored.

## Toolchain

The LaTeX + conversion tooling (Debian/Ubuntu):

```
sudo apt-get update
sudo apt-get install -y latexmk latexdiff pandoc \
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
