# IMMI Data Descriptor — CrabNet Hyperparameter Benchmark

LaTeX project for the *Integrating Materials and Manufacturing Innovation*
(IMMI, Springer Nature) **Data Descriptor** submission of the CrabNet
hyperparameter benchmark dataset (Zenodo DOI:
[10.5281/zenodo.7694268](https://doi.org/10.5281/zenodo.7694268)).

This is a 1:1 port of the prior Data in Brief manuscript at
[`reports/crabnet_hyperparameter/Datainbrief.docx`](../crabnet_hyperparameter/Datainbrief.docx)
into the official Springer Nature LaTeX template (`sn-jnl.cls`), so the
scientific rewrite for the IMMI venue can proceed directly in LaTeX.

## Layout

| Path                    | Purpose                                              |
| ----------------------- | ---------------------------------------------------- |
| `manuscript.tex`        | Main manuscript (sn-jnl, sn-mathphys-num style)      |
| `references.bib`        | Bibliography (mirror of sibling folder, regenerated) |
| `figures/*.png`         | Figures 1–5 (copied from sibling folder)             |
| `sn-jnl.cls`            | Springer Nature class file (vendored, see LICENSE)   |
| `sn-mathphys-num.bst`   | Numbered Math/Phys reference style (vendored)        |
| `latexmkrc`             | `latexmk` configuration                              |
| `LICENSE.sn-jnl`        | LPPL license for the vendored Springer files         |

The `references.bib` here is a copy of
[`../crabnet_hyperparameter/references.bib`](../crabnet_hyperparameter/references.bib)
and is documented in
[`../crabnet_hyperparameter/references-recommendations.md`](../crabnet_hyperparameter/references-recommendations.md).

## Build

Requires a TeX Live distribution with the `latex-recommended`,
`latex-extra`, `science`, `pictures`, `publishers`, `bibtex-extra`,
and `fonts-recommended` collections. On Debian / Ubuntu:

```bash
sudo apt-get install -y --no-install-recommends \
    texlive-latex-base texlive-latex-recommended texlive-latex-extra \
    texlive-fonts-recommended texlive-bibtex-extra texlive-science \
    texlive-pictures texlive-publishers latexmk biber ghostscript
```

Then from this directory:

```bash
latexmk            # produces manuscript.pdf
latexmk -c         # remove auxiliary files (keeps the PDF)
latexmk -C         # remove the PDF as well
```

Per IMMI / Springer Nature submission guidance, **the LaTeX source files
are required before a manuscript can be accepted** and the PDF is
re-built by Editorial Manager (TeX Live 2018) at submission time. A
locally-built copy of `manuscript.pdf` is committed alongside the
source for convenience; rebuild with `latexmk` after any edit.

## Why `sn-jnl` and `sn-mathphys-num`

The IMMI Author Instructions
([2025-08 PDF](https://media.springer.com/full/springer-instructions-for-authors-assets/pdf/40192_Integrating%20Materials%20and%20Manufacturing%20Innov_Aug_2025.pdf))
say:

> Manuscripts with mathematical content can also be submitted in LaTeX
> format. […] Templates are available at LaTeX2e macro package.

That "LaTeX2e macro package" is the Springer Nature `sn-jnl` template,
which is the single template the publisher maintains for all of its
journals (Springer, Nature Portfolio, BMC). The IMMI guide also
mandates **numbered, ascending in-text citations** in square brackets
("Reference citations in the text should be identified by numbers in
square brackets and should be in ascending numerical order"); among the
reference styles bundled with `sn-jnl`, `sn-mathphys-num` is the
numbered Math/Phys style and is the closest match.

## Article-type sizing

IMMI Data Descriptor articles (a sub-class of Technical Articles):

* ≤ 8,000 words.
* Abstract 150–250 words; 4–6 keywords.
* American English spelling.
* The dataset must be deposited in a public repository with a
  persistent identifier — satisfied by Zenodo
  [10.5281/zenodo.7694268](https://doi.org/10.5281/zenodo.7694268) (and
  the source release [10.5281/zenodo.7694289](https://doi.org/10.5281/zenodo.7694289)).

## Vendored Springer files

`sn-jnl.cls`, `sn-mathphys-num.bst`, and `LICENSE.sn-jnl` are
distributed by Springer Nature under the LaTeX Project Public License
(LPPL) and are vendored here at the version available 2026-05-06 from
the Springer-Nature-published mirror at
<https://github.com/DanySK/template-latex-springer-nature-sn-jnl>. They
are not modified.

The current Springer Nature template (December 2024 version, ~880 KB
ZIP) is also available directly from Springer Nature at
<https://www.springernature.com/gp/authors/campaigns/latex-author-support>;
update the vendored files from there if a newer release is needed for
the final submission.
