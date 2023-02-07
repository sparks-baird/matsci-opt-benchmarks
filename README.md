[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Built Status](https://api.cirrus-ci.com/github/<USER>/matsci-opt-benchmarks.svg?branch=main)](https://cirrus-ci.com/github/<USER>/matsci-opt-benchmarks)
[![ReadTheDocs](https://readthedocs.org/projects/matsci-opt-benchmarks/badge/?version=latest)](https://matsci-opt-benchmarks.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/matsci-opt-benchmarks/main.svg)](https://coveralls.io/r/<USER>/matsci-opt-benchmarks)
[![PyPI-Server](https://img.shields.io/pypi/v/matsci-opt-benchmarks.svg)](https://pypi.org/project/matsci-opt-benchmarks/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/matsci-opt-benchmarks.svg)](https://anaconda.org/conda-forge/matsci-opt-benchmarks)
[![Monthly Downloads](https://pepy.tech/badge/matsci-opt-benchmarks/month)](https://pepy.tech/project/matsci-opt-benchmarks)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/matsci-opt-benchmarks)
-->

# matsci-opt-benchmarks (WIP)

> A collection of benchmarking problems and datasets for testing the performance of
> advanced optimization algorithms in the field of materials science and chemistry for a
> variety of "hard" problems involving one or several of: constraints, heteroskedasticity,
> multiple objectives, multiple fidelities, and high-dimensionality.

There are already materials-science-specific resources related to datasets, surrogate models, and benchmarks out there:
- [Matbench](https://github.com/materialsproject/matbench) focuses on materials property
prediction using composition and/or crystal structure
- [Olympus](https://github.com/aspuru-guzik-group/olympus) focuses on small datasets
generated via experimental self-driving laboratories
- [Foundry](https://github.com/MLMI2-CSSI/foundry) focuses on delivering ML-ready datasets in materials science and chemistry
- [Matbench-genmetrics](https://github.com/sparks-baird/matbench-genmetrics) focuses on generative modeling for crystal
structure using metrics inspired by [guacamol](https://www.benevolent.com/guacamol) and
[CDVAE](https://github.com/txie-93/cdvae)

In March 2021, [pymatgen](https://github.com/materialsproject/pymatgen) reorganized the
code into [namespace
packages](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/),
which makes it easier to distribute a collection of related subpackages and modules
under an umbrella project. Tangent to that, [PyScaffold](https://pyscaffold.org/) is a project generator for high-quality Python
packages, ready to be shared on PyPI and installable via pip; coincidentally,
it also supports namespace package configurations. My plan for this
repository is to host
`pip`-installable packages that allow for loading datasets, surrogate
models, and benchmarks for recent manuscripts I've
written. It is primarily intended as a convenience for me, with a secondary benefit of
adding value to the community. I will look into hosting the datasets via Foundry and
using the surrogate model API via Olympus. I will likely do logging to a
[MongoDB](https://www.mongodb.com/)
database via [Atlas](https://www.mongodb.com/cloud/atlas) and later take a snapshot of
the dataset for Foundry. Initially, I will probably use a basic [scikit-learn](https://scikit-learn.org/) model, such
as
[RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
or [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html),
along with cross-validated hyperparameter optimization via
[RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
or
[HalvingRandomSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html)
for the surrogate model.

What will really differentiate the contribution of this
repository is *the modeling of heteroskedastic noise*, where the noise variance
can be a complex function of the input parameters. This is contrasted with
homoskedasticity, where the noise variance for a given parameter is fixed
[[Wikipedia](https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity)].

My goal is to win a ["Turing test"](https://en.wikipedia.org/wiki/Turing_test)
of sorts for the surrogate model, where the model is indistinguishable from the true,
underlying objective function.

To accomplish this, I plan to:
- run ~10 repeats for every set of parameters and fit separate models for quantiles
  of the noise distribution
- Get a large enough quasi-random sampling of the search space to accurately model intricate interactions between parameters (i.e. the response surface)
- Train a classification model that short-circuits the regression model: return NaN
  values for inaccessible regions of objective functions and return the regression
  model values for accessible regions


My plans for implementation include:
- packing fraction of a random 3D packing of spheres as a function of the number of
  spheres, 6 parameters that define three separate truncated log-normal
  distributions, and 3 parameters that define the weight fractions
  [[code](https://github.com/sparks-baird/bayes-opt-particle-packing)]
  [[paper1](https://github.com/sparks-baird/bayes-opt-particle-packing/blob/main/paper/main.pdf)] [[paper2](https://doi.org/10.26434/chemrxiv-2023-fjjk7)] [[data ![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7513019.svg)](https://doi.org/10.5281/zenodo.7513019)]
- discrete intensity vs. wavelength spectra (measured experimentally via a
  spectrophotometer) as a function of red, green, and blue LED powers and three sensor
  settings: number of integration steps, integration time per step, and signal gain
  [[code](https://github.com/sparks-baird/self-driving-lab-demo)]
  [[paper](https://doi.org/10.1016/j.matt.2022.11.007)]
- Two error metrics (RMSE and MAE) and two hardware performance metrics (runtime and
  memory) of a [CrabNet](https://github.com/sparks-baird/CrabNet) regression model
  trained on [the Matbench experimental band gap dataset](https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_expt_gap/)
  as a function of 23 CrabNet hyperparameters
  [[code](https://github.com/sparks-baird/crabnet-hyperparameter)]
  [[paper](https://doi.org/10.1016/j.commatsci.2022.111505)]


## Installation

In order to set up the necessary environment:

1. review and uncomment what you need in `environment.yml` and create an environment `matsci-opt-benchmarks` with the help of [conda]:
   ```
   conda env create -f environment.yml
   ```
2. activate the new environment with:
   ```
   conda activate matsci-opt-benchmarks
   ```

> **_NOTE:_**  The conda environment will have matsci-opt-benchmarks installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.


Optional and needed only once after `git clone`:

3. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

4. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.


Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact reproduction of your
   environment with:
   ```bash
   conda env export -n matsci-opt-benchmarks -f environment.lock.yml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml` using:
   ```bash
   conda env update -f environment.lock.yml --prune
   ```
## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── particle_packing    <- Actual Python package where the main functionality goes.
│   └── crabnet_hyperparameter <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.3.1 and the [dsproject extension] 0.7.2.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
