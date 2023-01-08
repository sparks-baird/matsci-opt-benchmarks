

from zenodo_client import Creator, Metadata, ensure_zenodo
from my_secrets import ZENODO_API_KEY, ZENODO_SANDBOX_API_KEY



# Define the metadata that will be used on initial upload
data = Metadata(
    title="Materials Science Optimization Benchmark Dataset for Multi-fidelity Hard-sphere Packing Simulations",
    upload_type="dataset",
    description="Benchmarks are an essential driver of progress in scientific disciplines. Ideal benchmarks mimic real-world tasks as closely as possible, where insufficient difficulty or applicability can stunt growth in the field. Benchmarks should also have sufficiently low computational overhead to promote accessibility and repeatability. The goal is then to win a “Turing test” of sorts by creating a surrogate model that is indistinguishable from the ground truth observation (at least within the dataset bounds that were explored), necessitating a large amount of data. In the fields of materials science and chemistry, industry-relevant optimization tasks are often hierarchical, noisy, multi-fidelity, multi-objective, high-dimensional, and non-linearly correlated while exhibiting mixed numerical and categorical variables subject to linear and non-linear constraints. To complicate matters, unexpected, failed simulation or experimental regions may be present in the search space. In this study, 438371 random hard-sphere packing simulations representing 279 CPU days worth of computational overhead were performed across nine input parameters with linear constraints and two discrete fidelities each with continuous fidelity parameters and results were logged to a free-tier shared MongoDB Atlas database. Two core tabular datasets resulted from this study: 1. a failure probability dataset containing unique input parameter sets and the estimated probabilities that the simulation will fail at each of the two steps, and 2. a regression dataset mapping input parameter sets (including repeats) to particle packing fractions and computational runtimes for each of the two steps. These two datasets can be used to create a surrogate model as close as possible to running the actual simulations by incorporating simulation failure and heteroskedastic noise. For the regression dataset, percentile ranks were computed within each of the groups of identical parameter sets to enable capturing heteroskedastic noise. This is in contrast with a more traditional approach that imposes a-priori assumptions such as Gaussian noise e.g., by providing a mean and standard deviation. A similar approach can be applied to other benchmark datasets to bridge the gap between optimization benchmarks with low computational overhead and realistically complex, real-world optimization scenarios.",
    creators=[
        Creator(
            name="Baird, Sterling G.",
            affiliation="University of Utah",
            orcid="0000-0002-4491-6876",
        ),
    ],
)

sandbox = False
access_token = ZENODO_SANDBOX_API_KEY if sandbox else ZENODO_API_KEY
res = ensure_zenodo(
    "matsciopt-particle-packing-benchmark-dataset",
    data=data,
    paths=[
        "data/processed/particle_packing/sobol_probability_filter.csv",
        "data/processed/particle_packing/sobol_regression.csv",
    ],
    sandbox=sandbox,  # remove this when you're ready to upload to real Zenodo
    access_token=access_token,
)
from pprint import pprint

pprint(res.json())
