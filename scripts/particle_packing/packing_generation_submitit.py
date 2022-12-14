# %% imports
from submitit import AutoExecutor
from os import path

import cloudpickle as pickle

from ax.modelbridge.factory import get_sobol
from ax.service.ax_client import AxClient

# https://www.chpc.utah.edu/documentation/software/modules-advanced.php
# ERROR: Could not install packages due to an OSError: [Errno 122] Disk quota exceeded: '/uufs/chpc.utah.edu/common/home/u1326059/software/pkg/miniconda3/envs/particle-packing/lib/python3.9/site-packages/joblib-1.2.0.dist-info/INSTALLER2p04xuw3.tmp'

from matsci_opt_benchmarks.particle_packing.utils.packing_generation import (
    evaluate,
    collect_results,
)
from matsci_opt_benchmarks.particle_packing.utils.data import get_parameters

dummy = True
slurm_savepath = path.join("data", "processed", "packing-generation-results.csv")
job_pkl_path = path.join("data", "interim", "packing-generation-jobs.pkl")
num_repeats = 10

(
    subfrac_names,
    parameters,
    generous_parameters,
    mean_names_out,
    std_names_out,
    orig_mean_names,
    orig_std_names,
) = get_parameters(remove_composition_degeneracy=True, remove_scaling_degeneracy=True)

ax_client = AxClient()
ax_client.create_experiment(
    name="boppf_sobol",
    parameters=parameters,
    objective_name="packing_fraction",
    minimize=False,
    parameter_constraints=["std1 <= std2", "comp1 + comp2 <= 1.0"],
)
search_space = ax_client.experiment.search_space
m = get_sobol(search_space, fallback_to_sample_polytope=True)
gr = m.gen(n=2**14)  # 2**14 == 16384
param_df = gr.param_df.copy()
param_df["num_particles"] = 100
parameter_sets = gr.param_df.to_dict(orient="records")
parameter_sets = parameter_sets * num_repeats
parameter_sets.shuffle()

if dummy:
    parameter_sets = parameter_sets[:10]
    batch_size = 5
else:
    batch_size = 100


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


parameter_batch_sets = list(chunks(parameter_sets, batch_size))

# %% submission
log_folder = "data/interim/particle_packing/%j"
walltime = 5 * batch_size
partition, account = ["notchpeak", "notchpeak"]
executor = AutoExecutor(folder=log_folder)
executor.update_parameters(
    timeout_min=walltime,
    slurm_partition=partition,
    slurm_cpus_per_task=1,
    slurm_additional_parameters={"account": account},
)

# sbatch array
# jobs = executor.map_array(evaluate_batch, parameter_batch_sets)
jobs = executor.map_array(evaluate, parameter_sets)
job_ids = [job.job_id for job in jobs]
# https://www.hpc2n.umu.se/documentation/batchsystem/job-dependencies
job_ids_str = ":".join(job_ids)  # e.g. "3937257_0:3937257_1:..."

with open(job_pkl_path, "wb") as f:
    pickle.dump(jobs, f)


collect_folder = "data/processed/%j"
walltime = 10
collector = AutoExecutor(folder=collect_folder)
collector.update_parameters(
    timeout_min=walltime,
    slurm_partition=partition,
    slurm_additional_parameters={
        "account": account,
        "dependency": f"afterok:{job_ids_str}",
    },
)
collector_job = collector.submit(
    collect_results, job_pkl_path, slurm_savepath
)  # sbatch array

print(
    f"Waiting for submission jobs ({job_ids_str}) to complete before running collector job ({collector_job.job_id}). Pickled results file saved to {slurm_savepath} after all jobs have run."
)
