# %% imports
from time import time
from submitit import AutoExecutor
from os import path
from datetime import datetime
import numpy as np

from my_secrets import MONGODB_API_KEY

from ax.modelbridge.factory import get_sobol
from ax.service.ax_client import AxClient
import json
import requests

from random import shuffle
from uuid import uuid4

# https://www.chpc.utah.edu/documentation/software/modules-advanced.php
# ERROR: Could not install packages due to an OSError: [Errno 122] Disk quota exceeded: '/uufs/chpc.utah.edu/common/home/u1326059/software/pkg/miniconda3/envs/particle-packing/lib/python3.9/site-packages/joblib-1.2.0.dist-info/INSTALLER2p04xuw3.tmp'

from matsci_opt_benchmarks.particle_packing.utils.packing_generation import (
    evaluate,
    evaluate_batch,
    # collect_results,
)
from matsci_opt_benchmarks.particle_packing.utils.data import get_parameters

dummy = False
SEED = 10
if dummy:
    num_samples = 2**3  # 2**3 == 8
    num_repeats = 2
else:
    num_samples = 2**16  # 2**16 == 65536
    num_repeats = 15

slurm_savepath = path.join("data", "processed", "packing-generation-results.csv")
job_pkl_path = path.join("data", "interim", "packing-generation-jobs.pkl")

session_id = str(uuid4())

(
    subfrac_names,
    parameters,
    generous_parameters,
    mean_names_out,
    std_names_out,
    orig_mean_names,
    orig_std_names,
) = get_parameters(remove_composition_degeneracy=True, remove_scaling_degeneracy=True)

parameters.append({"name": "num_particles", "type": "range", "bounds": [100, 1000]})
parameters.append({"name": "safety_factor", "type": "range", "bounds": [1.0, 2.5]})
ax_client = AxClient()
ax_client.create_experiment(
    name="boppf_sobol",
    parameters=parameters,
    objective_name="packing_fraction",
    minimize=False,
    parameter_constraints=["std1 <= std2", "comp1 + comp2 <= 1.0"],
)
search_space = ax_client.experiment.search_space
m = get_sobol(search_space, fallback_to_sample_polytope=True, seed=SEED)
gr = m.gen(n=num_samples)
param_df = gr.param_df.copy()
# param_df["num_particles"] = 1000
param_df["util_dir"] = path.join(
    "src", "matsci_opt_benchmarks", "particle_packing", "utils"
)
param_df["data_dir"] = path.join("data", "interim", "particle_packing")
parameter_sets = param_df.to_dict(orient="records")
parameter_sets = parameter_sets * num_repeats
shuffle(parameter_sets)

if dummy:
    parameter_sets = parameter_sets[:10]
    batch_size = 5
else:
    batch_size = 700

url = "https://data.mongodb-api.com/app/data-plyju/endpoint/data/v1/action/insertOne"


def mongodb_evaluate(parameter_set, verbose=False):
    """Evaluate a parameter set and save the results to MongoDB."""
    t0 = time()
    results = evaluate(parameter_set)
    runtime = time() - t0
    print(results)
    utc = datetime.utcnow()
    results = {
        **results,
        "session_id": session_id,
        "timestamp": utc.timestamp(),
        "date": str(utc),
        "runtime": time() - t0,
        "seed": SEED,
        "num_samples": num_samples,
        "num_repeats": num_repeats,
    }
    results.pop("data_dir")
    results.pop("util_dir")

    payload = json.dumps(
        {
            "collection": "sobol",
            "database": "particle-packing",
            "dataSource": "matsci-opt-benchmarks",
            "document": results,
        }
    )
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Request-Headers": "*",
        "api-key": MONGODB_API_KEY,
    }
    if verbose:
        print(f"Submitting {payload} to {url}...")

    response = requests.request("POST", url, headers=headers, data=payload)

    if verbose:
        print(response.text)
    return results


def mongodb_evaluate_batch(parameter_sets, verbose=False):
    return [mongodb_evaluate(p, verbose=verbose) for p in parameter_sets]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


parameter_batch_sets = list(chunks(parameter_sets, batch_size))

# %% submission
log_folder = "data/interim/particle_packing/%j"
walltime_min = int(round(((120 / 60) * batch_size) + 3))
# use `myallocation` command to see available account/partition combos
# account = "sparks"
# partition = "kingspeak"
account = "owner-guest"
partition = "kingspeak-guest"
executor = AutoExecutor(folder=log_folder)
executor.update_parameters(
    timeout_min=walltime_min,
    slurm_nodes=None,
    slurm_partition=partition,
    # slurm_cpus_per_task=1,
    slurm_additional_parameters={"ntasks": 1, "account": account},
)

# sbatch array
jobs = executor.map_array(mongodb_evaluate_batch, parameter_batch_sets)
# jobs = executor.map_array(mongodb_evaluate, parameter_sets)
print("Submitted jobs")
# job_ids = [job.job_id for job in jobs]
# # https://www.hpc2n.umu.se/documentation/batchsystem/job-dependencies
# job_ids_str = ":".join(job_ids)  # e.g. "3937257_0:3937257_1:..."

# with open(job_pkl_path, "wb") as f:
#     pickle.dump(jobs, f)


# collect_folder = "data/processed/%j"
# walltime = 10
# collector = AutoExecutor(folder=collect_folder)
# collector.update_parameters(
#     timeout_min=walltime,
#     slurm_partition=partition,
#     slurm_additional_parameters={
#         "account": account,
#         "dependency": f"afterok:{job_ids_str}",
#     },
# )
# collector_job = collector.submit(
#     collect_results, job_pkl_path, slurm_savepath
# )  # sbatch array

# print(
#     f"Waiting for submission jobs ({job_ids_str}) to complete before running collector job ({collector_job.job_id}). Pickled results file saved to {slurm_savepath} after all jobs have run."
# )

results = [job.result() for job in jobs]

1 + 1

# %% Code Graveyard
# import pymongo
# from urllib.parse import quote_plus
# password needs to be URL encoded
# client = pymongo.MongoClient(
#     f"mongodb+srv://{USERNAME}:{quote_plus(PASSWORD)}@matsci-opt-benchmarks.ehu7qrh.mongodb.net/?retryWrites=true&w=majority"
# )
# collection = client["particle-packing"]["sobol"]
# collection.insert_one(result)

# import cloudpickle as pickle
