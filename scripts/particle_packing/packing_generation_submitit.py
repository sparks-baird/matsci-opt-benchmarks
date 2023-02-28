# %% imports
import json
from datetime import datetime
from os import path
from random import shuffle
from time import time
from uuid import uuid4

import pandas as pd
import pymongo
import requests
import torch
from ax.modelbridge.factory import get_sobol
from ax.service.ax_client import AxClient
from my_secrets import MONGODB_API_KEY, MONGODB_PASSWORD, MONGODB_USERNAME
from submitit import AutoExecutor
from tqdm import tqdm

from matsci_opt_benchmarks.particle_packing.utils.data import get_parameters
from matsci_opt_benchmarks.particle_packing.utils.packing_generation import evaluate

# https://www.chpc.utah.edu/documentation/software/modules-advanced.php
# ERROR: Could not install packages due to an OSError: [Errno 122] Disk quota exceeded:
# '/uufs/chpc.utah.edu/common/home/u1326059/software/pkg/miniconda3/envs/
# particle-packing/lib/python3.9/site-packages/joblib-1.2.0.dist-info/
# INSTALLER2p04xuw3.tmp'


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
) = get_parameters(remove_composition_degeneracy=False, remove_scaling_degeneracy=False)

parameters.append({"name": "num_particles", "type": "range", "bounds": [100, 1000]})
parameters.append({"name": "safety_factor", "type": "range", "bounds": [1.0, 2.5]})
ax_client = AxClient()
ax_client.create_experiment(
    name="boppf_sobol",
    parameters=parameters,
    objective_name="packing_fraction",
    minimize=False,
    parameter_constraints=["std1 <= std2", "std2 <= std3"],
)
search_space = ax_client.experiment.search_space
# https://github.com/facebook/Ax/issues/740
# https://github.com/facebook/Ax/issues/1439
fallback = False
m = get_sobol(search_space, fallback_to_sample_polytope=fallback, seed=SEED)
# https://github.com/facebook/Ax/issues/1439
torch.manual_seed(SEED)
# increase max_rs_draws if you get an error about too many random draws
# as a last resort, switch fallback to True (above)
max_rs_draws = 1000000
gr = m.gen(n=num_samples, model_gen_options={"max_rs_draws": max_rs_draws})
# gr.param_df not working https://github.com/facebook/Ax/issues/1437
# param_df = gr.param_df.copy()
param_df = pd.DataFrame([arm.parameters for arm in gr.arms])


app_name = "data-oeodi"
url = f"https://us-east-1.aws.data.mongodb-api.com/app/{app_name}/endpoint/data/v1/action/insertOne"  # noqa: E501
# noqa: E501
collection_name = "sobol"
database_name = "particle-packing"
dataSource = "Cluster0"
cluster_uri = "cluster0.n03mvdg"

# to find this string, click connect to your MongoDB cluster on the website
# also needed to go to "Network Access", click "Add IP address", click "Allow access
# from anywhere", and add
client = pymongo.MongoClient(
    f"mongodb+srv://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{cluster_uri}.mongodb.net/?retryWrites=true&w=majority"  # noqa: E501
)
db = client[database_name]
collection = db[collection_name]

posts = collection.find({})
results = [post for post in tqdm(posts)]

parameter_names = [
    "mu1",
    "mu2",
    "mu3",
    "std1",
    "std2",
    "std3",
    "comp1",
    "comp2",
    "comp3",
    "num_particles",
    "safety_factor",
]

if len(results) > 0:
    mongo_df = pd.DataFrame(results)
    mongo_param_df: pd.DataFrame = mongo_df[parameter_names]

    # remove the entries that are already in the database, including repeats
    # the repeats are necessary for the variance calculation
    # this is sort of a setdiff

    mongo_param_df = mongo_param_df.groupby(
        mongo_param_df.columns.tolist(), as_index=False
    ).size()  # type: ignore
    mongo_param_df["group_id"] = (
        mongo_param_df[parameter_names]
        .round(6)
        .apply(lambda row: "_".join(row.values.astype(str)), axis=1)
    )
    param_df["group_id"] = (
        param_df[parameter_names]
        .round(6)
        .apply(lambda row: "_".join(row.values.astype(str)), axis=1)
    )

    # pd.concat((param_df["group_id"], mongo_param_df["group_id"])).drop_duplicates()

    param_df = param_df[~param_df["group_id"].isin(mongo_param_df["group_id"])]
    param_df = param_df[parameter_names]

    mongo_param_df = mongo_param_df[(num_repeats - mongo_param_df["size"]) > 0]

    # repeat the rows of mongo_param_df based on the size column
    # iterate through rows of mongo_param_df
    sub_dfs = []
    for index, row in mongo_param_df.iterrows():
        sub_dfs.append(
            pd.concat([row[parameter_names]] * row["size"], axis=1, ignore_index=True).T
        )

    if len(sub_dfs) > 0:
        mongo_param_df = pd.concat(sub_dfs, axis=0, ignore_index=True)[parameter_names]
    else:
        mongo_param_df = pd.DataFrame(columns=parameter_names)

# repeat the rows of param_df num_repeats times
param_df = pd.concat([param_df] * num_repeats, ignore_index=True)

if len(results) > 0:
    param_df = pd.concat([param_df, mongo_param_df])  # type: ignore

param_df["util_dir"] = path.join(
    "src", "matsci_opt_benchmarks", "particle_packing", "utils"
)
param_df["data_dir"] = path.join("data", "interim", "particle_packing")
parameter_sets = param_df.to_dict(orient="records")
# parameter_sets = parameter_sets * num_repeats
shuffle(parameter_sets)

if dummy:
    # parameter_sets = parameter_sets[:10]
    batch_size = 5
else:
    batch_size = 1400


def mongodb_evaluate(parameter_set, verbose=False):
    """Evaluate a parameter set and save the results to MongoDB."""
    t0 = time()
    results = evaluate(parameter_set)
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
            "collection": collection_name,
            "database": database_name,
            "dataSource": dataSource,
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
# account = "owner-guest"
# partition = "kingspeak-guest"
account = "sparks"
partition = "notchpeak-shared-freecycle"  # to allow for node sharing
executor = AutoExecutor(folder=log_folder)
executor.update_parameters(
    timeout_min=walltime_min,
    slurm_partition=partition,
    slurm_additional_parameters={"ntasks": 1, "account": account},
)

# mongodb_evaluate(parameter_sets[0], verbose=True)

# sbatch array
jobs = executor.map_array(mongodb_evaluate_batch, parameter_batch_sets)
# jobs = executor.map_array(mongodb_evaluate, parameter_sets)
print("Submitted jobs")

results = [job.result() for job in jobs]

1 + 1

# %% Code Graveyard
# import pymongo
# from urllib.parse import quote_plus
# password needs to be URL encoded
# client = pymongo.MongoClient(
#     f"mongodb+srv://{USERNAME}:{quote_plus(PASSWORD)}@matsci-opt-benchmarks.ehu7qrh.mongodb.net/?retryWrites=true&w=majority"# noqa: E501
# )
# collection = client["particle-packing"]["sobol"]
# collection.insert_one(result)

# import cloudpickle as pickle

# param_df = param_df[~param_df.isin(mongo_param_df[parameter_names]).all(1)]
# param_df = param_df[~param_df.isin(mongo_param_df).all(1)]

# setdiff between two dataframes
# https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-

# param_df["num_particles"] = 1000


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

# print( f"Waiting for submission jobs ({job_ids_str}) to complete before running
#     collector job ({collector_job.job_id}). Pickled results file saved to
# {slurm_savepath} after all jobs have run." )
