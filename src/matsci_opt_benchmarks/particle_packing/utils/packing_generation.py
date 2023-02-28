import os
import shutil
from math import ceil, pi
from os import path
from pathlib import Path
from random import randint
from subprocess import run
from time import time
from uuid import uuid4

import cloudpickle as pickle
import numpy as np
import pandas as pd
from scipy.stats import lognorm


def get_diameters(means, stds, comps, num_particles=100, seed=None):
    samples = []
    for mu, sigma, comp in zip(means, stds, comps):
        samples.append(
            lognorm.rvs(
                sigma, scale=mu, size=ceil(comp * num_particles), random_state=seed
            )
        )

    samples = np.concatenate(samples)[:num_particles]

    if len(samples) != num_particles:
        raise ValueError(
            f"Number of samples ({len(samples)}) does not match requested number of particles {num_particles}. Ensure `sum(comps)==1` (sum({comps})=={sum(comps)})"  # noqa: E501
        )

    return samples


def write_diameters(X, data_dir=".", uid=str(uuid4())):
    save_dir = path.join(data_dir, uid)
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    fpath = path.join(save_dir, "diameters.txt")
    np.savetxt(fpath, X)


def get_box_length(X, safety_factor=2.0):
    particles_vol = sum(4 / 3 * pi * (X / 2) ** 3)
    return (particles_vol * safety_factor) ** (1 / 3)


def run_simulation(flag, util_dir=".", data_dir=".", uid=str(uuid4())):
    exe_name = "PackingGeneration.exe"
    simpath = path.join(util_dir, exe_name)
    new_dir = path.join(data_dir, uid)

    run(
        [f"{path.join(os.getcwd(), simpath)}", flag],
        capture_output=True,
        text=True,
        cwd=new_dir,
        # stdout=PIPE,
        # stderr=STDOUT,
    )


def particle_packing_simulation(
    means,
    stds,
    comps,
    num_particles=100,
    contraction_rate=1e-3,
    seed=None,
    data_dir=".",
    util_dir=".",
    uid=None,
    safety_factor=2.0,
    cleanup=True,
):
    if seed is None:
        seed = randint(0, 100000)
    if uid is None:
        uid = str(uuid4())
        print(uid)

    save_dir = path.join(data_dir, uid)
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    generation_conf_fpath = path.join(save_dir, "generation.conf")
    packing_nfo_fpath = path.join(save_dir, "packing.nfo")
    packing_xyzd_fpath = path.join(save_dir, "packing.xyzd")

    # X = np.repeat(1.0, num_particles)
    X = get_diameters(means, stds, comps, num_particles=num_particles, seed=seed)

    write_diameters(X, data_dir=data_dir, uid=uid)
    box_length = get_box_length(X, safety_factor=safety_factor)

    # remove existing data files, if any
    names = [
        "packing.xyzd",
        "packing_init.xyzd",
        "packing_prev.xyzd",
        "contraction_energies.txt",
        "packing.nfo",
    ]
    [
        os.remove(path.join(data_dir, uid, name)) if path.exists(name) else None
        for name in names
    ]

    with open(generation_conf_fpath, "w") as f:
        lines = [
            f"Particles count: {num_particles}",
            f"Packing size: {box_length} {box_length} {box_length}",
            "Generation start: 1",
            f"Seed: {seed}",
            "Steps to write: 1000",
            "Boundaries mode: 1",
            f"Contraction rate: {contraction_rate}",
        ]
        f.writelines(lines)

    results = {}

    t0 = time()
    try:
        run_simulation("-fba", util_dir=util_dir, data_dir=data_dir, uid=uid)
        results["fba"] = read_packing_fraction(
            data_dir, uid, packing_xyzd_fpath, box_length
        )
    except Exception as e:
        print(e)
        results["fba"] = None

    t1 = time()
    results["fba_time_s"] = t1 - t0

    with open(generation_conf_fpath, "w") as f:
        lines = [
            f"Particles count: {num_particles}",
            f"Packing size: {box_length} {box_length} {box_length}",
            "Generation start: 0",
            f"Seed: {seed}",
            "Steps to write: 1000",
            "Boundaries mode: 1",
            f"Contraction rate: {contraction_rate}",
        ]
        f.writelines(lines)

    try:
        os.remove(packing_nfo_fpath)
    except Exception as e:
        print(e)
    try:
        run_simulation("-ls", util_dir=util_dir, data_dir=data_dir, uid=uid)
        results["ls"] = read_packing_fraction(
            data_dir, uid, packing_xyzd_fpath, box_length
        )
    except Exception as e:
        results["ls"] = None
        print(e)

    t2 = time()
    results["ls_time_s"] = t2 - t1

    # try:
    #     os.remove(packing_nfo_fpath)
    # except Exception as e:
    #     print(e)
    # try:
    #     run_simulation("-lsgd", util_dir=util_dir, data_dir=data_dir, uid=uid)
    #     results["lsgd"] = read_packing_fraction(
    #         data_dir, uid, packing_xyzd_fpath, box_length, final=True
    #     )
    # except Exception as e:
    #     results["lsgd"] = None
    #     print(e)
    #
    # t3 = time()
    # results["lsgd_time_s"] = t3 - t2

    """https://github.com/VasiliBaranov/packing-generation/issues/30#issue-1103925864"""

    if cleanup:
        shutil.rmtree(save_dir)

    return results


def read_packing_fraction(data_dir, uid, packing_xyzd_fpath, box_length, final=False):
    packing = np.fromfile(packing_xyzd_fpath).reshape(-1, 4)
    with open(path.join(data_dir, uid, "packing.nfo"), "r+") as nfo:
        lines = nfo.readlines()
        Theoretical_Porosity = float(lines[2].split()[2])
        Final_Porosity = float(lines[3].split()[2])
        # print(Theoretical_Porosity, Final_Porosity)

        scaling_factor = ((1 - Final_Porosity) / (1 - Theoretical_Porosity)) ** (1 / 3)

        real_diameters = packing[:, 3] * scaling_factor
        actual_density = (
            sum((4 / 3) * pi * (np.array(real_diameters) / 2) ** 3) / box_length**3
        )
        if final:
            packing[:, 3] = real_diameters
            # updating the packing: this line will modify diameters in the packing.xyzd
            packing.tofile(packing_xyzd_fpath)

            # update packing.nfo and set TheoreticalPorosity to FinalPorosity to avoid
            # scaling the packing once again the next time running this script.
            lines[3] = lines[3].replace(str(Final_Porosity), str(Theoretical_Porosity))
            nfo.seek(0)
            nfo.writelines(lines)

    return actual_density


def evaluate(parameters):
    # mu3 = 3.0
    # print("current working directory: ", os.getcwd())
    # means = [parameters[name] * mu3 for name in ["mu1_div_mu3", "mu2_div_mu3"]]
    # means.append(mu3)
    means = [parameters[name] for name in ["mu1", "mu2", "mu3"]]
    stds = [parameters[name] for name in ["std1", "std2", "std3"]]
    comps = [parameters[name] for name in ["comp1", "comp2", "comp3"]]
    # comps.append(1 - sum(comps))
    num_particles = parameters["num_particles"]
    safety_factor = parameters.get("safety_factor", 2.0)
    util_dir = parameters.get("util_dir", ".")
    data_dir = parameters.get("data_dir", ".")

    try:
        results = particle_packing_simulation(
            means,
            stds,
            comps,
            num_particles=num_particles,
            util_dir=util_dir,
            data_dir=data_dir,
            safety_factor=safety_factor,
        )
    except Exception as e:
        print(e)
        results = {"error": str(e)}
    results = {**parameters, **results}

    return results


def evaluate_batch(parameter_sets):
    return [evaluate(parameters) for parameters in parameter_sets]


def collect_results(job_pkl_path, slurm_savepath):
    with open(job_pkl_path, "rb") as f:
        jobs = pickle.load(f)

    results = [job.result() for job in jobs]
    pd.DataFrame(results).to_csv(slurm_savepath, index=False)


# %% Code Graveyard

# newpath = path.join(new_dir, exe_name)
# shutil.copyfile(simpath, newpath)
# f = Path(newpath)
# f.chmod(f.stat().st_mode | stat.S_IEXEC)

# cwd = os.getcwd()
# os.chdir(new_dir)

# print(result.stdout)
# print(result.stderr)
# os.chdir(cwd)

# packing_fraction = read_packing_fraction(
#     data_dir, uid, packing_xyzd_fpath, box_length, final=True
# )
