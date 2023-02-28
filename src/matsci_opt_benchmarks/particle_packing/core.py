"""Perform Bayesian optimization on the particle packing simulation parameters."""
from os import path
from pathlib import Path

import ray
import torch
from psutil import cpu_count

from matsci_opt_benchmarks.particle_packing.utils.ax import optimize_ppf


class BOPPF:
    def __init__(
        self,
        dummy=False,
        particles=int(2.5e4),
        n_sobol=None,
        n_bayes=100,
        save_dir="results",
        savename="experiment.json",
        max_parallel="cpu_count",
        include_logical_cores=False,  # hyperthreading or not
        debug=False,
        torch_device=torch.device("cuda"),
        use_saas=False,
        use_random=False,
        seed=10,
        data_augmentation=False,
        remove_composition_degeneracy=True,
        remove_scaling_degeneracy=False,
        use_order_constraint=False,
        ray_verbosity=3,
    ) -> None:
        self.particles = particles
        self.n_sobol = n_sobol
        self.n_bayes = n_bayes
        self.dummy = dummy

        if isinstance(max_parallel, str) and max_parallel == "cpu_count":
            max_parallel = max(1, cpu_count(logical=include_logical_cores))

        print(f"maximum number of parallel jobs: {max_parallel}")

        self.max_parallel = max_parallel
        self.torch_device = torch_device
        self.use_saas = use_saas
        self.use_random = use_random

        if dummy:
            save_dir = path.join(save_dir, "dummy")

        if use_saas:
            save_dir = path.join(save_dir, "saas")

        if use_random:
            save_dir = path.join(save_dir, "random")

        save_dir = path.join(
            save_dir,
            f"particles={particles}",
            f"max_parallel={max_parallel}",
            f"n_sobol={n_sobol},n_bayes={n_bayes},seed={seed}",
            f"augment={data_augmentation},drop_last={remove_composition_degeneracy},drop_scaling={remove_scaling_degeneracy},order={use_order_constraint}",  # noqa: E501
        )

        Path(save_dir).mkdir(exist_ok=True, parents=True)
        self.save_dir = save_dir
        self.savename = savename

        ray.shutdown()
        # https://github.com/ray-project/ray/issues/13511#issuecomment-774322180
        if debug:
            ray.init(
                local_mode=True,
                num_cpus=max_parallel,
                log_to_driver=False,  # maybe I should set this to True again
                include_dashboard=False,  # https://github.com/ray-project/ray/issues/9114#issuecomment-707325418 # noqa: E501
            )
        else:
            ray.init(
                num_cpus=max_parallel, log_to_driver=False, include_dashboard=False
            )

        self.seed = seed

        self.data_augmentation = data_augmentation
        self.remove_composition_degeneracy = remove_composition_degeneracy
        self.remove_scaling_degeneracy = remove_scaling_degeneracy
        self.use_order_constraint = use_order_constraint
        self.ray_verbosity = ray_verbosity

    def optimize(self, X_train, y_train, return_ax_client=False):
        # %% optimization
        self.ax_client, best_parameters, mean, covariance = optimize_ppf(
            X_train,
            y_train,
            particles=self.particles,
            n_sobol=self.n_sobol,
            n_bayes=self.n_bayes,
            save_dir=self.save_dir,
            max_parallel=self.max_parallel,
            torch_device=self.torch_device,
            use_saas=self.use_saas,
            use_random=self.use_random,
            seed=self.seed,
            data_augmentation=self.data_augmentation,
            remove_composition_degeneracy=self.remove_composition_degeneracy,
            remove_scaling_degeneracy=self.remove_scaling_degeneracy,
            use_order_constraint=self.use_order_constraint,
            ray_verbosity=self.ray_verbosity,
        )

        if return_ax_client:
            return best_parameters, mean, covariance, self.ax_client
        else:
            return best_parameters, mean, covariance


# %% Code Graveyard
# runtime_env = {"working_dir": getcwd(), "py_modules": [join("boppf", "utils")]}
# ray.init(local_mode=True, runtime_env=runtime_env)
