# v2 CrabNet rerun on BYU Research Computing

The v1 Sobol dataset ([Zenodo 10.5281/zenodo.7694268](https://doi.org/10.5281/zenodo.7694268)) trained one CrabNet network on the five `matbench_expt_gap` folds in turn, so folds 1 to 4 were scored on compositions the network had already seen. `submitit_evaluate` now builds a new network for every fold and returns each fold's scores. The files here rerun v1 hyperparameter sets with it as a Slurm job array on BYU's supercomputer.

| File | What it does |
|---|---|
| `setup_env.sh` | One-time setup on a login node: conda environment, v1 CSV, Matbench dataset |
| `rerun.py manifest` | Picks the runs for a design and splits them into array tasks of about 4 h of v1 (RTX 2080 Ti) runtime |
| `rerun.sbatch` | One array task: runs its share of the manifest, appending each result to `results/task_<id>.jsonl`. Preemptable (`--qos=standby`) by default |
| `rerun.py collect` | Merges the results, compares them with v1 and lists unfinished tasks |
| `orc.sh` | Login-node entry point: `setup`, `manifest`, `smoke`, `submit` (also resubmits preempted tasks) and `status` |

Designs: `smoke` (2 runs), `decision` (1,000 random sets plus the 200 best v1 sets, 1,190 runs, about 3 GPU-days at 2080 Ti speed), `one-per-set` (41,543 runs, 93 GPU-days) and `full` (173,203 runs, one per v1 run, 387 GPU-days).

## Before you start

- **Account.** Request one at <https://rc.byu.edu/account/create/>. Students need a sponsor who is CFS-track faculty. Approval takes 2 to 3 business days, and accounts are renewed every year ([Getting Started](https://rc.byu.edu/wiki/?id=Getting+Started)).
- **Two-factor login.** Enroll at <https://rc.byu.edu/account/authenticate/enroll> (separate from BYU's Duo). Every SSH login asks for your ORC password and a verification code, and SSH keys are not allowed ([Logging In](https://rc.byu.edu/wiki/?id=Logging+In), [Two-Factor Authentication](https://rc.byu.edu/wiki/?id=Two-Factor+Authentication)).
- **Optional: authenticate once per session** with SSH multiplexing ([SSH Multiplexing](https://rc.byu.edu/wiki/index.php?page=SSH+Multiplexing)):

  ```
  Host orc
          User <your username>
          HostName ssh.rc.byu.edu
          ControlMaster auto
          ControlPath ~/.ssh/master-%r@%h:%p.socket
          ControlPersist yes
          ServerAliveInterval 300
  ```

## 1. Set up (login node, once)

```bash
ssh <username>@ssh.rc.byu.edu
git clone https://github.com/sparks-baird/matsci-opt-benchmarks.git ~/matsci-opt-benchmarks
git -C ~/matsci-opt-benchmarks checkout claude/byu-rc-rerun   # until this is merged
bash ~/matsci-opt-benchmarks/scripts/crabnet_hyperparameter/byu_rc/setup_env.sh
```

Why it is done this way:

- **Downloads happen here.** Login nodes have internet access and compute nodes do not; the docs ask you to download on login nodes rather than through the compute-node proxy ([Internet Access from Jobs](https://rc.byu.edu/wiki/index.php?page=Internet+Access+from+Jobs)). The script fetches the v1 CSV and caches the Matbench dataset in `$RERUN_DIR/matminer_data` (`MATMINER_DATA`), which the jobs read.
- **Build on the login node.** GPU nodes should not be used for builds ([Internet Access from Jobs](https://rc.byu.edu/wiki/index.php?page=Internet+Access+from+Jobs)). The setup takes minutes, well inside the one hour of CPU time a login-node process may use ([Slurm](https://rc.byu.edu/wiki/?id=Slurm)).
- **Conda through `miniforge3`**, as recommended ([Conda Environments](https://rc.byu.edu/wiki/?id=Conda+Environments)). The docs discourage pip ([Python](https://rc.byu.edu/wiki/?page=Python)), but v1's exact builds (`torch==1.13.1+cu117`, `crabnet==2.0.8`, `matbench==0.6`) are pip wheels, so pip installs them into the conda environment. The torch wheel ships its own CUDA runtime, so no `cuda` module is needed.
- **Everything lives in your home directory**: the environment in `~/.conda/envs/crabnet-v2` and results in `~/crabnet_rerun` (override with `ENV_NAME` and `RERUN_DIR`). Home has 2 TiB and 2 million files and is backed up. `~/nobackup/autodelete` deletes files unused for 12 weeks, and programs should not be installed there ([Storage](https://rc.byu.edu/wiki/?id=Storage)). The full design writes about 250 MB of results. Check usage with `orcquota`.

## 2. Pick a GPU type

`torch==1.13.1+cu117` runs on GPUs up to compute capability 8.6, plus 8.9 through CUDA's binary compatibility. It does not run on Hopper or Blackwell GPUs ([pytorch#90761](https://github.com/pytorch/pytorch/issues/90761)). Always request a type: a plain `--gpus=1` can land on an H200. Counts are from [Compute Resources](https://rc.byu.edu/documentation/resources) and request syntax from [Getting Started With GPUs](https://rc.byu.edu/wiki/?id=Getting+Started+With+GPUs) and [Slurm](https://rc.byu.edu/wiki/?id=Slurm).

| GPU | Access | GPUs | Runs v1's torch | Request |
|---|---|---|---|---|
| L40S 48 GB | general | 16 (4 nodes x 4) | yes (8.9) | `--gpus=l40s:1` (the default in `rerun.sbatch`) |
| P100 16 GB | general | 160 (40 x 4) | yes (6.0) | `--gpus=1 --constraint=pascal` (no type name is documented) |
| A100 80 GB | preemption only | 120 | yes (8.0) | `--qos=standby` plus the type name from `sinfo -h -o "%G" \| sort -u` |
| V100 32 GB | preemption only | 1 | yes (7.0) | `--qos=standby` |
| H200 141 GB | general (32), preemption only (8) | 40 | no (9.0) | |
| H100, B200 | preemption only | 16, 8 | no | |
| GH200 | general | 2 | no (ARM CPU) | |

If runtime is measured again in v2, keep one GPU type for the whole campaign. Every result records the GPU name (`gpu`), so runs on different types can be told apart.

## 3. Smoke test

```bash
cd ~/crabnet_rerun
module load miniforge3 && eval "$(conda shell.bash hook)" && conda activate crabnet-v2
python ~/matsci-opt-benchmarks/scripts/crabnet_hyperparameter/byu_rc/rerun.py manifest --design smoke
sbatch --array=0 --qos=test --time=00:30:00 --export=ALL,DESIGN=smoke \
    ~/matsci-opt-benchmarks/scripts/crabnet_hyperparameter/byu_rc/rerun.sbatch
```

The `test` QOS allows one hour and 5 jobs at a time ([Slurm](https://rc.byu.edu/wiki/?id=Slurm), [Why won't my job submit?](https://rc.byu.edu/wiki/index.php?page=Why+won%27t+my+job+submit%3F)); drop `--qos=test` if it refuses GPU jobs. Submit from `~/crabnet_rerun` so the log lands in `~/crabnet_rerun/logs`. When it finishes, `rerun.py collect --design smoke` should report 2 of 2 runs done, and the model sizes should match `v1_model_size` in the manifest.

Or, in one step: `bash ~/matsci-opt-benchmarks/scripts/crabnet_hyperparameter/byu_rc/orc.sh smoke`.

## 4. Decision subset

```bash
ORC=~/matsci-opt-benchmarks/scripts/crabnet_hyperparameter/byu_rc/orc.sh
bash $ORC manifest decision      # 1,190 runs in 20 tasks
bash $ORC submit decision        # standby QOS, GPUS=l40s:1 unless you set GPUS
bash $ORC status decision        # any time: queue, GPU type names, collect
```

`submit` sends only the tasks that still have runs to do and are not already pending or running, so you can run it as often as you like (see step 6).

`collect` writes `decision/results_v2.csv` and prints the median MAE of each fold (v1 fell from fold 0 to fold 4; v2 should not), the v2/v1 ratios, the Spearman correlation between v1 and v2 MAE over sets, the overlap of the best 5% of sets, where v1's best set ranks in v2, and the v2/v1 runtime ratio per GPU type.

## 5. Full rerun

- `--design full` makes 2,322 tasks, under the limit of 5,000 tasks per job array ([slurm-auto-array](https://rc.byu.edu/wiki/?id=slurm-auto-array)). Throttle how many run at once with `%`, e.g. `--array=0-2321%40` ([Slurm](https://rc.byu.edu/wiki/?id=Slurm)).
- Each task holds about 4 h of v1 runtime, more than the 30 minutes the docs ask as a minimum ([lots of very short jobs](https://rc.byu.edu/wiki/?id=What%27s+the+best+way+to+submit+lots+of+very+short+jobs%3F)). The default `--time=12:00:00` leaves room for slower GPUs and is under the 3-day limit of the L40S, P100 and most preemption-only nodes ([Compute Resources](https://rc.byu.edu/documentation/resources)). Change `--hours` in the manifest step to resize tasks.
- **Preemptable GPUs (the default).** `--qos=standby` gives access to the privately owned A100s and lifts per-user job limits, but the job can be killed at any time ([Slurm](https://rc.byu.edu/wiki/?id=Slurm#preemption)). `rerun.sbatch` sets `--requeue`, which the docs reserve for jobs "specifically designed to bear automatic restarts". This one is: a restarted task skips the runs already in its results file. The warning time before a kill is not documented, so at most the run in progress is lost. The docs do not say whether preemption requeues or cancels a job, so `orc.sh submit` covers the second case (step 6). Pass `QOS=normal` for a non-preemptable job, or `QOS=test` for the smoke test.
- **GPU type under standby.** `GPUS=l40s:1` is the default because it is the only documented type name. For the A100s, run `bash orc.sh status` to list the type names and pass for example `GPUS=a100:1`. If a task lands on a GPU that v1's torch cannot use (H100, H200, B200), or on no GPU, `rerun.py run` stops before recording anything, and the next `submit` sends the task again.
- The scheduler picks the partition; do not set one ([Why won't my job submit?](https://rc.byu.edu/wiki/index.php?page=Why+won%27t+my+job+submit%3F)). `--time` and `--mem` are required, and `rerun.sbatch` sets both.

## 6. Monitor and resubmit

- `squeue --me`, `sacct -X -j <jobid> --format=JobID,State,Elapsed,NodeList` and `whypending <jobid>` ([Using sacct](https://rc.byu.edu/wiki/?id=Using+sacct), [Why won't my job start?](https://rc.byu.edu/wiki/index.php?page=Why+won%27t+my+job+start%3F)). While a task runs, `ssh <node>` and `nvidia-smi` show GPU use ([Getting Started With GPUs](https://rc.byu.edu/wiki/?id=Getting+Started+With+GPUs)).
- Email on completion: add `--mail-user=<address> --mail-type=END,FAIL,REQUEUE` ([job notification emails](https://rc.byu.edu/wiki/?id=Why+am+I+not+getting+job+notification+emails%3F++How+do+I+make+them+stop%3F)).
- Group priority: `sshare -a -l -A <account>` ([Account Coordinators](https://rc.byu.edu/documentation/slurm/account-coord)).
- **Rerunning preempted tasks.** `bash orc.sh submit <design>` resubmits every task that has runs left and is not pending or running, whether it was cancelled by preemption, hit its time limit, or was never submitted. It skips tasks Slurm has requeued, so nothing runs twice. Its settings (`QOS`, `GPUS`, `THROTTLE`, `TIME`) are environment variables, e.g. `GPUS=a100:1 THROTTLE=40 bash orc.sh submit full`.
- **Resubmitting on a schedule.** ORC's docs point to cron on the login nodes for automation ([Two-Factor Authentication](https://rc.byu.edu/wiki/?id=Two-Factor+Authentication)). The cron daemons are clustered, so edit the crontab from any login node (`crontab -e`) and each entry runs on one node at a time. For example, every 3 hours:

  ```
  17 */3 * * * bash -lc 'GPUS=l40s:1 bash ~/matsci-opt-benchmarks/scripts/crabnet_hyperparameter/byu_rc/orc.sh submit decision' >> ~/crabnet_rerun/logs/cron.log 2>&1
  ```

  Remove the line (`crontab -e`) once `status` reports 0 runs to go. A task that fails every time (for example a broken environment) is resubmitted every 3 hours until then, so check `logs/` after the first round.
- Failed runs are recorded with their error and are not retried; delete their lines from the results file to retry them.

## Not covered

- **Launching from GitHub Actions.** Every login needs your password and a verification code, SSH keys are not allowed, and sharing credentials can cost the account ([Two-Factor Authentication](https://rc.byu.edu/wiki/?id=Two-Factor+Authentication), [Acceptable Usage Policy](https://rc.byu.edu/documentation/principles)). For automation the docs point to cron on the login nodes (a page written for RHEL 7; the nodes now run RHEL 9.6) and ask you to open a support ticket for other workarounds. A login-node cron job that pulls from GitHub fits that; a workflow that logs in from GitHub would need the verification-code secret, so ask ORC first.
- **Hopper and Blackwell GPUs** need torch 2.x, and CrabNet 2.0.8 then fails in its SWA optimizer (`'SWA' object has no attribute '_optimizer_step_pre_hooks'`), so that route needs a CrabNet patch and a different software stack from v1.
