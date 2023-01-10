from os import getcwd
from os.path import join
from subprocess import run

print(getcwd())
# fpath = join("..", "..", "boppf", "utils", "particle_packing_sim.exe")
fpath = join(
    "src", "matsci_opt_benchmarks", "particle_packing", "utils", "PackingGeneration.exe"
)
run(fpath)
