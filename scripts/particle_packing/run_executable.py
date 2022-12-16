from os import getcwd
from subprocess import run
from os.path import join

print(getcwd())
# fpath = join("..", "..", "boppf", "utils", "particle_packing_sim.exe")
fpath = join(
    "src", "matsci_opt_benchmarks", "particle_packing", "utils", "PackingGeneration.exe"
)
run(fpath)
