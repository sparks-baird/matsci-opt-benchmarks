from os import getcwd
from subprocess import run
from os.path import join

fpath = join("..", "..", "boppf", "utils", "particle_packing_sim.exe")
run(fpath)
