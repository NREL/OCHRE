import cProfile, pstats
from guppy import hpy

from ochre import Dwelling
from bin.run_dwelling import dwelling_args

# h = hpy()
# print(h.heap().size // 1024 // 1024)  # in MB

# Initialization
# dwelling = Dwelling(**dwelling_args)
with cProfile.Profile() as profile:
    dwelling = Dwelling(**dwelling_args)
ps = pstats.Stats(profile).sort_stats('cumulative').print_stats(50)
# print(h.heap().size // 1024 // 1024)  # in MB

# Simulation
# df, metrics, hourly = dwelling.simulate()
# cProfile.run("df, metrics, hourly = dwelling.simulate()", sort='cumulative')
