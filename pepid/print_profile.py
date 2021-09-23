import pstats
pstats.Stats('profile0.log').sort_stats(pstats.SortKey.CUMULATIVE).print_stats()

