# subhalo-lensing
-nfwsubhalo_jit_el.py contains all of the functions for creating folds, populating subhalos, shooting rays, calculating magnification maps analytically, etc.

-rayshoot.py will create a realization of subhalos and then shoot rays, saving the mapping between X and Y (these files are usually large)

-calc_mags.py will take the rayshooting results, then calculate magnification statistics for gaussian sources

-shoot_parallel.sh and analyze_parallel.sh are bash scripts to run rayshoot.py and calc_mags.py respectively in batchs on the hpc

-*_nfw.npy and NFW_*.dat are helper files with tabulated functions for subhalos.
