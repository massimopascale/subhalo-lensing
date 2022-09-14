import scipy.signal as sg
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as cls
from tqdm import tqdm
import nfwsubhalo_jit_el as shj
from colossus.cosmology import cosmology
# initialize Planck 2018 concordance cosmology
cosmo = cosmology.setCosmology('planck18')
from astropy import units as u
from astropy import constants as c
import itertools
import multiprocessing as mp
import time
import sys
import os
import multiprocessing

# This is an example how we can set up a host dark matter halo
# and generate randomized substructures

zl = 0.435     # lens redshift
zs = 2.37      # source redshfit

# luminosity distances
dl = cosmo.angularDiameterDistance(zl)
ds = cosmo.angularDiameterDistance(zs)
dls = ds - dl*(1 + zl)/(1 + zs)

M200 = 1e15     # host halo r200 mass [Msun]
C = 8           # host halo concentration parameter c200

#input parameters
ka0 = float(sys.argv[1])#0.6;       # external (macro) convergence along line of sight
mut = int(sys.argv[2])#40;
mlow = int(sys.argv[3])#5;
mhigh = int(sys.argv[4])#8;
no_runs = int(sys.argv[5])#6;
run_no = int(sys.argv[6])#1;


#params for run number, etc. have to do in batches across nodes for memory and speed (we can only parallelize within a single node, not across nodes - could be fixed with mpi)
start_val = 300;
no_realizations = 900;
batch_size = int(no_realizations/no_runs);
seeds = range(run_no*batch_size+start_val, (run_no+1)*batch_size+start_val)



print("Using %d cores" %(multiprocessing.cpu_count()))
#print("Kappa %d" %(ka0));


# specify the background magnification magnification and shear
mur = 1/(-2*(ka0 - 1) -1/mut)
mutot = mut*mur                
ga0 = 0.5*(1/mur - 1/mut)
# an option to set up uniform external (macro) convergence and shear
ext_ka_ga = True

# impact parameter of line of sight (from the center of the host halo)
RE = 170.0 # [kpc]
# set third derivatives of the Fermat potential to be all zero if using uniform external convergence and shear
# only set third derivatives for a fold model
phi111 = 0.0
phi112 = 0.0
phi122 = 0.0
phi222 = 0.0

# create a macro lens model
# uniform convergence and shear in this case
fold = shj.Fold(ka0, phi111, phi112, phi122, phi222, ga0, ext_ka_ga)

# create a host (cluster) dark matter halo
# also set lens (cluster) and source redshifts
host = shj.Hosthalo(M200, zl, C, zs)

Aacc = host.abun;

#directory stuff
glob_dir ='/global/scratch/users/massimopascale/'#'/home/moss/repos/sunburst-lensing/hpc/'

local_dir = "run"+ '_' + "{:02d}".format(int(Aacc*100)) +'_'+ "{:02d}".format(int(10*ka0)) +'_'+str(mlow)+'_'+str(mhigh) + '_' + str(mut) + '/';

full_path = glob_dir + local_dir;
os.makedirs(full_path, exist_ok=True);

#shooting params
# Here we shoot a regular array of rays: resolution is n1xn2, spanning x1min to x1max and x2min to x2max [mas]
n1 = 4096*4
n2 = 256*4
x1min, x1max = -800, 800
x2min, x2max = -50, 50
x1 = np.linspace(x1min, x1max, n1)
x2 = np.linspace(x2min, x2max, n2)

for seed in seeds:
    print('starting realization with seed %i' % seed);
    # create a macro lens background
    # an ellipse of ellipticity 0.2 and semimajor axis 1000 mas
    paramlist = (itertools.product(x1,x2));
    mac = shj.Macrolens(host, fold, xrFoV=1000.0,ell=0.2, B=RE)

    # this can be used to add subhalo individually
    # (check out source code)
    #mac.AddSubhalo(300.0, 1e5, 0.0, 0.0, 0)

    # this generates randomized subhalos according to a CDM model
    # (check out source code)
    mac.GenRandSubhalos(100, 20, mlow, mhigh, seed)

    # this can be used to uniformly shoot rays on the image plane
    # (check out source code)
    #mac.RayShoot()
    start = time.time();
    print('starting ray shoot...')

    mac.setupRayShoot(n1,n2,x1min,x1max,x2min,x2max); #setup rayshooting parameters so we can calc in parallel with multiprocessing pool
    
    #shj.shoot_ext(mac); #in case you want to use jit instead, pool is faster
    
    #shoot a single ray
    def tempshoot(par):
        p1 = par[0];
        p2 = par[1];
        return mac.XtoY(p1+1.0j*p2);
        #return mac.XtoY_pj(p1+1.0j*p2,0.1,0.3)
        
    pool = mp.Pool();
    ylist = pool.map(tempshoot,paramlist);
    pool.close();
    np.save(full_path+'ylist_'+str(seed),ylist);

    print('rayshoot complete, finished with time %d' % (start-time.time()));
    '''
    print('now starting mu');
    #this computes mu maps analytically, takes a long time
    def tempmu(par):
        p1 = par[0];
        p2 = par[1];
        return mac.GetMu(p1+1.0j*p2);
    pool = mp.Pool();
    paramlist = (itertools.product(x1,x2));
    mulist = pool.map(tempmu,paramlist);
    pool.close();
    np.savetxt('mulist_'+str(seed),mulist);
    print('mu calc complete, finished with time %d' % (start-time.time()));
    '''
    del mac;
    #del mulist;
    del ylist;




