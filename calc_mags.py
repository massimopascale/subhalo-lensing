# Require a number of useful python packages:
#
#   astropy
#   numba       ( compiled python functions that run fast )
#   colossus    ( cosmology package; https://bdiemer.bitbucket.io/colossus/ )
#

import scipy.signal as sg
import numpy as np
from numba import *
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
import time
import sys
import os
import multiprocessing


ka0 = float(sys.argv[1])    # external (macro) convergence along line of sight
mut = int(sys.argv[2])  #tangential mag
mlow = int(sys.argv[3]) #low mass cutoff
mhigh = int(sys.argv[4]) #high mass cutoff
Aacc = float(sys.argv[5]) #subhalo abundance
size_no = int(sys.argv[6]); #run number for when multiple runs are done on the hpc with different nodes

glob_dir ='/global/scratch/users/massimopascale/'

local_dir = "run"+ '_' + "{:02d}".format(int(Aacc*100)) +'_'+ "{:02d}".format(int(10*ka0)) +'_'+str(mlow)+'_'+str(mhigh) + '_' + str(mut) + '/';

full_path = glob_dir + local_dir;
os.makedirs(local_dir, exist_ok=True);


size = size_no*0.05; #This was because I did multiple runs with different source sizes, which I increased in increments of 0.05

# Here we shoot a regular array of rays of resolution n1xn2:
n1 = 4096*4
n2 = 1024
no_reals = 1200; #number of realizations
ini_val = 0;
# set up a rectangular range on the image plane [mas]
x1min, x1max = -800, 800
x2min, x2max = -50, 50


x1 = np.linspace(x1min, x1max, n1)
x2 = np.linspace(x2min, x2max, n2)
X1, X2 = np.meshgrid(x1, x2, indexing='ij')
X_list = (X1 + 1.0j*X2).flatten()

@njit
def getSourceMagTemp(ylist,z,sigma):
    """
    Returns a magnification for an input source position given that rays have already been shot
    """
    dist = np.absolute(ylist - z);
    fluxes = (1/(2*np.pi*sigma**2))*np.exp((-1*dist**2)/ (2*sigma**2))
    area = (x1max-x1min)*(x2max-x2min)
    mag = (area)*np.sum(fluxes)/(n1*n2);
    return mag;
    
    
np.random.seed(42);
no_points = 50; #number of sources
p1 = np.random.uniform(-2,2,no_points);
p2 = np.random.uniform(-20,20,no_points);
p = p1 + 1.0j*p2; #source positions, we chose a region that would work for all parameter ranges, but this approach might not work for a fold case

all_mags = np.zeros((no_points*no_reals));


@njit(parallel=True)
def getMags(ylist,ind,all_mags,sig):
    """
    Parallelized magnification using njit
    """
    for j in prange(no_points):
        all_mags[ind*no_points+j] = getSourceMagTemp(ylist,p[j],sigma=sig);

#loop to calculate magnifications for each realization
for i in range(no_reals):
    print('starting realization %i' % i,flush=True);
    start = time.time();
    fn = full_path+'ylist_' + str(i+ini_val) + '.npy';
    Y_list = np.load(fn);
    getMags(Y_list,i,all_mags,size);
    print('finished with time %d' % (start-time.time()), flush=True);
magstr = 'test_mags_' + "{:02d}".format(int(100*size))    
np.save(local_dir+magstr,all_mags)
    


