import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import scipy
from scipy import interpolate
import scipy.integrate as integrate
from tqdm import tqdm
import pickle
from numba import *
from numba import prange, config
from numba import jitclass
from numba import types
from colossus.cosmology import cosmology
from multiprocessing import Pool
from threading import Thread
import itertools
import time
# set cosmology
cosmo = cosmology.setCosmology('planck18')

##########################################################################################

# mass in Msun
# distance in kpc
# velocity in km/sec
# cosmology
h = cosmo.h
Omega_m = cosmo.Om0
rho_crit0 = 279.031*h**2   # present-day critical density [Msun/kpc^3]
GN = 4.27788e-6            # gravitational constant: GN*Msun/kpc/(km/sec/c)^2
GNMsun2KpcInMas = 9.84403e-9   # GN*Msun/kpc/mas/c^2
GNMsun2Kpc = 4.77252e-17       # dimensionless parameter: GN*Msun/kpc/c^2

# some unit conversion constants
Radian2Mas = 1.0/np.pi*180*60*60*1e3
Sigma_crit_unit = 1.66741e15   # lensing critical density [Msun*kpc^-2] corresponding to DLS*DL/DS = 1 kpc

##########################################################################################

# my own interpolation functions

# 1D linear interpolation
@jit(f8(f8, types.Array(f8, 1, 'A', readonly=True), types.Array(f8, 1, 'A', readonly=True)))
def linear_interp_1d(x, X, Z):
    i = np.searchsorted(X, x)
    x1, x2 = X[i], X[i+1]
    u1, u2 = x - x1, x2 - x
    z1, z2 = Z[i], Z[i+1]
    return (z1*u2 + z2*u1)/(x2 - x1)

# 2D bilinear interpolation
@jit(f8(f8, f8, types.Array(f8, 1, 'A', readonly=True), types.Array(f8, 1, 'A', readonly=True), types.Array(f8, 2, 'A', readonly=True)))
def bilinear_interp_2d(x, y, X, Y, Z):
    i = np.searchsorted(X, x)
    j = np.searchsorted(Y, y)
    x1, x2 = X[i], X[i+1]
    u1, u2 = x - x1, x2 - x
    y1, y2 = Y[j], Y[j+1]
    v1, v2 = y - y1, y2 - y
    z11, z12, z21, z22 = Z[i, j], Z[i, j+1], Z[i+1, j], Z[i+1, j+1]
    return (z11*u2*v2 + z12*u2*v1 + z21*u1*v2 + z22*u1*v1)/(x2 - x1)/(y2 - y1)

##########################################################################

# tabulate angular diameter distances in units of kpc
cosmo_logz_grid = np.linspace(np.log10(1e-3), np.log10(499.0), 5000)
cosmo_logdA_grid = np.log10(cosmo.angularDiameterDistance(10.0**cosmo_logz_grid)/h*1e3)

# interpolate angular diameter distance
@jit(f8(f8))
def get_angular_diameter_distance(z):
    return 10.0**linear_interp_1d(np.log10(z), cosmo_logz_grid, cosmo_logdA_grid)

##########################################################################################

# reduction of r_max due to tidal stripping
xi_tidal = 2.0

##########################################################################################

# --------   modelling the spatial distribution of subhalos ------------- #

# model parameters
m0 = 1e10/h
fs = 0.56
Aacc = 0.08
alpha = 0.90
mustar = 0.34
sigst = 1.1
beta = 1.0

#########################################################################################

# ---------  mathematics ------------- #

@jit(f8(f8))
def myHeaviside(x):
    return 0.5*(np.sign(x) + 1.0)

@jit(f8(f8, f8))
def H1(t, mu):
    mu2 = mu**2
    t2 = t**2
    return mu2/2.0/(1.0 + t)/(1.0 + mu2)**2*(-2.0*t*(1.0 + mu2) + 4.0*mu*(1.0 + t)*np.arctan(t/mu) \
            + 2.0*(1.0 + t)*(mu2 - 1)*np.log((1.0 + t)*mu) - (1.0 + t)*(mu2 - 1.0)*np.log(t2 + mu2))

@jit(f8(f8, f8))
def H11(t, mu):
    mu2 = mu**2
    t2 = t**2
    return t/(1.0 + t)**2/(1.0 + t2/mu2)

@jit(f8(f8, f8))
def H2(t, mu):
    mu2 = mu**2
    t2 = t**2
    return mu/2.0/t/(1.0 + mu2)**2*(-2.0*(2.0*mu2 + t*(mu2 - 1))*np.arctan(mu/t) \
           + mu*(2.0*np.pi*mu - 2.0*t*(1.0 + mu2)*np.log(1.0 + 1.0/t) - 2.0*(1.0 + t)*(mu2 - 1)*np.log(t/(1.0 + t)) \
           + 2.0*(mu2 - 1.0)*np.log(mu) + (1.0 + 2.0*t - mu2)*np.log(1.0 + mu2/t2)))

@jit(f8(f8, f8, f8))
def SubhDeflIntegrand(eta, xi, mu):
    eps = 0.0      # artificially introduced to regulate the divergence at eta = 0
    y = (eta**2 + xi**2 + eps**2)**0.5
    return xi*H1(y, mu)/y**3

@jit(f8(f8, f8, f8))
def SubhKaIntegrand(eta, xi, mu):
    eps = 0.0     # artificially introduced to regulate the divergence at eta = 0
    y = (eta**2 + xi**2 + eps**2)**0.5
    return xi**2/y**4*(H11(y, mu) + (2.0*eta**2/xi**2 - 1.0)*H1(y, mu)/y)

@jit(f8(f8, f8, f8))
def SubhGaIntegrand(eta, xi, mu):
    eps = 0.0      # artificially introduced to regulate the divergence at eta = 0
    y = (eta**2 + xi**2 + eps**2)**0.5
    return xi**2/y**4*(3.0*H1(y, mu)/y - H11(y, mu))

# NOTE: these functions CANNOT be broadcasted
def CalcS(xi, mu):
    return scipy.integrate.quad(SubhDeflIntegrand, 0.0, np.inf, args=(xi, mu), epsabs=0.0, limit=200)[0]

def CalcK(xi, mu):
    return scipy.integrate.quad(SubhKaIntegrand, 0.0, np.inf, args=(xi, mu), epsabs=0.0, limit=200)[0]

def CalcG(xi, mu):
    return scipy.integrate.quad(SubhGaIntegrand, 0.0, np.inf, args=(xi, mu), epsabs=0.0, limit=200)[0]

f = open('functions_interp_xi1000_mu600.pckl', 'rb')
interp_data = pickle.load(f, encoding='latin1')   # Python 3 compatible
f.close()
# scipy.interpolate.RectBivariateSpline has requirement that input array must be passed in ascending order
#logS_interp = scipy.interpolate.RectBivariateSpline(interp_data[0], interp_data[1], interp_data[2], kx=3, ky=3)
#logK_interp = scipy.interpolate.RectBivariateSpline(interp_data[0], interp_data[1], interp_data[3], kx=3, ky=3)
#logG_interp = scipy.interpolate.RectBivariateSpline(interp_data[0], interp_data[1], interp_data[4], kx=3, ky=3)

# instead use scipy.interpolate.interp2d; but pay attention to the shape of the third argument matrix
# interpolation kind = 'linear' or 'cubic'
logS_interp = scipy.interpolate.interp2d(interp_data[0], interp_data[1], interp_data[2].transpose(), kind='linear')
logK_interp = scipy.interpolate.interp2d(interp_data[0], interp_data[1], interp_data[3].transpose(), kind='linear')
logG_interp = scipy.interpolate.interp2d(interp_data[0], interp_data[1], interp_data[4].transpose(), kind='linear')

logxi = interp_data[0]
logmu = interp_data[1]
logS = interp_data[2]
logK = interp_data[3]
logG = interp_data[4]

# Range of xi to use tabulated values: [xi_lo, xi_hi]
# Asymptotic formulae are used beyond this range
xi_lo = 1e-3
xi_hi = 3e2

#########################################################################################

# some auxillary functions

@jit(f8(f8))
def f(x):
    return np.log(1.0 + x) - x/(1.0 + x)

@jit(f8(f8))
def f1(x):
    if x <= 1.0:
        return x/4.0
    else:
        return 0.25 + np.log(1.0 + x) - (np.log(4.0)*(1.0 + x) + x - 1.0)/2.0/(1.0 + x)

@jit(f8(f8))
def f2(x):
    if x <= 1.0:
        return x/4.0
    else:
        return x**2/(1.0 + x)**2

# determine the tidal radius in units of the scale radius mu = r_t/r_s
@jit(f8(f8))
def g(mu):
    return mu**2/(1.0 + mu**2)**2*((mu**2 - 1)*np.log(mu) + mu*(np.pi - mu) - 1.0)

@jit(f8(f8))
def rhsf(mu):
    return 1.42*np.log10((mu/g(mu))**0.5) + np.log10(mu)

log10_mu_grid = np.linspace(-3, 3, 200)
rhsf_grid = np.array([ rhsf(10.0**x) for x in log10_mu_grid ])
#rhsf_inv_fit = scipy.interpolate.interp1d(rhsf_grid, log10_mu_grid, bounds_error=False, fill_value='extrapolate',kind='linear')

@jit(f8(f8))
def GetMuFromRhsf(y):
    return 10.0**linear_interp_1d(y, rhsf_grid, log10_mu_grid)

@jit(f8(f8, f8))
def GetLhsf(m, rt):
    return np.log10(2.163*rt*h/1000.0) + 4.63 + np.log10(xi_tidal) - 1.42*np.log10(0.465*(GN*m/rt)**0.5)

# tabulate the function c^3/f(c)
log10_c_grid = np.linspace(0, 3, 500)
log10_c3fc_grid = np.array([ np.log10((10.0**x)**3/f(10.0**x)) for x in log10_c_grid ])
#c3fc_inv_fit = scipy.interpolate.interp1d(log10_c3fc_grid, log10_c_grid, bounds_error=False, fill_value='extrapolate',kind='linear')

@jit(f8(f8))
def GetcFromc3fc(c3fc):
    return 10.0**linear_interp_1d(np.log10(c3fc), log10_c3fc_grid, log10_c_grid)

# tabulate the function mu^3/g(mu)
log10_mu_grid = np.linspace(-3, 3, 500)
log10_mu3gmu_grid = np.array([ np.log10((10.0**x)**3/g(10.0**x)) for x in log10_mu_grid ])
#mu3gmu_inv_fit = scipy.interpolate.interp1d(log10_mu3gmu_grid, log10_mu_grid, bounds_error=False, fill_value='extrapolate',kind='linear')

@jit(f8(f8))
def GetmuFrommu3gmu(mu3gmu):
    return 10.0**linear_interp_1d(np.log10(mu3gmu), log10_mu3gmu_grid, log10_mu_grid)

###################################################################################

# Note that the range of mu tabulated is limited; check 10**logmu

@jit(f8(f8, f8))
def S_interp_val(xi, mu):
    return 10.0**bilinear_interp_2d(np.log10(xi), np.log10(mu), logxi, logmu, logS)

@jit(f8(f8, f8))
def K_interp_val(xi, mu):
    return 10.0**bilinear_interp_2d(np.log10(xi), np.log10(mu), logxi, logmu, logK)

@jit(f8(f8, f8))
def G_interp_val(xi, mu):
    return 10.0**bilinear_interp_2d(np.log10(xi), np.log10(mu), logxi, logmu, logG)

@jit(f8(f8, f8))
def GetS(xi, mu):
    if xi <= xi_lo:
        return S_interp_val(xi_lo, mu)*(xi/xi_lo)**0.8
    elif xi > xi_lo and xi <= xi_hi:
        return S_interp_val(xi, mu)
    else:
        return g(mu)/xi

@jit(f8(f8, f8))
def GetK(xi, mu):
    if xi <= xi_lo:
        return K_interp_val(xi_lo, mu)/(xi/xi_lo)**0.2
    elif xi > xi_lo and xi <= xi_hi:
        return K_interp_val(xi, mu)
    else:
        return K_interp_val(xi_hi, mu)/(xi/xi_hi)**4

@jit(f8(f8, f8))
def GetG(xi, mu):
    if xi <= xi_lo:
        return G_interp_val(xi_lo, mu)
    elif xi > xi_lo and xi <= xi_hi:
        return G_interp_val(xi, mu)
    else:
        return 2.0*g(mu)/xi**2

##########################################################################
#
#  Function used to calculate deflection, convergence and shear for an UNTRUNCATED NFW profile
#
#  S( xi, mu-->inf )
#  K( xi, mu-->inf )
#  G( xi, mu-->inf )
#
#  ( range of tabulation for xi is limited; check this range if needed )

logS_vs_logxi_interp = np.load('logS_vs_logxi_nfw.npy')
logK_vs_logxi_interp = np.load('logK_vs_logxi_nfw.npy')
logG_vs_logxi_interp = np.load('logG_vs_logxi_nfw.npy')

@jit(f8(f8))
def GetS_inf_mu(xi):
    return 10**linear_interp_1d(np.log10(xi), logS_vs_logxi_interp[:, 0], logS_vs_logxi_interp[:, 1])

@jit(f8(f8))
def GetK_inf_mu(xi):
    return 10**linear_interp_1d(np.log10(xi), logK_vs_logxi_interp[:, 0], logK_vs_logxi_interp[:, 1])

@jit(f8(f8))
def GetG_inf_mu(xi):
    return 10**linear_interp_1d(np.log10(xi), logG_vs_logxi_interp[:, 0], logG_vs_logxi_interp[:, 1])


##########################################################################

# --------------- tidally truncated NFW profiles ----------------- #

cofM_datafile_list = ['NFW_c_Planck_z0.dat', 'NFW_c_Planck_z1.dat', 'NFW_c_Planck_z2.dat', 'NFW_c_Planck_z3.dat']
cofM_data = [np.loadtxt(cofM_datafile_list[i]) for i in range(len(cofM_datafile_list))]

## Note that cofM_data tabulates log10 of the concentration parameter

#cofM_fit = [scipy.interpolate.interp1d(cofM_data[i][:,0], cofM_data[i][:,1], \
#            bounds_error=False, fill_value='extrapolate') for i in range(len(cofM_datafile_list))]

logz_grid = np.log10(1.0 + np.array([0.0, 1.0, 2.0, 3.0]))
logM_grid = np.linspace(-6.0, 15.0, 100)
#cofM_fit_grid = np.array([cofM_fit[i](logM_grid) for i in range(len(cofM_fit))])
cofM_fit_grid = np.array([ [ linear_interp_1d(logM_grid[j], cofM_data[i][:,0], cofM_data[i][:,1])  \
                  for j in range(len(logM_grid))  ] for i in range(len(cofM_datafile_list))])
#cofM_fit = scipy.interpolate.RectBivariateSpline(logz_grid, logM_grid, cofM_fit_grid, kx=2, ky=1)

##########################################################################

# critical density at redshift z [M_sun/kpc^3]
@jit(f8(f8))
def Getrhocrit(z):
    return rho_crit0*(Omega_m*(1.0 + z)**3 + 1.0 - Omega_m)

# concentration parameter as a function of M_200 [Msun] and redshift z
@jit(f8(f8, f8))
def GetcofM(M200, z):
    return 10.0**bilinear_interp_2d(np.log10(1.0+z), np.log10(M200*h), logz_grid, logM_grid, cofM_fit_grid)

# find scale radius R_s given M_200, redshift z, concentration factor c
@jit(f8(f8, f8, f8))
def GetRs(M200, z, C):
    return (M200/4.0/np.pi/Getrhocrit(z)/200.0*3.0/C**3)**(1.0/3.0)

# dark matter mass enclosed within radius R of NFW profile
@jit(f8(f8, f8, f8, f8))
def GetMofR(R, M200, Rs, C):
    return M200*f(R/Rs)/f(C)

# dark matter density at radius R of NFW profile
@jit(f8(f8, f8, f8, f8))
def GetrhoofR(R, M200, Rs, C):
    x = R/Rs
    return M200/f(C)/4.0/np.pi/Rs**3/x/(1.0 + x)**2

# dark matter + baryon density at radius R of an NFW halo
@jit(f8(f8, f8, f8, f8))
def GetrhoofRTotal(R, M200, Rs, C):
    x = R/Rs
    if x <= 1.0:
        return M200/f(C)/4.0/np.pi/Rs**3/4.0/x**2
    else:
        return M200/f(C)/4.0/np.pi/Rs**3/x/(1.0 + x)**2

# dark matter + baryon total mass enclosed within R of an NFW halo
@jit(f8(f8, f8, f8, f8))
def GetMofRTotal(R, M200, Rs, C):
    x = R/Rs
    fac = np.log(1.0 + x) - (np.log(4.0)*(1.0 + x) + x - 1.0)/2.0/(1.0 + x)
    if x <= 1.0:
        return M200/4.0/f(C)*x
    else:
        return M200/f(C)*(1.0/4.0 + fac)

# logarithmic slope of dark matter + baryon total enclosed mass at radius R
@jit(f8(f8, f8, f8, f8))
def GetdMdlnRTotal(R, M200, Rs, C):
    return 4.0*np.pi*R**3*GetrhoofRTotal(R, M200, Rs, C)

# logarithmic slope of dark matter only enclosed mass at radius R
@jit(f8(f8, f8, f8, f8))
def GetdMdlnR(R, M200, Rs, C):
    return 4.0*np.pi*R**3*GetrhoofR(R, M200, Rs, C)

# total mass of truncated NFW profile
@jit(f8(f8, f8, f8))
def Getm(m200, c, mu):
    return m200/f(c)*mu**2/(1.0 + mu**2)**2*((mu**2 - 1)*np.log(mu) + mu*(np.pi - mu) - 1.0)

# total mass of the truncated subhalo expected from the equation of tidal radius
@jit(f8(f8, f8, f8, f8, f8, f8))
def GetmTidal(mu, rs, R, M200, Rs, C):
    M = GetMofRTotal(R, M200, Rs, C)
    dMdlnR = GetdMdlnRTotal(R, M200, Rs, C)
    return (mu*rs/R)**3*np.absolute(3.0*M - dMdlnR)

# ---------  modelling the spatial distribution of subhalos  ---------------- #

def dndlnmBias(alp, lnmu, sig):
    sq2 = np.sqrt(2.0)
    return np.exp(lnmu*alp + 0.5*alp**2*sig**2)*scipy.special.erfc((lnmu + alp*sig**2)/sq2/sig) \
           /scipy.special.erfc(lnmu/sq2/sig)

@jit(f8(f8, f8, f8, f8, f8))
def dndlnmacc(macc, R, M200, z, C):
    if C==0:
        C = GetcofM(M200, z)
    Rs = GetRs(M200, z, C)
    rhoR = GetrhoofR(R, M200, Rs, C)
    return Aacc*rhoR/m0/(macc/m0)**alpha

def dndlnm(m, R, M200, z, C):
    if C==0:
        C = GetcofM(M200, z)
    Rs = GetRs(M200, z, C)
    R200 = Rs*C
    rhoR = GetrhoofR(R, M200, Rs, C)
    lnmu = np.log(mustar*(R/R200)**beta)
    return fs*Aacc*rhoR/m0/(m/m0)**alpha*dndlnmBias(alpha, lnmu, sigst)

def ncum(m1, m2, R, M200, z, C):  # m1 < m2
    if C==0:
        C = GetcofM(M200, z)
    Rs = GetRs(M200, z, C)
    R200 = Rs*C
    rhoR = GetrhoofR(R, M200, Rs, C)
    lnmu = np.log(mustar*(R/R200)**beta)
    return fs*Aacc*rhoR/m0/alpha*((m0/m1)**alpha - (m0/m2)**alpha)*dndlnmBias(alpha, lnmu, sigst)

# fraction of dark matter reside in subhalos of mass between m1 and m2
def fsubhalo(m1, m2, R, M200, z, C):
    if C==0:
        C = GetcofM(M200, z)
    Rs = GetRs(M200, z, C)
    R200 = Rs*C
    lnmu = np.log(mustar*(R/R200)**beta)
    return fs*Aacc/(1.0 - alpha)*((m2/m0)**(1.0 - alpha) - (m1/m2)**(1.0 - alpha))*dndlnmBias(alpha, lnmu, sigst)

#### projected number density of subhalos
# differential number density (per logarithmic bin in subhalo mass)

def dn2DdlnmIntegrand(R, m, B, M200, z, C):
    return R/np.sqrt(R**2 - B**2)*dndlnm(m, R, M200, z, C)

def dn2Ddlnm(m, B, M200, z, C):
    return 2.0*scipy.integrate.quad(dn2DdlnmIntegrand, B*(1.0 + 1e-8), np.inf, \
    args=(m, B, M200, z, C), epsabs=0.0, limit=200)[0]

# cumulative number density (subhalo mass between m1 and m2)
def n2DcumIntegrand(R, m1, m2, B, M200, z, C):
    return R/np.sqrt(R**2 - B**2)*ncum(m1, m2, R, M200, z, C)

def n2Dcum(m1, m2, B, M200, z, C):
    return 2.0*scipy.integrate.quad(n2DcumIntegrand, B*(1.0 + 1e-8), np.inf, \
    args=(m1, m2, B, M200, z, C), epsabs=0.0, limit=200)[0]


    
    
    

###########################################################################

Hosthalo_type = deferred_type()
Subhalo_type = deferred_type()
Fold_type = deferred_type()
Macrolens_type = deferred_type()

# ---------  host halo class ------------ #
spec_Hosthalo = [
                 ('M200', f8),
                 ('z', f8),
                 ('zs', f8),
                 ('C', f8),
                 ('Rs', f8),
                 ('R200', f8),
                 ('DL', f8),       # angular diameter distance to the lens plane [kpc]
                 ('DS', f8),       # angular diameter distance to the source plane [kpc]
                 ('DLS', f8),      # angular diameter distance from the lens plane to the lens plane [kpc]
                 ('Sigma_crit', f8),   # critical surface mass density for lensing [Msun * kpc^-2]
                 ('abun', f8)   # abundance for ease of calling
                ]

@jitclass(spec_Hosthalo)
class Hosthalo():
    """
    A parent DM halo hosting many subhalos
    """
    def __init__(self, M200, z, C, zs):
        """
        M200: host halo characteristic mass [Msun]
        z:    host halo (lens) redshift
        C:    concentration parameter
        Rs:   scale radius [kpc]
        R200: characteristic (virial) radius [kpc]
        zs:   source redshift
        """
        self.M200 = M200
        self.z = z

        if C == 0:
            self.C = GetcofM(self.M200, self.z)
        else:
            self.C = C

        self.Rs = GetRs(self.M200, self.z, self.C)
        self.R200 = self.Rs*self.C
        self.zs = zs

        self.DL = get_angular_diameter_distance(self.z)
        self.DS = get_angular_diameter_distance(self.zs)
        self.DLS = self.DS - self.DL*(1.0 + self.z)/(1.0 + self.zs)
        self.Sigma_crit = Sigma_crit_unit*self.DS/self.DL/self.DLS
        self.abun = Aacc;

Hosthalo_type.define(Hosthalo.class_type.instance_type)

# ---------  subhalo class ------------ #
spec_Subhalo = [
                ('host', optional(Hosthalo_type)),
                ('id', i8),       # unique subhalo index
                ('R', f8),        # halocentric distance [kpc]
                ('m', f8),        # subhalo bound mass [Msun]
                ('x1', f8),        # projected (angular) position on the lens plane (x1, x2) [mas]
                ('x2', f8),
                ('c', f8),        # concentration parameter
                ('fc', f8),
                ('Mdenom', f8),
                ('rt', f8),       # subhalo tidal truncation radius [kpc]
                ('mu', f8),
                ('gmu', f8),
                ('rs', f8),       # subhalo scale radius [kpc]
                ('m200', f8),     # subhalo virial mass [Msun]
                ('lst', optional(Subhalo_type)),  # pointer to the last subhalo
                ('nxt', optional(Subhalo_type)),  # pointer to the next subhalo
               ]

@jitclass(spec_Subhalo)
class Subhalo():
    """
    A subhalo
    """
    def __init__(self, host, R, m, x1, x2, sh_id):
        """
        host: host halo (Hosthalo class object)
        R:    halocentric distance [kpc]
        m:    subhalo bound mass [Msun]
        x1, x2: projected position of subhalo center [mas]
        """
        self.id = sh_id

        self.host = host
        self.R = R
        self.m = m
        self.x1 = x1
        self.x2 = x2

        self.c = GetcofM(self.m, self.host.z)  \
              *(1.0 + ((1.5*self.host.R200)**2/(self.R**2 + (0.1*self.host.R200)**2))**0.5/15.0)
        self.fc
        self.fc = f(self.c)

        self.Mdenom = 3.0*GetMofR(self.R, self.host.M200, self.host.Rs, self.host.C)  \
                - GetdMdlnR(self.R, self.host.M200, self.host.Rs, self.host.C)
        self.rt = (self.m/self.Mdenom)**(1.0/3.0)*self.R

        self.mu = GetmuFrommu3gmu(4.0*np.pi*Getrhocrit(self.host.z)*self.R**3/self.Mdenom*200.0/3.0*self.c**3/self.fc)
        self.gmu = g(self.mu)
        self.rs = self.rt/self.mu
        self.m200 = self.m/self.gmu*self.fc

        self.lst = None
        self.nxt = None

    def set_lst(self, lst):
        self.lst = lst

    def set_nxt(self, nxt):
        self.nxt = nxt

    def GetDeflSubhalo(self, x, lam=1.0):
        """
        Compute deflection angle from the subhalo [mas]
        as a function of the ray position on the lens plane x [mas] (passed as a complex number)
        """
        x1 = x.real
        x2 = x.imag
        xr = np.sqrt((self.x1-x1)**2 + (self.x2-x2)**2)
        xi = self.host.DL*xr/self.rs*np.pi/180.0/60/60/1e3   # remember to convert mas into radian
        alp = 4.0*self.host.DLS/self.host.DS*GNMsun2KpcInMas*self.m200/self.fc/self.rs*GetS(xi, self.mu)*lam
        a1 = alp*(x1 - self.x1)/xr
        a2 = alp*(x2 - self.x2)/xr
        return a1 + 1.0j*a2
        
    def GetDeflPoint(self,x,lam=1.0):
        """
        Compute deflection angle from the subhalo [mas] assuming it is a point mass
        as a function of the ray position on the lens plane x [mas] (passed as a complex number)
        """
        x1 = x.real
        x2 = x.imag
        xr = np.sqrt((self.x1-x1)**2 + (self.x2-x2)**2);
        theta = np.arcsin((x2-self.x2)/xr)
        #xi = self.host.DL*xr/self.rs*np.pi/180.0/60/60/1e3   # remember to convert mas into radian
        b = self.host.DL*xr*np.pi/180.0/60/60/1e3;
        alp = 4.0*self.host.DLS/self.host.DS*GNMsun2KpcInMas/b*self.m
        a1 = alp*(x1 - self.x1)/xr
        a2 = alp*(x2 - self.x2)/xr
        return a1 + 1.0j*a2
        
    def GetDeflPJ(self,x,b,s,a,lam=1.0):
        """
        Compute deflection angle from the subhalo [mas] assuming it is a psuedo-jaffe profile
        as a function of the ray position on the lens plane x [mas] (passed as a complex number)
        """
        x1 = x.real
        x2 = x.imag
        xr = np.sqrt((self.x1-x1)**2 + (self.x2-x2)**2);
        theta = np.arcsin((x2-self.x2)/xr)
        #s = 1;
        #a = 10;
        #einstsin radius
        #b = self.host.DL*xr*np.pi/180.0/60/60/1e3;
        #er =  4.0*self.host.DLS/self.host.DS*GNMsun2KpcInMas*self.m/self.host.DL
        er = b;
        alpha_1 = er*(np.sqrt(xr**2 + s**2) - s)/xr;
        alpha_2 = er*(np.sqrt(xr**2 + a**2) - a)/xr;
        alp = alpha_1 - alpha_2
        a1 = alp*(x1 - self.x1)/xr
        a2 = alp*(x2 - self.x2)/xr
        return a1 + 1.0j*a2
    
    
    def GetKaSubhalo(self, x, lam=1.0):
        """
        Compute the convergence ka from the subhalo
        as a function of the ray position on the lens plane x [mas] (passed as a complex number)
        """
        x1 = x.real
        x2 = x.imag
        xr = np.sqrt((self.x1 - x1)**2 + (self.x2 - x2)**2)
        xi = self.host.DL*xr/self.rs*np.pi/180.0/60/60/1e3   # remember to convert mas into radian
        ka = self.host.DLS*self.host.DL/self.host.DS*2.0*GNMsun2Kpc*self.m200/self.fc/self.rs**2*GetK(xi, self.mu)
        return ka*lam

    def GetJacobianSubhalo(self, x, lam=1.0):
        """
        Compute contribution to the elements of the Jacobian matrix from the subhalo
        as a function of the ray position on the lens plane x [mas] (passed as a complex number)
        """
        x1 = x.real
        x2 = x.imag
        xr = np.sqrt((self.x1 - x1)**2 + (self.x2 - x2)**2)
        xi = self.host.DL*xr/self.rs*np.pi/180.0/60/60/1e3   # remember to convert mas into radian
        ka = self.host.DLS*self.host.DL/self.host.DS*2.0*GNMsun2Kpc*self.m200/self.fc/self.rs**2*GetK(xi, self.mu)
        ga_sh = - self.host.DLS*self.host.DL/self.host.DS*2.0*GNMsun2Kpc*self.m200/self.fc/self.rs**2*GetG(xi, self.mu)
        ga1 = ga_sh*((x1 - self.x1)**2 - (x2 - self.x2)**2)/xr**2
        ga2 = ga_sh*2.0*(x1 - self.x1)*(x2 - self.x2)/xr**2
        jac11 = - ka - ga1
        jac12 = - ga2
        jac22 = - ka + ga1
        return np.array([[jac11, jac12], [jac12, jac22]])*lam

Subhalo_type.define(Subhalo.class_type.instance_type)

# ---------  fold lens model class  ---------------- #
spec_Fold = [
            ('ka', f8),
            ('ga', f8),
            ('phi11', f8),
            ('phi12', f8),
            ('phi22', f8),
            ('phi111', f8),
            ('phi112', f8),
            ('phi122', f8),
            ('phi222', f8),
            ('d1', f8),
            ('d2', f8),
            ('d', f8),
            ('phid', f8),
            ('ext_ka_ga', boolean),  # False for a fold model; True for uniform external convergence and shear
            ]

@jitclass(spec_Fold)
class Fold():
    """
    Define a fold model near a macro lensing critical curve
    """
    def __init__(self, ka, phi111=0.0, phi112=0.0, phi122=0.0, phi222=0.0, ga=None, ext_ka_ga=False):
        """
        ka: mean local convergence (coarse-grained)
        ga: mean local shear (coarse-grained)
        phi11, phi12, phi22: second derivatives of the lensing potential
        phi111, phi112, phi122, phi222: third derivatives of the lensing potential [mas^-1]
        d1, d2: two components of the local gradient vector d of the inverse magnification [mas^-1]
        d: magnitude of the local gradient vector d of the inverse magnification [mas^-1]
        phid: orientation of the local gradient vector d of the inverse magnification [radian]
        """
        self.ka = ka

        if ext_ka_ga:
            self.ga = ga
        else:
            self.ga = 1.0 - self.ka

        self.phi11 = self.ka + self.ga
        self.phi12 = 0.0
        self.phi22 = self.ka - self.ga

        if ext_ka_ga:
            self.phi111 = 0.0
            self.phi112 = 0.0
            self.phi122 = 0.0
            self.phi222 = 0.0
        else:
            self.phi111 = phi111
            self.phi112 = phi112
            self.phi122 = phi122
            self.phi222 = phi222

        self.d1 = self.phi111
        self.d2 = self.phi112
        self.d = np.sqrt(self.d1**2 + self.d2**2)
        self.phid = np.angle(self.d2 - 1.0j*self.d1)

    def GetDeflFold(self, x):
        """
        Compute deflection angle from the fold lens model [mas]
        as a function of the ray position on the lens plane x [mas] (passed as a complex number)
        """
        x1 = x.real
        x2 = x.imag
        a1 = self.phi11*x1 + self.phi12*x2 + 0.5*(self.phi111*x1**2 + 2.0*self.phi112*x1*x2 + self.phi122*x2**2)
        a2 = self.phi12*x1 + self.phi22*x2 + 0.5*(self.phi112*x1**2 + 2.0*self.phi122*x1*x2 + self.phi222*x2**2)
        return a1 + 1.0j*a2

    def GetJacobianFold(self, x):
        """
        Compute contribution to the elements of the Jacobian matrix from the fold lens model
        as a function of the ray position on the lens plane x [mas] (passed as a complex number)
        """
        x1 = x.real
        x2 = x.imag
        ka = self.ka + 0.5*(self.phi111*x1 + self.phi112*x2 + self.phi122*x1 + self.phi222*x2)
        ga1 = self.ga + 0.5*(self.phi111*x1 + self.phi112*x2 - self.phi122*x1 - self.phi222*x2)
        ga2 = self.phi12 + self.phi112*x1 + self.phi122*x2
        return np.array([[1.0 - ka - ga1, -ga2], [-ga2, 1.0 - ka + ga1]])
    
    def GetKaFold(self, x):
        """
        Compute the lensing convergence of the fold lens model
        as a function of the ray position on the lens plane x [mas] (passed as a complex number)
        """
        x1 = x.real
        x2 = x.imag
        ka = self.ka + 0.5*(self.phi111*x1 + self.phi112*x2 + self.phi122*x1 + self.phi222*x2)
        return ka


Fold_type.define(Fold.class_type.instance_type)

# ---------  Macro lens model class  ---------------- #
spec_Macrolens = [
                  ('host', optional(Hosthalo_type)),
                  ('fold', optional(Fold_type)),
                  ('sh_list_head', optional(Subhalo_type)),     # head of the linked list of subhalos
                  ('sh_list_tail', optional(Subhalo_type)),     # tail of the linked list of subhalos
                  ('sh_count', i8),                        # total number of subhalos
                  ('xrFoV', f8),        # angular radius of the disk [mas] within which subhalos are populated
                  ('ell', f8),        # angular radius of the disk [mas] within which subhalos are populated
                  ('mu0', f8),        # angular radius of the disk [mas] within which subhalos are populated
                  ('foci', f8),        # angular radius of the disk [mas] within which subhalos are populated
                  ('RFoV', f8),         # proper radius of the disk [kpc] within which subhalos are populated
                  ('B', f8),            # impact parameter of the line of sight to the center of the host halo [kpc]
                  ('ka_sh_mean', f8),   # mean lensing convergence (coarse grained) of all the subhalos generated
                  ('nx1_ray_shoot', i8),
                  ('nx2_ray_shoot', i8),
                  ('x1_min', f8),
                  ('x1_max', f8),
                  ('x2_min', f8),
                  ('x2_max', f8),
                  ('X_list', c16[:]),
                  ('Y_list', c16[:]),
                  ('mu_list', f8[:]),
                  ('n_neb_src', i8),          # total number of nebular sources
                  ('R0_neb_src', f8),         # characteristic source-plane radius for the nebular sources [mas]
                  ('y1_neb_src', f8[:]),    # 1st coorindates for the source plane centers of the nebular sources [mas]
                  ('y2_neb_src', f8[:]),    # 2nd coordinates for the Source plane centers of the nebular sources [mas]
                  ('R_neb_src', f8[:]),       # Source plane radii of the nebular sources (modeled as a 2D Gaussian profile) [mas]
                  ('F_neb_src', f8[:]),       # (Intrinsic) flux normalization of the nebular sources (arbitrary flux units)
                 ]

@jitclass(spec_Macrolens)
class Macrolens():
    """
    A macro lens model setup
    """
    def __init__(self, host, fold, xrFoV=1000.0, ell = 0.5, B=50.0):
        """
        host: host halo parameters (pass a Hosthalo class object)
        fold: fold lens model parameters (pass a Fold class object)
        """
        self.host = host
        self.fold = fold
        self.sh_list_head = None
        self.sh_list_tail = None
        self.ell = ell;
        self.xrFoV = xrFoV
        self.RFoV = self.xrFoV/Radian2Mas*(self.host.DL)
        self.B = B
        self.mu0 = np.arctanh(self.ell)
        self.foci = self.xrFoV/np.cosh(self.mu0);
    
    def RayShoot(self, nx1_ray_shoot=100, nx2_ray_shoot=100, x1_min=-500.0, x1_max=500.0, x2_min=-500.0, x2_max=500.0,multi=False,numthreads=4):
        """
        Parameters to sample the source plane by ray shooting
        We shoot rays within a rectangular region on the image plane (in units of mas):
              x1     from     [ x1_min, x1_max ]
              x2     from     [ x2_min, x2_max ]
        """

        self.nx1_ray_shoot = nx1_ray_shoot
        self.nx2_ray_shoot = nx2_ray_shoot
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_min = x2_min
        self.x2_max = x2_max

        self.X_list = np.zeros((nx1_ray_shoot*nx2_ray_shoot), dtype=np.complex128)
        self.Y_list = np.zeros((nx1_ray_shoot*nx2_ray_shoot), dtype=np.complex128)
        
        x1 = np.linspace(self.x1_min, self.x1_max, self.nx1_ray_shoot)
        x2 = np.linspace(self.x2_min, self.x2_max, self.nx2_ray_shoot)    
        for i in range(self.nx1_ray_shoot):
            for j in range(self.nx2_ray_shoot):
                self.X_list[i*self.nx2_ray_shoot + j] = x1[i] + 1.0j*x2[j]
                self.Y_list[i*self.nx2_ray_shoot + j] = self.XtoY(self.X_list[i*self.nx2_ray_shoot + j])
                
    def setupRayShoot(self, nx1_ray_shoot=100, nx2_ray_shoot=100, x1_min=-500.0, x1_max=500.0, x2_min=-500.0, x2_max=500.0):
        self.nx1_ray_shoot = nx1_ray_shoot
        self.nx2_ray_shoot = nx2_ray_shoot
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_min = x2_min
        self.x2_max = x2_max

        self.X_list = np.zeros((nx1_ray_shoot*nx2_ray_shoot), dtype=np.complex128)
        self.Y_list = np.zeros((nx1_ray_shoot*nx2_ray_shoot), dtype=np.complex128)
        self.mu_list = np.zeros((nx1_ray_shoot*nx2_ray_shoot), dtype=np.float64)     
                   
    def RayShoot_pool(self, param):
        """
        For use with multiprocessing
        Parameters to sample the source plane by ray shooting
        We shoot rays within a rectangular region on the image plane (in units of mas):
              x1     from     [ x1_min, x1_max ]
              x2     from     [ x2_min, x2_max ]
        """
        for p in param:
            i = p[0]
            j = p[1];
            x1 = self.x1_min + i*(self.x1_max - self.x1_min)/(self.nx1_ray_shoot -1);
            x2 = self.x2_min + j*(self.x2_max - self.x2_min)/(self.nx2_ray_shoot -1);
            self.X_list[i*self.nx2_ray_shoot + j] = x1 + 1.0j*x2;
            self.Y_list[i*self.nx2_ray_shoot + j] = self.XtoY(self.X_list[i*self.nx2_ray_shoot + j]);
            
    def RayShoot_single(self,p):
        """
        Shoot a single ray given a point
        """
        i = p[0]
        j = p[1];
        x1 = self.x1_min + i*(self.x1_max - self.x1_min)/(self.nx1_ray_shoot -1);
        x2 = self.x2_min + j*(self.x2_max - self.x2_min)/(self.nx2_ray_shoot -1);
        self.X_list[i*self.nx2_ray_shoot + j] = x1 + 1.0j*x2;
        self.Y_list[i*self.nx2_ray_shoot + j] = self.XtoY(self.X_list[i*self.nx2_ray_shoot + j]);
        
        
    def getSourceMag(self,z,sigma=0.5):
        """
        Returns a magnification for an input source position given that rays have already been shot
        """
        dist = np.absolute(self.Y_list - z);
        fluxes = (1/(2*np.pi*sigma**2))*np.exp((-1*dist**2)/ (2*sigma**2))
        area = (self.x1_max-self.x1_min)*(self.x2_max-self.x2_min)
        mag = (area)*np.sum(fluxes)/(self.nx1_ray_shoot*self.nx2_ray_shoot);
        return mag;
        
    def sampleMags(self,sigma=0.5,nsamples=1000,rand_seed=42):
        """
        Returns a magnification distibution by generating random source locations given that rays have already been shot
        ***This is old and outdated
        """
        np.random.seed(rand_seed)
        #determine range for which we can create samples (r_e<distance from boundary)
        y1 = self.Y_list.real
        y2 = self.Y_list.imag
        y1_min = min(y1) + 2*sigma
        y2_min = min(y2) + 2*sigma
        y1_len = max(y1)-min(y1) - 4*sigma;
        y2_len = max(y2)-min(y2) - 4*sigma;
        
        #we don't necessarily want to center on a pixel, so let's create random positions
        positions = np.random.rand(nsamples,2)
        pos_1 = y1_len*positions[:,0] + y1_min
        pos_2 = y2_len*positions[:,1] + y2_min
        mags = [];
        for i in range(nsamples):
            pos = pos_1[i]+ 1.0j*pos_2[i]
            mags.append(self.getSourceMag(pos,sigma));
        return mags;
    
    def ShootandSample(self,positions,sigma=0.5,nx1_ray_shoot=100, nx2_ray_shoot=100, x1_min=-500.0, x1_max=500.0, x2_min=-500.0, x2_max=500.0):
        """
        Runs ray shooting and then samples magnifications given source positions
        """
        self.RayShoot(nx1_ray_shoot,nx2_ray_shoot,x1_min,x1_max,x1_min,x2_max);
        mags = [];
        for pos in positions:
            mags.append(self.getSourceMag(pos,sigma));
        return mags;
        
    def AddSubhalo(self, R, m, x1, x2, sh_id):
        """
        Generate a new subhalo and append it to the end of the linked list of subhalos: self.sh_list
        """
        sh = Subhalo(self.host, R, m, x1, x2, sh_id)
        if self.sh_count == 0:
            self.sh_list_head = sh
            self.sh_list_tail = sh
        else:
            tpl = self.sh_list_tail
            tpl.set_nxt(sh)
            sh.set_lst(self.sh_list_tail)
            self.sh_list_tail = sh
        self.sh_count = self.sh_count + 1

    def GenRandSubhalos(self, Nlogl=100, Nlogmacc=50, logm_min=6.0, logm_max=10.0, rand_seed=0):
        """
        Generate random subhalos in the vicinity of the line of sight according to some subhalo population model
        And append them one by one to the linked list of subhalos
        """
        # set a random seed
        np.random.seed(rand_seed)

        # make bins in the line of sight coordinate
        # in units of the host halo scale radius Rs
        Nlogl = 100
        logl_grid = np.linspace(-2,2,Nlogl+1) + np.log10(self.host.Rs)
        logl_mid = 0.5*(logl_grid[:-1] + logl_grid[1:])
        dlogl = logl_grid[1:] - logl_grid[:-1]

        # make bins in the subhalo initial mass [Msun]
        Nlogmacc = 50
        logmacc_grid = np.linspace(5.0, 10.0, Nlogmacc+1)
        logmacc_mid = 0.5*(logmacc_grid[:-1] + logmacc_grid[1:])
        dlogmacc = logmacc_grid[1:] - logmacc_grid[:-1]

        total_sh_mass = 0

        # loop over bins of the line of sight coordinate
        for i in range(Nlogl):

            l = 10.0**logl_mid[i]
            R = np.sqrt(l**2 + self.B**2)
            rhoR = GetrhoofR(R, self.host.M200, self.host.Rs, self.host.C)

            # loop over bins in the subhalo initial mass
            for j in range(Nlogmacc):
                macc = 10.0**logmacc_mid[j]
                DnDlnmacc = dndlnmacc(macc, R, self.host.M200, self.host.z, self.host.C)
                # Expectation value for the number of subhalos at given m_acc and at given distance R
                # note that only a fraction f_s survives tidal disruption
                mean = 2.0*np.pi*self.ell*(self.RFoV)**2*l*DnDlnmacc*fs*dlogl[i]*np.log(10.0)*dlogmacc[j]*np.log(10.0)
                nsh = np.random.poisson(lam=mean)  # generate a random subhalo number from Poisson statistics

                # generate subhalos one by one
                for k in range(nsh):

                    ln_m2macc = np.random.normal(loc=np.log(mustar*(R/self.host.R200)**beta), scale=sigst)

                    if(ln_m2macc < 0.0):
                        m = macc*np.exp(ln_m2macc)

                    # apply cuts on the subhalo mass
                    if np.log10(m)>=logm_min and np.log10(m)<logm_max:
                        angle = np.random.uniform(0.0, 2.0*np.pi)
                        radius = np.sqrt(np.random.uniform(0.0, 1))/(self.host.DL)*Radian2Mas
                        self.AddSubhalo(R, m, radius*self.RFoV*np.cos(angle), radius*self.RFoV*self.ell*np.sin(angle), self.sh_count)
                        total_sh_mass = total_sh_mass + m

        self.ka_sh_mean = total_sh_mass/(np.pi*self.ell*self.RFoV**2)/self.host.Sigma_crit

        print("Total number of subhalos = ", self.sh_count)
        print("ka_sh_mean = ", self.ka_sh_mean)
        print("Successfully!!")

    def GetDeflCompensatedRFoV(self, x, lam=1.0):
        """
        Compute compensation to the deflection angle [mas] from a uniform disk of negative surface mass
        as a function of the ray position on the lens plane x [mas] (passed as a complex number)
        """
        #get the elliptic coords from x;
        xx = x.real;
        yy = x.imag;
        bb = xx**2 + yy**2 - self.foci**2;
        
        p = (-1*bb + np.sqrt(bb**2 + 4*self.foci**2 * yy**2))/2/self.foci**2
        q = (-1*bb - np.sqrt(bb**2 + 4*self.foci**2 * yy**2))/2/self.foci**2
        nu = np.arcsin(np.sqrt(p));
        mu = 0.5*np.log(1-2*q+2*np.sqrt(q**2 - q));
        
        if(xx<0 and yy>=0):
            #1st quad
            nu = np.pi-nu;
        elif(xx<=0 and yy<0):
            nu = np.pi + nu;
        elif(xx>0 and yy<0):
            nu = 2*np.pi - nu;
            
        a = -1*self.dphidx(mu,nu,self.foci,self.mu0,self.ka_sh_mean) - 1.0j*self.dphidy(mu,nu,self.foci,self.mu0,self.ka_sh_mean);
        
        return a*lam;

    def XtoY(self, x, lam=1.0):
        """
        Compute the source-plane point y [mas] given the ray coordinate x [mas] on the lens plane
        The deflection angle consists of three pieces of contributions:
            [1] the smooth fold model describing the macro lens model near the macro critical curve;
            [2] sum of contributions from indiviudal subhalos;
            [3] a compensation term which accounts for the fact that some mass
                 is now locked into subhalos but the total mass needs to be conserved;
        """
        y = x - self.fold.GetDeflFold(x) - self.GetDeflCompensatedRFoV(x, lam)
        sh = self.sh_list_head
        while sh is not None:
            y = y - sh.GetDeflSubhalo(x, lam)
            sh = sh.nxt
        return y

    def XtoY_p(self, x, lam=1.0):
        """
        For point source
        Compute the source-plane point y [mas] given the ray coordinate x [mas] on the lens plane
        The deflection angle consists of three pieces of contributions:
            [1] the smooth fold model describing the macro lens model near the macro critical curve;
            [2] sum of contributions from indiviudal subhalos;
            [3] a compensation term which accounts for the fact that some mass
                 is now locked into subhalos but the total mass needs to be conserved;
        """
        y = x - self.fold.GetDeflFold(x) - self.GetDeflCompensatedRFoV(x, lam)
        sh = self.sh_list_head
        while sh is not None:
            y = y - sh.GetDeflPoint(x, lam)
            sh = sh.nxt
        return y
        
    def XtoY_pj(self,x,s,a,lam=1.0):
        #for psuedo jaffe lens
        y = x - self.fold.GetDeflFold(x) - self.GetDeflCompensatedRFoV(x, lam)
        sh = self.sh_list_head
        while sh is not None:
            rt = (a-s)/Radian2Mas*self.host.DL
            k0 = sh.m/(np.pi)/self.host.Sigma_crit/rt;
            k0 = Radian2Mas*k0/self.host.DL
            y = y - sh.GetDeflPJ(x,k0,s,a, lam)
            sh = sh.nxt
        return y
        
    def XtoY_array(self, x, lam=1.0):
        """
        Compute the signed magnification factor for a list of rays
        x should be a list of complex numbers [mas]
        """
        res = np.zeros((len(x)), dtype=np.complex128)
        for i in range(len(x)):
            res[i] = self.XtoY(x[i], lam)
        return res

    def GetJacobianCompensateRFoV(self, x, lam=1.0):
        """
        Compute compensation to the elements of the Jacobian matrix from a uniform disk of negative surface mass
        as a function of the ray position on the lens plane x [mas] (passed as a complex number)
        """
        #get the elliptic coords from x;
        xx = x.real;
        yy = x.imag;
        bb = xx**2 + yy**2 - self.foci**2;
        
        p = (-1*bb + np.sqrt(bb**2 + 4*self.foci**2 * yy**2))/2/self.foci**2
        q = (-1*bb - np.sqrt(bb**2 + 4*self.foci**2 * yy**2))/2/self.foci**2
        nu = np.arcsin(np.sqrt(p));
        mu = 0.5*np.log(1-2*q+2*np.sqrt(q**2 - q));
        
        if(xx<0 and yy>=0):
            #1st quad
            nu = np.pi-nu;
        elif(xx<=0 and yy<0):
            nu = np.pi + nu;
        elif(xx>0 and yy<0):
            nu = 2*np.pi - nu;
        
        dxx = self.dphidxdx(mu,nu,self.foci,self.mu0,self.ka_sh_mean);
        dyy = self.dphidydy(mu,nu,self.foci,self.mu0,self.ka_sh_mean);
        ka = -1*(dxx+dyy)/2
        ga1 = (dxx-dyy)/2;
        ga2 = self.dphidxdy(mu,nu,self.foci,self.mu0,self.ka_sh_mean);
        return np.array([[- ka - ga1, -ga2], [-ga2, - ka + ga1]])*lam
    
    def GetKaSubhalos(self, x, lam=1.0):
        """
        Compute the lensing convergence summed over all subhalos given the 2D ray position x [mas] on the image plane
        x should be passed as a complex number
        """
        ka = 0.0
        sh = self.sh_list_head
        while sh is not None:
            ka = ka + sh.GetKaSubhalo(x, lam)
            sh = sh.nxt
        return ka
    
    def GetKaTotal(self, x, lam=1.0):
        """
        Compute the total lensing convergence (smooth model + subhalos) given the 2D ray position x [mas] on the image plane
        x should be passed as a complex number
        """
        ka = self.fold.GetKaFold(x)
        sh = self.sh_list_head
        while sh is not None:
            ka = ka + sh.GetKaSubhalo(x, lam)
            sh = sh.nxt
        return ka
    
    def GetKaSubhalos_array(self, x, lam=1.0):
        """
        Compute the lensing convergence from all subhalos for a list of rays
        x should be a list of complex numbers [mas]
        """
        res = np.zeros((len(x)), dtype=np.float64)
        for i in range(len(x)):
            res[i] = self.GetKaSubhalos(x[i], lam)
        return res
    
    def GetKaTotal_array(self, x, lam=1.0):
        """
        Compute the total lensing convergence (smooth model + subhalos) for a list of rays
        x should be a list of complex numbers [mas]
        """
        res = np.zeros((len(x)), dtype=np.float64)
        for i in range(len(x)):
            res[i] = self.GetKaTotal(x[i], lam)
        return res

    def GetJacobianTot(self, x, lam=1.0):
        """
        Comptue the lensing Jacobian matrix given the 2D ray position x [mas] on the image plane
        x should be passed as a complex number
        """
        jac = self.fold.GetJacobianFold(x) + self.GetJacobianCompensateRFoV(x, lam)
        sh = self.sh_list_head
        while sh is not None:
            jac = jac + sh.GetJacobianSubhalo(x, lam)
            sh = sh.nxt
        return jac

    def GetMu_array(self, x, lam=1.0):
        """
        Compute the signed magnification factor for a list of rays
        x should be a list of complex numbers [mas]
        """
        res = np.zeros((len(x)), dtype=np.float64)
        for i in range(len(x)):
            res[i] = 1.0/np.linalg.det(self.GetJacobianTot(x[i], lam))
        self.mu_list = res;
        return res

    def mu_pool(self,start,nproc,lam=1.0):
        #for use with multiprocessing
        res = np.zeros(nproc,dtype=np.float64);
        for i in range(start,start+nproc):
            res[i-start] = 1.0/np.linalg.det(self.GetJacobianTot(self.X_list[i], lam))
            #print(self.mu_list[i])
        return res;

    def GetMu(self,x,lam=1.0):
        return 1.0/np.linalg.det(self.GetJacobianTot(x, lam));

    def dxdlam_solve_ray_eqn(self, x, t):
        """
        Evolve the image position according to dx / dlambda = J^-1 * t
        Here t is a two-component constant vector on the source plane
             J is the two-by-two Jacobian matrix evaluated at image position x
             lambda is a parameter in [0, 1]
        ### Note that here both x and t should be passed as complex numbers ###
        """
        jac = self.GetJacobianTot(x)
        jac_inv = np.linalg.inv(jac)
        return (jac_inv[0, 0]*t.real + jac_inv[0, 1]*t.imag)  \
             + (jac_inv[1, 0]*t.real + jac_inv[1, 1]*t.imag)*1.0j

    def EvolveImage(self, x0, y, niter=10):
        """
        Iteratively find the image position that maps to the source position y
        Use image position x0 as an intial guess
        """
        x = x0
        # iteratively improve the guess for the solution x
        for i in range(niter):
            y0 = self.XtoY(x)
            t = y - y0
            x = x + self.dxdlam_solve_ray_eqn(x, t)
        return x

    def FindCounterImages(self, x, niter=15, ntop=50, xtol=0.001, eps=1.0, xabs_max=1000.0):
        """
        Given image position x, find other images that correspond to the same source position
        """
        y = self.XtoY(x)
        dy2 = np.zeros((len(self.Y_list)), dtype=np.float64)

        for i in range(len(self.Y_list)):
            dy2[i] = (y.real - self.Y_list[i].real)**2 + (y.imag - self.Y_list[i].imag)**2

        ind_sort = np.argsort(dy2)
        x0_list = self.X_list[ind_sort][:ntop]
        xsol_list = np.array([ self.EvolveImage(x0, y, niter) for x0 in x0_list ])
        xsol_list_rounded_re = np.array([ xsol_list[i].real - ((xsol_list[i].real)%xtol)  for i in range(len(xsol_list))])
        xsol_list_rounded_im = np.array([ xsol_list[i].imag - ((xsol_list[i].imag)%xtol)  for i in range(len(xsol_list))])
        xsol_list_rounded = xsol_list_rounded_re + 1.0j*xsol_list_rounded_im

        xsol_unique = np.zeros((len(xsol_list_rounded)), dtype=np.complex128)
        unique_count = 0
        reduce = (xsol_list_rounded*np.conjugate(xsol_list_rounded)).real < xabs_max**2
        xsol_remain = xsol_list_rounded[reduce]
        while len(xsol_remain) > 1:
            xsol_unique[unique_count] = xsol_remain[0]
            unique_count = unique_count + 1
            select = ((xsol_remain - xsol_remain[0])*np.conjugate(xsol_remain - xsol_remain[0])).real > eps**2
            xsol_remain = xsol_remain[select]
        if len(xsol_remain) == 1:
            xsol_unique[unique_count] = xsol_remain[0]
            unique_count = unique_count + 1

        return xsol_unique[0:unique_count]

    def gen_rand_nebular_src(self, n_neb_src, y1min, y1max, y2min, y2max, Fmin=1, Fmax=100, R0=10.0, rand_seed=0):
        """
        Generate n_neb_src randomized nebular sources
        Each of them is modeled to have a 2D isotropic Gaussian surface brightness profile of one-dimensional half-width R [mas]
        The center is randomly drawn within the source-plane region:

            y1min < y1 < y1max, y2min < y2 < y2max

        The cumulative distribution for R is assumed to follow an exponential distribution P(>R) = exp(-R/R0), where R0 is a characteristic half-width
        The nebula's flux F integrated over its surface brightness profile is randomly drawn within the range:

            Fmin < F < Fmax

        The cumulative distribution for F is assumed to follow a power-law: N(>F) ~ F^-2
        """
        # set the seed for random number generation
        np.random.seed(rand_seed)

        # total number of nebular sources
        self.n_neb_src = n_neb_src
        # characteristic nebular size
        self.R0_neb_src = R0

        # nebula center
        self.y1_neb_src = np.zeros((self.n_neb_src), dtype=np.float64)
        self.y2_neb_src = np.zeros((self.n_neb_src), dtype=np.float64)
        # nebula size
        self.R_neb_src = np.zeros((self.n_neb_src), dtype=np.float64)
        # nebula flux
        self.F_neb_src = np.zeros((self.n_neb_src), dtype=np.float64)

        for i in range(self.n_neb_src):
            self.y1_neb_src[i] = np.random.uniform(y1min, y1max)
            self.y2_neb_src[i] = np.random.uniform(y2min, y2max)
            self.R_neb_src[i] = self.R0_neb_src*np.log(1.0/np.random.uniform(0.0, 0.9))
            #self.R_neb_src[i] = self.R0_neb_src*np.random.uniform(0.5, 2.0)
            self.F_neb_src[i] = 1.0/(np.random.uniform(1.0/Fmax, 1.0/Fmin))

        return

    def get_nebular_surface_brightness(self, Y_list):
        """
        Provide an array of source-plane coordinates
        Compute the nebular surface brightness at every source-plane location by summing over all nebulae sources
        """

        sb = np.zeros((len(Y_list)), dtype=np.float64)
        y1 = np.zeros((len(Y_list)), dtype=np.float64)
        y2 = np.zeros((len(Y_list)), dtype=np.float64)

        # compute for all source-plane locations
        for i in range(len(sb)):
            y1[i] = Y_list[i].real
            y2[i] = Y_list[i].imag
            # loop over all nebular sources
            for j in range(self.n_neb_src):
                sb[i] += self.F_neb_src[j]/(2.0*np.pi*self.R_neb_src[j]**2)  \
                        *np.exp(-0.5*((y1[i] - self.y1_neb_src[j])**2  \
                                    + (y2[i] - self.y2_neb_src[j])**2)/self.R_neb_src[j]**2)

        return sb
    # ---------  elliptical helper functions  ---------------- #
    def dmudy(self,mu,nu,a):
        return np.cosh(mu)*np.sin(nu)/a/(np.sinh(mu)**2 + np.sin(nu)**2);

    def dnudy(self,mu,nu,a):
        return np.sinh(mu)*np.cos(nu)/a/(np.sinh(mu)**2 + np.sin(nu)**2); 

    def dmudx(self,mu,nu,a):
        return np.sinh(mu)*np.cos(nu)/a/(np.sinh(mu)**2 + np.sin(nu)**2);

    def dnudx(self,mu,nu,a):
        return -1*np.cosh(mu)*np.sin(nu)/a/(np.sinh(mu)**2 + np.sin(nu)**2); 
    
    def dmudydmu(self,mu,nu,a):
        j = (np.sinh(mu)**2 + np.sin(nu)**2);
        return np.sinh(mu)*np.sin(nu)/a/j - np.cosh(mu)*np.sin(nu)*2*np.sinh(mu)*np.cosh(mu)/a/j**2;
    
    def dmudydnu(self,mu,nu,a):
        j = (np.sinh(mu)**2 + np.sin(nu)**2);
        return np.cosh(mu)*np.cos(nu)/a/j - np.cosh(mu)*np.sin(nu)*2*np.sin(nu)*np.cos(nu)/a/j**2;
    
    def dmudxdnu(self,mu,nu,a):
        j = (np.sinh(mu)**2 + np.sin(nu)**2);
        return -1*np.sinh(mu)*np.sin(nu)/a/j - np.sinh(mu)*np.cos(nu)*2*np.sin(nu)*np.cos(nu)/a/j**2;
   
    def dmudxdmu(self,mu,nu,a):
        j = (np.sinh(mu)**2 + np.sin(nu)**2);
        return np.cosh(mu)*np.cos(nu)/a/j - np.sinh(mu)*np.cos(nu)*2*np.sinh(mu)*np.cosh(mu)/a/j**2;
    
    def dnudydmu(self,mu,nu,a):
        j = (np.sinh(mu)**2 + np.sin(nu)**2);
        return np.cosh(mu)*np.cos(nu)/a/j - np.sinh(mu)*np.cos(nu)*2*np.sinh(mu)*np.cosh(mu)/a/j**2;
    
    def dnudydnu(self,mu,nu,a):
        j = (np.sinh(mu)**2 + np.sin(nu)**2);
        return -1*np.sinh(mu)*np.sin(nu)/a/j - np.sinh(mu)*np.cos(nu)*2*np.sin(nu)*np.cos(nu)/a/j**2
    
    def dnudxdmu(self,mu,nu,a):
        j = (np.sinh(mu)**2 + np.sin(nu)**2);
        return -1*np.sinh(mu)*np.sin(nu)/a/j + np.cosh(mu)*np.sin(nu)*2*np.sinh(mu)*np.cosh(mu)/a/j**2;
    
    def dnudxdnu(self,mu,nu,a):
        j = (np.sinh(mu)**2 + np.sin(nu)**2);
        return -1*np.cosh(mu)*np.cos(nu)/a/j + np.cosh(mu)*np.sin(nu)*2*np.sin(nu)*np.cos(nu)/a/j**2;


    def dphidmu(self,mu,nu,a,mu0,k0):
        g1 = a**2 *np.sinh(mu0) * np.cosh(mu0) *k0;
        g2 = a**2 *np.sinh(mu0) * np.cosh(mu0) *k0 * np.log(a/2);
        c2 = 0;#np.sinh(2*mu0)*k0*a**2/4 - g1/2;
        c1 = -1*c2 - k0*a**2 /4/(np.cosh(2*mu0)+np.sinh(2*mu0));
        d1 = -2*c2;
        d2 = g1*mu0 + g2 + 2*c2*mu0 - np.cosh(2*mu0)*k0*a**2 /4;
        f1 = (c1*np.sinh(2*mu0)+c2*np.cosh(2*mu0))/(np.sinh(2*mu0)-np.cosh(2*mu0));
        f2 = -1*f1;
        return np.where(mu<=mu0,2*np.sinh(2*mu)*k0*a**2 /4 + (c1*np.sinh(2*mu)+c2*np.cosh(2*mu))*2*np.cos(2*nu) + d1,
                          (f1*np.sinh(2*mu)+f2*np.cosh(2*mu))*2*np.cos(2*nu) + g1);

    def dphidnu(self,mu,nu,a,mu0,k0):
        g1 = a**2 *np.sinh(mu0) * np.cosh(mu0) *k0;
        g2 = a**2 *np.sinh(mu0) * np.cosh(mu0) *k0 * np.log(a/2);
        c2 = 0;#np.sinh(2*mu0)*k0*a**2/4 - g1/2;
        c1 = -1*c2 - k0*a**2 /4/(np.cosh(2*mu0)+np.sinh(2*mu0));
        d1 = -2*c2;
        d2 = g1*mu0 + g2 + 2*c2*mu0 - np.cosh(2*mu0)*k0*a**2 /4;
        f1 = (c1*np.sinh(2*mu0)+c2*np.cosh(2*mu0))/(np.sinh(2*mu0)-np.cosh(2*mu0));
        f2 = -1*f1;
        return np.where(mu<=mu0,-2*np.sin(2*nu)*k0*a**2 /4 - (c1*np.cosh(2*mu)+c2*np.sinh(2*mu))*2*np.sin(2*nu),
                          -1*(f1*np.cosh(2*mu)+f2*np.sinh(2*mu))*2*np.sin(2*nu));  

    def dphidmudmu(self,mu,nu,a,mu0,k0):
        g1 = a**2 *np.sinh(mu0) * np.cosh(mu0) *k0;
        g2 = a**2 *np.sinh(mu0) * np.cosh(mu0) *k0 * np.log(a/2);
        c2 = 0;#np.sinh(2*mu0)*k0*a**2/4 - g1/2;
        c1 = -1*c2 - k0*a**2 /4/(np.cosh(2*mu0)+np.sinh(2*mu0));
        d1 = -2*c2;
        d2 = g1*mu0 + g2 + 2*c2*mu0 - np.cosh(2*mu0)*k0*a**2 /4;
        f1 = (c1*np.sinh(2*mu0)+c2*np.cosh(2*mu0))/(np.sinh(2*mu0)-np.cosh(2*mu0));
        f2 = -1*f1;
        return np.where(mu<=mu0,k0*a**2 *np.cosh(2*mu) + (c1*np.cosh(2*mu)+c2*np.sinh(2*mu))*4*np.cos(2*nu),
                 (f1*np.cosh(2*mu) + f2*np.sinh(2*mu))*4*np.cos(2*nu));
   
    def dphidmudnu(self,mu,nu,a,mu0,k0):
        g1 = a**2 *np.sinh(mu0) * np.cosh(mu0) *k0;
        g2 = a**2 *np.sinh(mu0) * np.cosh(mu0) *k0 * np.log(a/2);
        c2 = 0;#np.sinh(2*mu0)*k0*a**2/4 - g1/2;
        c1 = -1*c2 - k0*a**2 /4/(np.cosh(2*mu0)+np.sinh(2*mu0));
        d1 = -2*c2;
        d2 = g1*mu0 + g2 + 2*c2*mu0 - np.cosh(2*mu0)*k0*a**2 /4;
        f1 = (c1*np.sinh(2*mu0)+c2*np.cosh(2*mu0))/(np.sinh(2*mu0)-np.cosh(2*mu0));
        f2 = -1*f1;
        return np.where(mu<=mu0,-4*np.sin(2*nu)*(c1*np.sinh(2*mu)+c2*np.cosh(2*mu)),
                       -4*np.sin(2*nu)*(f1*np.sinh(2*mu) + f2*np.cosh(2*mu)));
                 
    def dphidnudnu(self,mu,nu,a,mu0,k0):
        g1 = a**2 *np.sinh(mu0) * np.cosh(mu0) *k0;
        g2 = a**2 *np.sinh(mu0) * np.cosh(mu0) *k0 * np.log(a/2);
        c2 = 0;#np.sinh(2*mu0)*k0*a**2/4 - g1/2;
        c1 = -1*c2 - k0*a**2 /4/(np.cosh(2*mu0)+np.sinh(2*mu0));
        d1 = -2*c2;
        d2 = g1*mu0 + g2 + 2*c2*mu0 - np.cosh(2*mu0)*k0*a**2 /4;
        f1 = (c1*np.sinh(2*mu0)+c2*np.cosh(2*mu0))/(np.sinh(2*mu0)-np.cosh(2*mu0));
        f2 = -1*f1;
        return np.where(mu<=mu0,-1*k0*a**2 * np.cos(2*nu) -(c1*np.cosh(2*mu)+c2*np.sinh(2*mu))*4*np.cos(2*nu),
                       -4*np.cos(2*nu)*(f1*np.cosh(2*mu)+f2*np.sinh(2*mu)));

    def dphidy(self,mu,nu,a,mu0,k0):
        return self.dphidmu(mu,nu,a,mu0,k0)*self.dmudy(mu,nu,a) + self.dphidnu(mu,nu,a,mu0,k0)*self.dnudy(mu,nu,a);

    def dphidx(self,mu,nu,a,mu0,k0):
        return self.dphidmu(mu,nu,a,mu0,k0)*self.dmudx(mu,nu,a) + self.dphidnu(mu,nu,a,mu0,k0)*self.dnudx(mu,nu,a);

    def dphidydy(self,mu,nu,a,mu0,k0):
        dmu = (self.dphidmudmu(mu,nu,a,mu0,k0)*self.dmudy(mu,nu,a) + self.dphidmu(mu,nu,a,mu0,k0)*self.dmudydmu(mu,nu,a) + self.dphidmudnu(mu,nu,a,mu0,k0)*self.dnudy(mu,nu,a) 
                + self.dphidnu(mu,nu,a,mu0,k0)*self.dnudydmu(mu,nu,a))*self.dmudy(mu,nu,a);
        dnu = (self.dphidmudnu(mu,nu,a,mu0,k0)*self.dmudy(mu,nu,a) + self.dphidmu(mu,nu,a,mu0,k0)*self.dmudydnu(mu,nu,a) + self.dphidnudnu(mu,nu,a,mu0,k0)*self.dnudy(mu,nu,a)
               + self.dphidnu(mu,nu,a,mu0,k0)*self.dnudydnu(mu,nu,a))*self.dnudy(mu,nu,a);
        return dmu+dnu;

    def dphidxdx(self,mu,nu,a,mu0,k0):
        dmu = (self.dphidmudmu(mu,nu,a,mu0,k0)*self.dmudx(mu,nu,a) + self.dphidmu(mu,nu,a,mu0,k0)*self.dmudxdmu(mu,nu,a) + self.dphidmudnu(mu,nu,a,mu0,k0)*self.dnudx(mu,nu,a) 
                + self.dphidnu(mu,nu,a,mu0,k0)*self.dnudxdmu(mu,nu,a))*self.dmudx(mu,nu,a);
        dnu = (self.dphidmudnu(mu,nu,a,mu0,k0)*self.dmudx(mu,nu,a) + self.dphidmu(mu,nu,a,mu0,k0)*self.dmudxdnu(mu,nu,a) + self.dphidnudnu(mu,nu,a,mu0,k0)*self.dnudx(mu,nu,a)
               + self.dphidnu(mu,nu,a,mu0,k0)*self.dnudxdnu(mu,nu,a))*self.dnudx(mu,nu,a);
        return dmu+dnu;

    def dphidxdy(self,mu,nu,a,mu0,k0):
        dmu = (self.dphidmudmu(mu,nu,a,mu0,k0)*self.dmudy(mu,nu,a) + self.dphidmu(mu,nu,a,mu0,k0)*self.dmudydmu(mu,nu,a) + self.dphidmudnu(mu,nu,a,mu0,k0)*self.dnudy(mu,nu,a) 
                + self.dphidnu(mu,nu,a,mu0,k0)*self.dnudydmu(mu,nu,a))*self.dmudx(mu,nu,a);
        dnu = (self.dphidmudnu(mu,nu,a,mu0,k0)*self.dmudy(mu,nu,a) + self.dphidmu(mu,nu,a,mu0,k0)*self.dmudydnu(mu,nu,a) + self.dphidnudnu(mu,nu,a,mu0,k0)*self.dnudy(mu,nu,a)
               + self.dphidnu(mu,nu,a,mu0,k0)*self.dnudydnu(mu,nu,a))*self.dnudx(mu,nu,a);
        return dmu+dnu;

    def dphidydx(self,mu,nu,a,mu0,k0):
        dmu = (self.dphidmudmu(mu,nu,a,mu0,k0)*self.dmudx(mu,nu,a) + self.dphidmu(mu,nu,a,mu0,k0)*self.dmudxdmu(mu,nu,a) + self.dphidmudnu(mu,nu,a,mu0,k0)*self.dnudx(mu,nu,a) 
                + self.dphidnu(mu,nu,a,mu0,k0)*self.dnudxdmu(mu,nu,a))*self.dmudy(mu,nu,a);
        dnu = (self.dphidmudnu(mu,nu,a,mu0,k0)*self.dmudx(mu,nu,a) + self.dphidmu(mu,nu,a,mu0,k0)*self.dmudxdnu(mu,nu,a) + self.dphidnudnu(mu,nu,a,mu0,k0)*self.dnudx(mu,nu,a)
               + self.dphidnu(mu,nu,a,mu0,k0)*self.dnudxdnu(mu,nu,a))*self.dnudy(mu,nu,a);
        return dmu+dnu;

Macrolens_type.define(Macrolens.class_type.instance_type)

@njit(parallel=True)
def shoot_ext(mac):
    #Ray shoot using njit parallel rather than pool
    x1 = np.linspace(mac.x1_min, mac.x1_max, mac.nx1_ray_shoot)
    x2 = np.linspace(mac.x2_min, mac.x2_max, mac.nx2_ray_shoot)    
    for i in prange(mac.nx1_ray_shoot):
        for j in range(mac.nx2_ray_shoot):
            #print(j);
            mac.X_list[i*mac.nx2_ray_shoot + j] = x1[i] + 1.0j*x2[j]
            mac.Y_list[i*mac.nx2_ray_shoot + j] = mac.XtoY(mac.X_list[i*mac.nx2_ray_shoot + j])

'''
@njit(parallel=True)
def mu_ex(mac,x,lam=1.0):
    """
    Compute the signed magnification factor for a list of rays
    x should be a list of complex numbers [mas]
    """
    res = np.zeros((len(x)), dtype=np.float64)
    for i in prange(len(x)):
        res[i] = 1.0/np.linalg.det(mac.GetJacobianTot(x[i], lam))
    return res
'''
#####################################################################################
'''
    x = mac.X_list;
    
    for i in prange(start,start+nproc):
        mac.mu_list[i] = 1.0/np.linalg.det(mac.GetJacobianTot(x[i], lam))
    return;
'''
# def EvolveImage(x0, y, xtoy, dxdlam, niter=1, npt=2):
#     """
#     Find the image position that maps to the source position y
#     Use image position x0 as an intial guess
#     Need to provide two functions:
#         (1) xtoy: map image position to source position
#         (2) dxdlam: evaluate J(x)^-1 dot t, where t is a given constant vector
#     """
#     x = x0
#     # iterative algorithm
#     for i in range(niter):
#         y0 = xtoy(x)
#         t = y - y0
#         t_vec = np.array([t.real, t.imag])
#         x_vec = np.array([x.real, x.imag])
#         lam = np.linspace(0.0, 1.0, npt)
#         x_vec = integrate.odeint(dxdlam, x_vec, lam, args=(t_vec,))[-1]
#         x = x_vec[0] + 1.0j*x_vec[1]
#     return x
