#!/usr/bin/env python3
#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
    Created on Thur Jan 9 2021
    Latest Version: Mon Apr 21 2021
    
    @author: Shane
    
    1-Dimensional model of debris discs
    Grains affected by radiation pressure (using beta = Frad/Fgrav)
    Optical Constants for grain: miepython
    """

#libraries
import miepython as mpy
import re
import pandas as pd
import astropy
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter, FixedFormatter, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import random
import numpy as np
import math
import subprocess
import os
from os import path
import time
from skimage.transform import rescale
from astropy.io import fits
from astropy.io import ascii
from astropy.convolution import CustomKernel
from astropy.convolution import convolve
from astropy.modeling.models import BlackBody as BBody
from astropy import units as u
from PyAstronomy.pyasl import planck
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import shift
from scipy.integrate import simps, quad, trapz
from photutils.centroids import fit_2dgaussian
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import shift
from scipy import interpolate
from scipy.integrate import quad, simps, trapz
import scipy.stats as st
from scipy.special import logit, expit
from scipy.interpolate import CubicSpline
from csaps import csaps
#import fast_histrogram
from fast_histogram import histogram1d, histogram2d
import emcee
import corner
#import numba
#from numba import njit, jit, prange

#Constants (kg-m-s)
#Universal Contants
G = 6.673705*10**-11        #Gravitational constant
c = 299792458           #Speed of Light
h = 6.62607004*10**-34      #Planck's constant
kb = 1.38064852*10**-23     #boltzman constant

#Sun values (Note in Solar units L=M=R=1)
L_s = 3.845*10**26           #luminosity
M_s = 1.9885*10**30          #mass
R_s = 6.957*10**8             #radius
T_s = 5770                #surface temperature (K)

#Solar System Units
au = 149597870700       #astronomical unit defined by IAU (https://cneos.jpl.nasa.gov/glossary/au.html)
Me = 5.972*10**24           #Earth Mass [kg]
pc = 3.086*10**16           #parsec [m]

#Mathematical constants
pi = 3.1415926535       #pi (10 d.p.)


#initial time
t0 = time.time()

#Format tick labels
formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))


#-----------------------------------------------------------------#
#Function: Bolometric Black Body function as a function of wavelength
def Blam(T,w):
  
    #Inputs:
    #T: Temperature (float) [Kelvin]
    #w: Wavelength range (array float) [metres]
    L = (2*h*c**2/(w**5))*(np.exp(h*c/(w*kb*T))-1)**-1
    
    #Ouput
    #Luminosity as function of wavelength [SI Units]
    
    return L

#-----------------------------------------------------------------#


#-----------------------------------------------------------------#
##Function:Chi Squared Test ##
##Input: discreet values + uncertainty, function to fit
def Chi2(phot_wav,phot_flux,phot_unc,func_wav,func_flux):
    func_wf = interpolate.interp1d(func_wav,func_flux)
    flam = func_wf(phot_wav)
    #print(flam)
    c2 = 0
    for i in range(len(phot_wav)):
        c2 = c2 + (((phot_flux[i]-flam[i]))/phot_unc[i])**2

    return [c2]

#-----------------------------------------------------------------#
##emcee functions###
##Inlike() function for emcee fitting stellar atmosphere
#def lnlike(phot_wav,phot_flux,phot_unc,func_wav,func_flux):
def lnlike(theta,phot_wav,phot_flux,phot_unc):
#    sm,dfrac,qv,rm,rw = theta
#    func_flux = DustyMM(sm,smax_r,dfrac,qv,rm,rw,rin,rout)
    amp = theta
    func_flux = amp*flux_nu
    func_wf = interpolate.interp1d(wr,func_flux)
    flam = func_wf(phot_wav)
#    #print(flam)
    c2 = 0
    for i in range(len(phot_wav)):
        c2 = c2 + (((phot_flux[i]-flam[i]))/phot_unc[i])**2
#    lp = -0.5*np.sum( (phot_flux - flam) /phot_unc **2)
    chi2 = -0.5*c2

    return chi2

def lnprior(theta):
    amp = theta
    if 0.01 < amp < 20.0:
        return 0.0
    else:
        return -np.inf


def lnprob(theta,wav,flx,unc):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,wav,flx,unc)

##Inlike() function for emcee 2 Temperature Model
#def lnlike(phot_wav,phot_flux,phot_unc,func_wav,func_flux):
def lnlike2(theta,phot_wav,phot_flux,phot_unc):
    Td_w, Td_c, Ad_w, Ad_c = theta
    func_flux = bbody2temp(Td_w, Td_c, Ad_w, Ad_c, wr)
    func_wf = interpolate.interp1d(wr,func_flux)
    flam = func_wf(phot_wav)
#    #print(flam)
    c2 = 0
    for i in range(len(phot_wav)):
        c2 = c2 + (((phot_flux[i]-flam[i]))/phot_unc[i])**2
#    lp = -0.5*np.sum( (phot_flux - flam) /phot_unc **2)
    chi2 = -0.5*c2

    return chi2

def lnprior2(theta):
    Td_w, Td_c, Ad_w, Ad_c = theta
    if 50 < Td_w < 500.0 and 1 < Td_c < 100 and 0.0001 < Ad_w < 2 and 0.00001 < Ad_c < 2:
        return 0.0
    else:
        return -np.inf


def lnprob2(theta,wav,flx,unc):
    lp = lnprior2(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike2(theta,wav,flx,unc)

#-----------------------------------------------------------------#
#Two Temperature Black body model
def bbody2temp(Td_w, Td_c, Ad_w, Ad_c, wrange):
    #Td_w: temperature warm inner belt [K]
    #Td_c: temperature cool outer belt [K]
    #Ad_w: amplitude of flux for warm inner belt [Jy]
    #AD_c: amplitude of flux for cool outer belt [Jy]
    
    bb_w = BBody(temperature=Td_w*u.K)
    bb_c = BBody(temperature=Td_c*u.K)
    
    wav = wrange*u.micron
    
    Temp2DD = Ad_w*bb_w(wav)/np.max(bb_w(wav)) + Ad_c*bb_c(wav)/np.max(bb_c(wav))
    
    return Temp2DD
    

#Single Temperature Black body model
def bbody1temp(Td, Ad, wrange):
    #Td_w: temperature warm inner belt [K]
    #Td_c: temperature cool outer belt [K]
    #Ad_w: amplitude of flux for warm inner belt [Jy]
    #AD_c: amplitude of flux for cool outer belt [Jy]
    
    bb = BBody(temperature=Td*u.K)
    
    wav = wrange*u.micron
    
    Temp1DD = Ad*bb(wav)/np.max(bb(wav))
    
    return Temp1DD
    

#-----------------------------------------------------------------#
## Wavelength Space##
wr = np.geomspace(0.1,3000,3000) # Wavelength space in microns
#wr1 = np.geomspace(0.1,3000,3000) # Wavelength space in microns
#WRange = [1,10,100,1000] #Add specific wavelengths of interest???
#wr2 = np.array(WRange)
#wr3 = np.append(wr1,wr2)
#wr = np.sort(wr3)

#-----------------------------------------------------------------#
## Host Star Properties
object = 'AU-Mic'

#Import CASSIS
mir_spek = np.loadtxt('cassis_tbl_opt_11401984.txt', dtype=[('wavelength', float),('flux_spek', float),('flux_unc', float),('flag1', float),('flag2', float),('flag3', float)])

#Convert, Extrapolate and Calibrate with Blackbody Spectrum of similar temperature
mir_wave = mir_spek["wavelength"] #Convert from Angstroms to Microns
mir_spec = mir_spek["flux_spek"]
mir_unc = mir_spek["flux_unc"]
mir_2spe = mir_spek["flag2"]

#Find synthetic photometry
#lam_syn = 35
lam_synr = np.linspace(34.5,35.5,101)
f_mirs = interpolate.interp1d(mir_wave, mir_spec, kind = 'linear',fill_value = 'extrapolate')
f_mirs_u = interpolate.interp1d(mir_wave, mir_unc, kind = 'linear',fill_value = 'extrapolate')
f_syn = f_mirs(lam_synr)
f_syn_u = f_mirs_u(lam_synr)
f_syn_35 = np.round(np.sum(f_syn)/len(f_syn),3)
f_syn_35_u = np.round(np.max(f_syn_u),3)




#Stellar Photometric values for comparison

#All flux values
#lam = [0.43,    0.55,   3.350,  4.6,    5.800,  8,      9.0000, 11.60,  12.00,  18.00,  22.10,  24,     25,     35.0,   60,     70.00,160.0,250.0,350.00,450.0,500.00,850.00,1300.0]
#flx = [0.278,   0.985,  4.820,  4.260,  2.200,  1.270,  0.9120, 0.543,  0.517,  0.246,  0.183,  0.158,  0.244,  0.11,   0.269,  0.231,0.243,0.134,0.0844,0.085,0.0476,0.0144,0.0085]
#unc = [0.104,   0.185,  0.199,  0.199,  0.055,  0.0020, 0.0022, 0.060,  0.030,  0.036,  0.021,  0.03,   0.063,  0.046,  0.02,   0.016,0.017,0.008,0.0054,0.042,0.0038,0.0018,0.0020]
lam = [0.43,    0.55,   3.350,  4.6,    5.800,  8,      9.0000, 11.60,  18.00,  22.10,  24,     35,         70.00,160.0,250.0,350.00,450.0,500.00,850.00,1300.0]
flx = [0.278,   0.985,  4.820,  4.260,  2.200,  1.270,  0.9120, 0.543,  0.246,  0.183,  0.158,  f_syn_35,   0.231,0.243,0.134,0.0844,0.085,0.0476,0.0144,0.0085]
unc = [0.104,   0.185,  0.199,  0.199,  0.055,  0.0020, 0.0022, 0.060,  0.036,  0.021,  0.03,   f_syn_35_u, 0.016,0.017,0.008,0.0054,0.042,0.0038,0.0018,0.0020]

#optical and near/mid infrared photometry
onr_lam = [0.43, 0.55,3.350,5.800,9.0000,11.60,12.00]
onr_flx = [0.278,0.985,4.820,2.200,0.9120,0.543,0.517]
onr_unc = [0.104,0.185, 0.199,0.055,0.0022,0.060,0.030]
#Schuppler et al. (20150

##Host Star properties in Solar units##
Ls = 0.09         #Luminosity of Star [Solar Luminosity]
Ms = 0.5        #Mass of Star [Solar Masses]  Tess light curve
Rs = 0.75          #Radius of Star [Solar Radius]
Ts = 3700       #star temperature [K]
dpc = 9.71       #distance to star system [pc] - check GAIA

##Star Flux values: Blackbody + Spectrum (if available)
##Blackbody Stellar spectrum (default)
#bb = BBody(temperature=Ts*u.K)
#d = dpc*pc   #distance to Star [m]
#Rad = Rs*R_s   #Star Radius: Sun [m]
#As = pi*(Rad/d)**2 #Amplitude of Star's blackbody planck fn (i.e. for this case: cross-section of Sun) and convert to Jansky (as BBody units are given in: erg / (cm2 Hz s sr))
#flux_s = As*Blam(Ts,wr*10**-6)*10**26*((wr*10**-6)**2)/(c)

##Stellar Spectrum##
#Import Model Atmosphere values
f_spek = np.loadtxt('au-mic-3500-m-1g4.txt', dtype=[('wave', float),('fspek', float)])

#Convert, Extrapolate and Calibrate with Blackbody Spectrum of similar temperature
wave = f_spek["wave"]*0.0001 #Convert from Angstroms to Microns
fspek = f_spek["fspek"]

#Extrapolate/Interplate F(wave)
fun_spek = interpolate.interp1d(np.log10(wave),np.log10(fspek),kind = 'linear', fill_value = 'extrapolate')

#Convert flux (wave) to flux (frequency)
sc_flux = 11.55
f_10spek = fun_spek(np.log10(wr))
flux_nu = (10**f_10spek)*((wr*10**-6)**2)/(c)  #unscaled flux (frequency)
flux_nu = sc_flux*flux_nu/np.max(flux_nu)

fun_f_nu = interpolate.interp1d(wr,flux_nu,kind = 'linear', fill_value = 'extrapolate')
f_sval = fun_f_nu(lam)

f_sub = flx[7:-1] - f_sval[7:-1]
f_subw = lam[7:-1]
f_subu = f_sub*unc[7:-1]/flx[7:-1]



##Emcee Inputs##
nwalkers = 50
niter = 200
initial = np.array([200,50,0.015,0.25]) #Variables to be tested: 'Td_w', 'Td_c', 'Ad_w', 'Ad_c'
ndim = len(initial)
p0 = [np.array(initial) + np.array([random.uniform(-5,55),random.uniform(-5,5),random.uniform(-0.1,0.1),random.uniform(-0.1,0.1)]) for i in range(nwalkers)]
#p0 = [np.array(initial) + np.array([random.uniform(-1,1)],) for i in range(nwalkers)]
print('Walkers for simulation:')
print(p0)
#print(zed)
data = f_subw,f_sub,f_subu


def main(p0,nwalkers,niter,ndim,lnprob2,data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state
    
sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob2,data)

samples = sampler.flatchain

#plt.show()

labels = ['Td_w', 'Td_c', 'Ad_w', 'Ad_c']
fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84])
cornerfig = object+'_2TempFit'+'_Emcee_corner_nwalkers'+str(nwalkers)+'_niter'+str(niter)+'.eps'
#fig.savefig('Emcee_corner_30_500.eps')
fig.savefig(cornerfig)

Td_w_mcmc, Td_c_mcmc, Ad_w_mcmc, Ad_c_mcmc = np.median(samples, axis=0)

print(f'Td_w: {Td_w_mcmc}')
print(f'Td_c: {Td_c_mcmc}')
print(f'Ad_w: {Ad_w_mcmc}')
print(f'Ad_c: {Ad_c_mcmc}')

temp2 = bbody2temp(Td_w_mcmc, Td_c_mcmc, Ad_w_mcmc, Ad_c_mcmc, wr)

warmbelt = bbody1temp(Td_w_mcmc,Ad_w_mcmc,wr)
coolbelt = bbody1temp(Td_c_mcmc,Ad_c_mcmc,wr)

f_wbelt = interpolate.interp1d(wr,warmbelt,kind = 'linear', fill_value = 'extrapolate')

#substract warm belt flux values from original flux data set
flx_ws = flx - f_wbelt(lam)
unc_ws = flx_ws*unc/flx


df_aumic_cold = pd.DataFrame({'Wavelength': lam, 'Flux [Jy]': flx_ws, 'Unc(abs) [Jy]': unc_ws})
df_aumic_cold.to_csv('df_aumic_cold.txt', sep='\t',index=False, header = False)



plt.clf()
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6,4.8))
#ax.plot(lam,flx,')
#ax.plot(lam,flx_ws,'rs')
ax.plot(wr,flux_nu,'k')
ax.plot(mir_wave,mir_spec,'b')
#plt.errorbar(onr_lam,onr_flx,yerr=onr_unc,fmt='o',mec='skyblue',mfc='skyblue',ecolor='black',capsize=4.,capthick=1, label = 'Photometry')
ax.errorbar(lam,flx,yerr=unc,fmt='o',mec='skyblue',mfc='skyblue',ecolor='skyblue',capsize=4.,capthick=1, label = 'Photometry')
#ax.errorbar(lam,flx_ws,yerr=unc_ws,fmt='s',mec='red',mfc='red',ecolor='red',capsize=4.,capthick=1, label = 'Photometry')
#plt.plot(lam,f_sval,'x')
ax.errorbar(f_subw,f_sub,yerr=f_subu,fmt='^',mec='red',mfc='red',ecolor='red',capsize=4.,capthick=1, label = 'Photometry')

#plt.plot(f_subw,f_sub,'^')
#ax.errorbar(f_subw,f_sub,yerr=f_subu,fmt='^',mec='red',mfc='red',ecolor='red',capsize=4.,capthick=1, label = 'Photometry')

ax.plot(wr, temp2, 'k')
ax.plot(wr, warmbelt, 'r-.')
ax.plot(wr, coolbelt, 'c-.')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim([0.001,20])
ax.set_xlim([0.3,3000])
#plt.show()
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
ax.set_xlabel('Wavelength [$\mu$m]')
ax.set_ylabel('Flux Density [Jy]')
plt.savefig('SED_warm_cold_au-mic.eps')


print(flx_ws)
print(zed)



##Emcee Inputs##
nwalkers = 20
niter = 100
initial = np.array([10]) #Variables to be tested: sm,dfrac,qv,rm,rw
ndim = len(initial)
#p0 = [np.array(initial) + np.array([random.uniform(-0.5,0.5),random.uniform(-0.01,0.01),random.uniform(-0.5,0.5),random.uniform(-5,5),random.uniform(-95,10),] for i in range(nwalkers)]
p0 = [np.array(initial) + np.array([random.uniform(-1,1)],) for i in range(nwalkers)]
print('Walkers for simulation:')
print(p0)
#print(zed)
data = onr_lam,onr_flx,onr_unc


def main(p0,nwalkers,niter,ndim,lnprob,data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state
    
sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)

samples = sampler.flatchain

#plt.show()

labels = ['Amp']
fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84])
cornerfig = object+'_StellarAtmFit'+'_Emcee_corner_nwalkers'+str(nwalkers)+'_niter'+str(niter)+'.eps'
#fig.savefig('Emcee_corner_30_500.eps')
fig.savefig(cornerfig)

sm_amp = np.median(samples, axis=0)

print(f'Amp: {sm_amp}')

flux_nu_s = sm_amp*flux_nu

col_format = "{:<5}" * 2 + "\n"   # 2 left-justfied columns with 5 character width

with open("au-mic_spek.csv", 'w') as of:
    for x in zip(wr, flux_nu_s):
        of.write(col_format.format(*x))


Chi2_v = Chi2(onr_lam,onr_flx,onr_unc,wr,flux_nu_s)

print(f'Chi2: {Chi2_v}')

plt.clf()
plt.plot(wr,flux_nu_s)
plt.errorbar(onr_lam,onr_flx,yerr=onr_unc,fmt='o',mec='skyblue',mfc='skyblue',ecolor='black',capsize=4.,capthick=1, label = 'Photometry')
#plt.plot(wr,flux_sa,'k')
plt.xscale('log')
plt.yscale('log')
plt.ylim([0.001,20])
plt.xlim([0.3,3000])
plt.show()
#print(zed)



t1 = time.time()
t = round((t1 - t0)/60,4)
print(f'Total Time: {t} minutes')



