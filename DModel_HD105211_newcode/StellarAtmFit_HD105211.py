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
    flam = amp*fun_spek_nu(phot_wav)
    
#    #print(flam)
    c2 = 0
    for i in range(len(phot_wav)):
        c2 = c2 + (((phot_flux[i]-flam[i]))/phot_unc[i])**2
#    lp = -0.5*np.sum( (phot_flux - flam) /phot_unc **2)
    chi2 = -0.5*c2
    print(chi2)
    return chi2

def lnprior(theta):
    amp = theta
    if 0.1 < amp < 200:
        return 0.0
    else:
        return -np.inf


def lnprob(theta,wav,flx,unc):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,wav,flx,unc)


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
#WRange = [1,10,100,1000] 0000000000000000000a0f000000000000000000#Add specific wavelengths of interest???
#wr2 = np.array(WRange)
#wr3 = np.append(wr1,wr2)
#wr = np.sort(wr3)

#-----------------------------------------------------------------#
## Host Star Properties
object = 'HD 105211'
#Stellar Photometric values for comparison
#onr_lam = [0.349,0.411,0.440,0.466,0.546,0.55,0.64,0.79,1.26,1.6,2.2200]
#onr_flx = [24.02,62.4,69.91,72.7,69.91,79.99,92.2,88.4,59.16,45.85,29.78]
#onr_unc = [0.24,0.62,6.45,0.77,6.45,7.38,0.46,0.4,7.65,5.93,1.92]
onr_lam = [0.64,0.79,1.235,1.662,2.159]
onr_flx = [92.2,88.4,66.7,52.3,35.1]
onr_unc = [0.46,0.4,18.4,12.5,8.5]
#onr_lam = [0.349,0.411,0.440,0.466,0.546,0.55,0.64,0.79,1.235,1.662,2.159]
#onr_flx = [24.02,62.4,69.91,72.7,69.91,79.99,92.2,88.4,66.7,52.3,35.1]
#onr_unc = [0.24,0.62,6.45,0.77,6.45,7.38,0.46,0.4,18.4,12.5,8.5]

#HD 105211
irs_spec = ascii.read('CASHD10511.txt', delimiter = ',')
irs_spec = np.loadtxt('CASHD10511.txt', dtype=[('wavelength', float), ('flux', float), ('error', float)])

##Host Star properties in Solar units##
Ls = 6.6          #Luminosity of Star [Solar Luminosity]
Ms = 1.63        #Mass of Star [Solar Masses]
Rs = 1.66          #Radius of Star [Solar Radius]
Ts = 7244       #star temperature [K]
dpc = 19.76        #distance to star system [pc]

##Star Flux values: Blackbody + Spectrum (if available)
##Blackbody Stellar spectrum (default)
bb = BBody(temperature=Ts*u.K)
d = dpc*pc   #distance to Star [m]
Rad = Rs*R_s   #Star Radius: Sun [m]
As = pi*(Rad/d)**2 #Amplitude of Star's blackbody planck fn (i.e. for this case: cross-section of Sun) and convert to Jansky (as BBody units are given in: erg / (cm2 Hz s sr))
flux_s = As*Blam(Ts,wr*10**-6)*10**26*((wr*10**-6)**2)/(c)


##Star Flux values: Blackbody + Spectrum (if available)
##Blackbody Stellar spectrum (default)
#bb = BBody(temperature=Ts*u.K)
#d = dpc*pc   #distance to Star [m]
#Rad = Rs*R_s   #Star Radius: Sun [m]
#As = pi*(Rad/d)**2 #Amplitude of Star's blackbody planck fn (i.e. for this case: cross-section of Sun) and convert to Jansky (as BBody units are given in: erg / (cm2 Hz s sr))
#flux_s = As*Blam(Ts,wr*10**-6)*10**26*((wr*10**-6)**2)/(c)

##Stellar Spectrum##
#Import Model Atmosphere values
f_spek = np.loadtxt('BTGen7200.txt', dtype=[('wave', float),('fspek', float)])

#Convert, Extrapolate and Calibrate with Blackbody Spectrum of similar temperature
wave = f_spek["wave"]*0.0001 #Convert from Angstroms to Microns
fspek = f_spek["fspek"]
#Extrapolate/Interplate F(wave)
fun_spek = interpolate.interp1d(np.log10(wave),np.log10(fspek),kind = 'linear', fill_value = 'extrapolate')

#Convert flux (wave) to flux (frequency)
f_10spek = fun_spek(np.log10(wr))
flux_nu = (10**f_10spek)*((wr*10**-6)**2)/(c)  #unscaled flux (frequency)
flux_nu = flux_nu/np.max(flux_nu) #scale max to 1 Jy
fun_spek_nu = interpolate.interp1d(wr,flux_nu,kind = 'linear', fill_value = 'extrapolate') #interpolate unscaled flux


##Emcee Inputs##
nwalkers = 10
niter = 200
initial = np.array([1]) #Variables to be tested: 'Td_w', 'Td_c', 'Ad_w', 'Ad_c'
ndim = len(initial)
p0 = [np.array(initial) + np.array([random.uniform(-0.5,50)]) for i in range(nwalkers)]
#p0 = [np.array(initial) + np.array([random.uniform(-1,1)],) for i in range(nwalkers)]
print('Walkers for simulation:')
print(p0)
#print(zed)
data = onr_lam, onr_flx, onr_unc


def main(p0,nwalkers,niter,ndim,lnprob,data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100)
    burnin = sampler.get_chain()
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state, burnin

sampler, pos, prob, state, burnin = main(p0,nwalkers,niter,ndim,lnprob,data)

samples = sampler.flatchain

#plt.show()

labels = ['Amp']
fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84])
cornerfig = object+'_StellarModel'+'_Emcee_corner_nwalkers'+str(nwalkers)+'_niter'+str(niter)+'.eps'
fig.savefig(cornerfig)

Amp_mcmc = np.median(samples, axis=0)


#onr_lam = [0.349,0.411,0.440,0.466,0.546,0.55,0.64,0.79,1.26,1.6,2.2200]
#onr_flx = [24.02,62.4,69.91,72.7,69.91,79.99,92.2,88.4,59.16,45.85,29.78]
#onr_unc = [0.24,0.62,6.45,0.77,6.45,7.38,0.46,0.4,7.65,5.93,1.92]
onr_lam = [0.349,0.411,0.440,0.466,0.546,0.55,0.64,0.79,1.235,1.662,2.159]
onr_flx = [24.02,62.4,69.91,72.7,69.91,79.99,92.2,88.4,66.7,52.3,35.1]
onr_unc = [0.24,0.62,6.45,0.77,6.45,7.38,0.46,0.4,18.4,12.5,8.5]
#onr_lam = [0.64,0.79,1.235,1.662,2.159]
#onr_flx = [92.2,88.4,66.7,52.3,35.1]
#onr_unc = [0.46,0.4,18.4,12.5,8.5]



print(f'Amp: {Amp_mcmc}')
print(f'Max Flux_nu: {np.max(flux_nu)}')
plt.clf()
fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.plot(wr,Amp_mcmc*flux_nu,'k')
#ax.plot(onr_lam,onr_flx, 'o')
ax.errorbar(onr_lam,onr_flx,yerr=onr_unc,fmt='o',mec='green',mfc='green',ecolor='black',capsize=4.,capthick=1, label = 'Optical',zorder=4)
ax.set_xlabel('Wavelength [$\mu$m]')
ax.set_ylabel('Flux Density [Jy]')
ax.set_xlim([0.1, 10000])
ax.set_ylim([10**-3, 200])
ax.set_xscale('log')
ax.set_yscale('log')
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.show()

#Plot chains
plt.clf()
fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
#s2 = burnin.append(samples)
s2 = np.concatenate((burnin, samples), axis=0)

for i in range(ndim):
    ax = axes[i]
    ax.plot(s2[:, :, i], "k", alpha=0.3)
    ax.plot(burnin[:, :, i], "r", alpha=0.3)  #plot burn-in independently
    ax.set_xlim(0, len(burnin) + len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

plt.show()

#print(zed)0000000



t1 = time.time()
t = round((t1 - t0)/60,4)
print(f'Total Time: {t} minutes')



