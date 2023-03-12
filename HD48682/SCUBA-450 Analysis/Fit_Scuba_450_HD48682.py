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
from matplotlib import image
import random
import numpy as np
import math
import subprocess
import os
from os import path
import time
from skimage.transform import rescale
#from skimage.misc import imread
from astropy.io import fits
from astropy.io import ascii
from astropy.convolution import CustomKernel
from astropy.convolution import convolve
from astropy.modeling.models import BlackBody as BBody
from astropy import units as u
from PyAstronomy.pyasl import planck
#from scipy.misc import imread
from skimage import io
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
from numpy import*
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
def lnlike(theta,realimage):
#    sm,dfrac,qv,rm,rw = theta
#    func_flux = DustyMM(sm,smax_r,dfrac,qv,rm,rw,rin,rout)
    x1,y1,A1,sig1x,sig1y,x2,y2,A2,sig2x,sig2y = theta
    zz = twoansaeGauss(x1,y1,A1,sig1x,sig1y,x2,y2,A2,sig2x,sig2y,X)
    
       
    residual = SCUBA450 - zz
    chi2 = -0.5*np.sum(residual)
    
#    print(chi2)
    
    
    return chi2

def lnprior(theta):
    x1,y1,A1,sig1x,sig1y,x2,y2,A2,sig2x,sig2y = theta
    if -50 < x1 < -30 and 10 < y1 < 30 and 0 < A1 < 1 and 0.01 < sig1x < 20 and 0.01 < sig1y < 20 and 50 < x2 < 70 and -25 < y2 < -15 and 0 < A2 < 1 and 0.01 < sig2x < 20 and 0.01 < sig2y < 20:
        return 0.0
    else:
        return -np.inf


def lnprob(theta,realimage):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,realimage)


#-----------------------------------------------------------------#
def twoansaeGauss(x1,y1,A1,sig1x,sig1y,x2,y2,A2,sig2x,sig2y,X): #produce two ansae as two 2-D Gaussian
#Produce 2 ansae of 2-D Gaussian nature for Image size (X,X) with origin in center of Image
#Use a meshgrid???
    nc = int((X-1)/2)
    x = np.arange(X) - nc
    y = np.arange(X) - nc
    xx,yy = np.meshgrid(x, y)

    return A1*np.exp(-( (xx-x1)**2/(2*sig1x**2) +  (yy-y1)**2/(2*sig1y**2) ) ) + A2*np.exp(-( (xx-x2)**2/(2*sig2x**2) +  (yy-y2)**2/(2*sig2y**2) ))
#    return A1*np.exp(-(xx))

#SCUBA450 = image.imread("hd48682-450v3.png",image.IMREAD_GRAYSCALE)
SCUBA450 = io.imread('hd48682-450v3.png', as_gray=True)
X = len(SCUBA450)

SCUBA450[SCUBA450 < 0.52] = 0

#plt.hist(SCUBA450[:])
#plt.show()
#print(zed)

##for j in SCUBA450:
###    SCUBA450_new = asarray([[i[0],i[0]] for i in j])
##
##
#plt.imshow(SCUBA450)
#plt.show()
#    ##print(len(SCUBA450))
#print(zed)
##
##print(SCUBA450)
#
#zz = twoansaeGauss(-37,17,1,10,10,61,-21.2,1,10,10,len(SCUBA450))
#
#print(zz)
#
#residual = SCUBA450 - zz
#
#print(zz.shape)
#
#
#X = len(SCUBA450)
#nc = int((X-1)/2)
#x = np.arange(X) - nc
#y = np.arange(X) - nc
#xx,yy = np.meshgrid(x, y)
#
#h = plt.contourf(xx,yy,residual)
##plt.axis('scaled')
#plt.colorbar()
#plt.imshow(zz)
#plt.show()
#
#plt.imshow(SCUBA450)
#plt.show()
#print(zd)
#
##print(np.sum(residual))
#print(np.max(SCUBA450))
#print(np.max(zz))
#plt.show()

#
#
#print(zed)

#return 2-D Array f(x,y)

#-----------------------------------------------------------------#
## Host Star Properties
object = 'HD 48682'



##Emcee Inputs##
nwalkers = 50
niter = 1000
initial = np.array([-37,17,0.5,5,5,61,-21,0.5,5,5]) #Variables to be tested: x1,y1,A1,sig1x,sig1y,x2,y2,A2,sig2x,sig2y
ndim = len(initial)
p0 = [np.array(initial) + np.array([random.uniform(-5,5),random.uniform(-5,5),random.uniform(-0.5,0.4),random.uniform(-2.5,2.5),random.uniform(-2.5,2.5),random.uniform(-5,5),random.uniform(-5,5),random.uniform(-0.5,0.4),random.uniform(-2.5,2.5),random.uniform(-2.5,2.5)],) for i in range(nwalkers)]
#p0 = [np.array(initial) + np.array([random.uniform(-1,1)],) for i in range(nwalkers)]
print('Walkers for simulation:')
print(p0)
#print(zed)
data = [1]


def main(p0,nwalkers,niter,ndim,lnprob,data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
    
    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 1000)
    burnin = sampler.get_chain()
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state, burnin

sampler, pos, prob, state, burnin = main(p0,nwalkers,niter,ndim,lnprob,data)

samples = sampler.flatchain

#plt.show()

labels = ['x1','y1','A1','sig1x','sig1y','x2','y2','A2','sig2x','sig2y']
fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84])
cornerfig = object+'_2DGaussFit'+'_Emcee_corner_nwalkers'+str(nwalkers)+'_niter'+str(niter)+'.eps'
fig.savefig(cornerfig)

x1,y1,A1,sig1x,sig1y,x2,y2,A2,sig2x,sig2y = np.median(samples, axis=0)



#Plot chains
plt.clf()
fig, axes = plt.subplots(10, figsize=(10, 7*10), sharex=True)
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


axes[-1].set_xlabel("step number");

chains = object+'_2DGaussFit_chains_nwalkers'+str(nwalkers)+'_niter'+str(niter)+'.pdf'
fig.savefig(chains)

Model2D = twoansaeGauss(x1,y1,A1,sig1x,sig1y,x2,y2,A2,sig2x,sig2y,X)


plt.clf()
fig, axes = plt.subplots(1, figsize=(10, 7), sharex=True)
plt.imshow(Model2D)
plt.savefig('2DGauss.pdf')

print(f'A1: {A1}, A2: {A2}.  Max. Value of 2D Gauss model: {np.max(Model2D)}')

plt.clf()
plt.imshow(Model2D-SCUBA450)
plt.savefig('Residual.pdf')

print(np.sum(Model2D-SCUBA450))

#Pixel to arcsec/au
#570 pixel = 60 arsec
#1 arscec = 16.65au
#1 pixel = (60/570)
pconv = (60/570)*16.65

rmean = pconv*(np.sqrt((x2-x1)**2+(y2-y1)**2))/2

sigmean = pconv*(sig1x+sig1y + sig2x + sig2y)/4

print(f'radial distance approx. : {rmean}')
print(f'sigma width: {sigmean}')


t1 = time.time()
t = round((t1 - t0)/60,4)
print(f'Total Time: {t} minutes')



