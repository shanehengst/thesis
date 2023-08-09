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
import matplotlib
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
#from skimage.transform import rescale
from astropy.io import fits
from astropy.io import ascii
from astropy.convolution import CustomKernel
from astropy.convolution import convolve
from astropy.modeling.models import BlackBody as BBody
from astropy import units as u
from PyAstronomy.pyasl import planck
#from scipy.ndimage.interpolation import rotate
#from scipy.ndimage.interpolation import shift
#from photutils.centroids import fit_2dgaussian
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
import matplotlib.colors as colors
import copy #https://stackoverflow.com/questions/9455044/problems-with-zeros-in-matplotlib-colors-lognorm
import h5py
from tqdm import tqdm
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
##Function: Orbit
def orbit(a,b,e,P,dt):
    #input values:
    #a: semi-major distance [au]
    #b: semi-minor distance [au]
    #e: eccentricity [-] 0 < e < 1
    #P: Orbital Period [years]
    #dt: timestep [years]
    #Option to include: no: number of orbits (assumes grain is released from perihelion distance)
    
    tht = [] #new angle value as function of t
    rd = [] #empty array for distance values
    N = math.ceil(P/dt)#*(P/a) #number of timesteps
    #print(f'Number of Timesteps: {N}')
    ti = np.linspace(0,P,N)  #time step values
    th = 0 #inital theta angle
    
    for t in ti:
   
        rth = (a*(1-e**2)/(1+e*math.cos(th)))#/(au/100) #distance between orbiting bodies (r) in au
        thn = th + (a*b*2*pi)/(P*rth**2)*dt  #iterative step to determine next angle as a function of time

        rd.append(rth)   #append radial component
        tht.append(thn) #append angle component
        th = thn  #new angle for next iteration
    
    return [rd,tht,ti] #rd: radial distance(au), tht: angle values (radians), ti: time [years]
#-----------------------------------------------------------------#
#Function: Bolometric Black Body function as a function of wavelength
def Blam(T,w):
  
    #Inputs:
    #T: Temperature (float) [Kelvin]
    #w: Wavelength range (array float) [metres]
    L = (2*h*c**2/(w**5))*(np.exp(h*c/(w*kb*T))-1)**-1
    
    #Ouput
    #Luminosity as a function of wavelength [SI Units]
    
    return L
#-----------------------------------------------------------------#

##Function: probability function for orbit of grain with collisions
def prob_grain_rp_p(be,rpi,t):
    #input values:
    #be: beta value [-]
    #rpi: initial periastron distance [au]
    #t: time since initial birth ring of grains[years]
    #Pr: Planet's orbital radius - if present. If not Pr = 0
    #returns
    #g_prob: probability of finding beta sized grains at time t in system

    e = be/(1-be)            #eccentricity [-] 0 < e < 1 of the grain
    ai = rpi/(1-e)            #initial semi major axis of the grain
    ts = 400*ai**2/(Ms*be)           #time taken to hit star for the grain
    rai = ai*(1+e)          #initial apostron distance
    tcoll = ai**(3/2)*Ms**(-1/2)/(4*pi*tau)  #approx. collisional time
    
    
    
    #    print(rpi)
#    print(rai)
    
    
    ##after time t##
    afs = ai**2 - be*t*Ms/400    #square of final semi major axis
    if afs > 0:
        rpf = np.sqrt(afs)*(1-e)   #final periastron distance [au]
        fg = t #/ts                     #increase amount of grains
#        print(fg)
#        print(be)
    else:
        
        fg = ts #1                   #reaches maximum number of grains (i.e. continuously supply and destruction of beta sized grains) but relative number will be less than other beta values
        rpf = 1                    #new periastron distance for grains remaining
        
#    print(f'beta {be}, s {s}, rp {rpi}, ra {rai}, rpf {rpf}, afs {afs}')

    
#    n_rp = np.int64(np.ceil((t/10e4)))  #number of steps between rpi and rpf for consistency
    n_rp = 100
    rp_space = np.linspace(rpi,rpf,n_rp) #periastron space for grain
    
#    print(rp_space)
#    t_space = np.linspace(0,t,n_rp)
#    print(rp_space)
#    print(n_rp)
#    print(zed)
 
    g_time = np.zeros(len(rbins_bm))

    for rp in rp_space:
    
        af = rp/(1-e)               #semi-major value corresponding to rp value
        bf = af*math.sqrt(1-e**2)   #semi-minor value corresponding to af value
        P = af**(3/2)*Ms**(-0.5)               #Period of grain for one orbit [years]
        ra = 2*af-rp                #apoastron distance for grain [au]
        trp = (400/(Ms*be))*(ai**2-af**2)          #time taken to reach rp

#        print(f'tcoll: {tcoll}, trp: {trp}')
        if trp < tcoll:  #if the PR time is greater than the collisional time, grain has survived.
            #Find inner values dependent on where rp lies | r_in: inner most bin value, r_min: actual rp
            if rp % math.ceil(rp) == 0:
                r_in = rp
                r_min = rp
            else:
                r_in = math.ceil(rp)
                r_min = rp

            #find outer values dependent on where ra lies | r_out: outer most bin value, r_out: actual ra
            if ra > hd_max:
                #The ra value is beyond the scope of the model, therefore maximum radial value is hd_max
                r_out = hd_max
                r_max = hd_max
                
                #defining radial values between r_in and r_out
                r_steps = np.int64((r_out-r_in)/hd_bs + 1)
                rad = np.linspace(r_in, r_out,r_steps)

                #adding true r_min
                if r_in != rp:
                    rad = np.insert(rad,0,r_min)

            else:
                r_out = math.floor(ra)
                r_max = ra

                #defining radial values between r_in and r_out
                r_steps = np.int64((r_out-r_in)/hd_bs + 1)
                rad = np.linspace(r_in, r_out,r_steps)

                #adding true r_max
                if r_out != r_max:
                    rad = np.append(rad,r_max)
                
                #adding true r_min
                if r_in != rp:
                    rad = np.insert(rad,0,r_min)

            #determine angles th1 and th2 at radial values
            th1 = np.arccos(np.round( ( af*(1-e**2)/rad[:-1] - 1 )/e , 8))
            th2 = np.arccos(np.round( ( af*(1-e**2)/rad[1:] - 1 )/e , 8))
            #find difference between th1 and th2
            dth = (th2 - th1)

            #find the time taken from from th1 to th2
            dt_i = np.multiply(dth,P*rad[:-1]**2/(af*bf*2*pi))
      
            
            #normalise, i.e. find the probability of finding the grain
    #        dt_ip = dt_i/np.sum(dt_i)
            
            #fill remainding radial values in spatial model with zeros
            inn_o = np.zeros(np.int64(np.floor(rp)))
            out_o = np.zeros(np.int64(hd_max - np.ceil(r_max)))
            g_time_r = np.concatenate((inn_o,dt_i,out_o), axis = None)
            
            g_time = g_time + g_time_r #add to previous iteration
    

    g_prob = fg*g_time/np.sum(g_time)


    
    return g_prob #, dt_f #ra, rp #g_prob: probablity of finding grain at a specific location | dt: time taken to traverse the bin at given location


#-----------------------------------------------------------------#
#Dust Migration model - FAST Method
#Revamping the original code to remove the reliance on dataframes
#Simply return a SED flux
def DustyMM(smin,smax,dfrac,q,rm,rw,rin,rout):
    ##Planestimal Belt Characteristics##
    print(f'Min. Grain Size: {smin}, dfrac: {dfrac}, q: {q}, rmean: {rm}, rsigma: {rw}')
    ##Total Grain Mass##
    DiscMass = dfrac*Me*10**3 #Total grain mass as a fraction of Earth Mass (converted to grams) because denisty(rho) is given in g/cm^3
#    print(f'Mass of Disc: {DiscMass} g') #inform human

#    Number of grain sizes
    no_s = 100

#    create grain size space where most grains are towards the smaller size (logarithmic spaced in linear fashion)
    s_gs = np.geomspace(smin,smax,no_s)
    print(f'Number of Grain Sizes: {len(s_gs)} ranging from {s_gs[0]} to {s_gs[-1]} microns')

    gr_blowout = [0]
    grainsizes = []

    #Initiate variables
    dM_radial_nb = 0   #Mass of non-beta grains
    flux_nb = np.zeros(len(wr))        #flux of non-beta grains
 
    dM_radial_b = 0   #mass of beta grains
    flux_b = np.zeros(len(wr))         #flux of beta grains


    ##grain size distribution dN \propto s^-q ds #Power law mass distribution + pre-fill dataframes for later summing of angle release
    ##Determine Constant to find exact number and mass of grains according to size##
    #print(f'Calculating Mass proportional constant...')
    dMs = (s_gs*1e-4)**-q*rho*(4/3)*pi*(s_gs*1e-4)**3 #note: last power used to be a '3' or is it suppose to be '3'
    Msum = np.sum(dMs)      #unscaled mass component
    dNc = DiscMass/Msum    #finding constant
    
    #Initial array for
    #dNsr = np.zeros(len(rbins_bm))
    
    #print(f'Proportional Constant for dN proto s**-q: {dNc} where grain size (s) in units of cm')

    for s in s_gs:
        

        flux_nb = 0

        #Beta-grains model
        #Determine beta-value | considering radiation pressure on grains
        be = 0.574*Ls*Qpr_s(s)/(Ms*rho*s) #beta-value in solar units.  Density of grain in g/cm^3
         
        if be == 0.5:
            #blow out limit
            print(f'Blowout grain size: {a}')
            #ignores grain
        if be > 0.5:
            print(f'Grains with size: {s} micron & beta: {be} are on unbound orbits are ignored, i.e. blown out of system (be > 0.5)')
            if gr_blowout[-1] != s:
                gr_blowout.append(s)
        if be < 0:
            print(f'something is screwed (be < 0), beta = {be}, Ls = {Ls}, Qpr = {Qpr_s(s)}.  Exiting...')
            print(zed)
        elif be >= 0 and be < 0.5:
            #Bound grains
            #Orbital values for 0 <= be < 0.5
            #print(f'Grain size: {a} is on a bound orbit with be: {be} (0 < be < 0.5)')
            
            dNb = f_sd(s,rbins_bm)
            dNb = dNb.flatten()
            dNb = (dNb/np.sum(dNb))*(dNc*(s*1e-4)**-q)
#            dNb = (dNb)*(dNc*(s*1e-4)**-q)
#            dNb = dNb/np.sum(dNb)
          
#            dNb_apert = dNb[0:rmax]
#            dNbr = np.trim_zeros(dNb_apert)  #remove zeroes from array
            dNbr = np.trim_zeros(dNb)  #remove zeroes from array
       
            #Summing mass values
            dM_radial_b = dM_radial_b + np.sum(dNb*rho*(4/3)*pi*(s*1e-4)**3)
            
#            print('test 1')
            
#            dNbf = dNb.flatten()
            glocID = dNb/dNb                       #Mask for grain locations
            glocID[np.isnan(glocID)] = 0            #Remove NaN
            
#            print(glocID)
#            print(glocID*rbins_bm)
            
#            print('test 2')
            
            gloc = np.trim_zeros(rbins_bm*glocID)   #radial locations where grains of size (s) are present
            
#            print(gloc)
#
#            print('test 3')
#            print(zed)
            
            GTg = Tg_sr(s,gloc).flatten()           #Grain temperatures as a function of radial locations
            BBfns = f_BBgT(GTg,wr)
            
            #BBfns = np.multiply(np.multiply(BBfns,Qabs_sw(s,wr)),np.multiply(dNb,dNb))    #multiply blackbody fns by corresponding number density * radial distance
            BBfns = np.multiply(np.multiply(BBfns,Qabs_sw(s,wr)),np.multiply(dNbr[::-1],gloc[::-1]))    #multiply blackbody fns by corresponding number density * radial distance
                            #BBfns = np.multiply(BBfns,dNbr)    #multiply blackbody fns by corresponding number density * radial distance
            BBfns_sum = np.sum(BBfns, axis = 1)     #sum all BB functions together
            #instead of summing
            
           
            #Thermal Emission
            Flam = BBfns_sum*8*pi**2*(s*10**-6)**3*(dpc*pc)**-2/hd_bs
      
#                #Scattered light
#                Alam = Qsca_sw(s,1:)/(Qsca_sw(s,1:)+Qabs_sw(s,1:)) #Albedo for scattered light
#                Flsca = dNb*((Rs*R_s)/(dpc*pc))**2*pi*Blam(Ts,wr*10**-6)*Alam*Qsca_sw(s,1:)*(s*10**-6/(2*rg*au))**2 #Scattered Emission
#
            #Summing Fluxes
#                Flamtot = Flam + Flsca #Total flux as a function of wavelength
#                Fnu = 10**26*Flamtot*((wr*10**-6)**2)/(c) #Total flux as a function of frequency [Jankys]
            Fnu = 10**26*Flam*((wr*10**-6)**2)/(c) #Total flux as a function of frequency [Jankys] for a specific sized grain (s)


#
            flux_b = flux_b + Fnu #Add flux
    
    #Summing values
    #Grav only
#    #Mass
#    print(f'Total Disc Mass | Grav Only Model (dmSum): {dM_radial_nb} g ')
#    MPer = round((dM_radial_nb/DiscMass)*100,3)
#    #print(f'Fraction of Mass for dM values (all au): {MPer}%')
#
#    #Beta model
#    #Mass
#    print(f'Total Disc Mass | Beta Model (dmSum): {dM_radial_b} g ')
    MPerb = round((dM_radial_b/DiscMass)*100,3)
#    #print(f'Fraction of Mass for dM values (all au): {MPerb}%')
#
    SED_total = np.add(flux_b,flux_sa)
##    SED_totalnb = np.add(flux_nb,flux_sa)

    
    return SED_total,flux_b,MPerb #returns SED Total for given set of initial conditions
#    return SED_total
#-----------------------------------------------------------------#
#Slower method for extracting flux/distance values as a function of size
def DustyMM_ext(smin,smax,dfrac,q,rm,rw,rin,rout,filter):
    ##Planestimal Belt Characteristics##
    print(f'Min. Grain Size: {smin}, dfrac: {dfrac}, q: {q}, rmean: {rm}, rsigma: {rw}')
    ##Total Grain Mass##
    DiscMass = dfrac*Me*10**3 #Total grain mass as a fraction of Earth Mass (converted to grams) because denisty(rho) is given in g/cm^3
    
#    df_rbins_pg = pd.DataFrame({'R': rbins_bm}) #probability of grains of size (s) to each bin: (p)
#    df_rbins_dNg = pd.DataFrame({'R': rbins_bm}) #number of grains of size (s) for a given a power law exponent (q) in each bin: (dN)
#    df_rbins_dMg = pd.DataFrame({'R': rbins_bm}) #mass of grains of size (s) in each bin: (dM)
#
#    #Dataframe for flux values corresponding to each model
#    df_fluxsmbg = pd.DataFrame({'Wavelength (um)': wr}) #Wavelength values
#    df_fluxsmbg['Frequency (Hz)'] = c/(df_fluxsmbg['Wavelength (um)']*1e-6)#*2.99792458e+14 |frequency values
##    print(f'Mass of Disc: {DiscMass} g') #inform human

#    Number of grain sizes
    no_s = 100

#    create grain size space where most grains are towards the smaller size (logarithmic spaced in linear fashion)
    s_gs = np.geomspace(smin,smax,no_s)
    print(f'Number of Grain Sizes: {len(s_gs)} ranging from {s_gs[0]} to {s_gs[-1]} microns')

    gr_blowout = [0]
    grainsizes = []

    #Initiate variables
    dM_radial_nb = 0   #Mass of non-beta grains
    flux_nb = np.zeros(len(wr))        #flux of non-beta grains
 
    dM_radial_b = 0   #mass of beta grains
    flux_br = np.zeros(len(rbins_bm))         #flux of beta grain


    ##grain size distribution dN \propto s^-q ds #Power law mass distribution + pre-fill dataframes for later summing of angle release
    ##Determine Constant to find exact number and mass of grains according to size##
    #print(f'Calculating Mass proportional constant...')
    dMs = (s_gs*1e-4)**-q*rho*(4/3)*pi*(s_gs*1e-4)**3
    Msum = np.sum(dMs)      #unscaled mass component
    dNc = DiscMass/Msum    #finding constant
    
    #Initial array for
    #dNsr = np.zeros(len(rbins_bm))
    
    #print(f'Proportional Constant for dN proto s**-q: {dNc} where grain size (s) in units of cm')

    for s in s_gs:
        s_colname = str(round(s,4))
        
#        df_rbins_pg[s_colname] = 0
#        df_rbins_dNg[s_colname] = 0
#        df_rbins_dMg[s_colname] = 0

#        flux_nb = 0

        #Beta-grains model
        #Determine beta-value | considering radiation pressure on grains
        be = 0.574*Ls*Qpr_s(s)/(Ms*rho*s) #beta-value in solar units.  Density of grain in g/cm^3
         
        if be == 0.5:
            #blow out limit
            print(f'Blowout grain size: {a}')
            #ignores grain
        if be > 0.5:
            print(f'Grains with size: {s} micron & beta: {be} are on unbound orbits are ignored, i.e. blown out of system (be > 0.5)')
            if gr_blowout[-1] != s:
                gr_blowout.append(s)
        if be < 0:
            print(f'something is screwed (be < 0), beta = {be}, Ls = {Ls}, Qpr = {Qpr_s(s)}.  Exiting...')
            print(zed)
        elif be >= 0 and be < 0.5:
            #Bound grains
            #Orbital values for 0 <= be < 0.5
            #print(f'Grain size: {a} is on a bound orbit with be: {be} (0 < be < 0.5)')
            
            dNb_sums = f_sd(s,rbins_bm)
            dNb_sum = (np.sum(dNb_sums))  #scaling the probability relative to itself --- need to check

            
            dNb = f_sd(s,rbins_bm).flatten()*dNc*((s*1e-4)**-q)/dNb_sum  #number of grains at location r with size s
            Nrs = dNb*(s*1e-4)*rbins_bm/hd_bs  #Grain number density

            dMb = dNb*rho*(4/3)*pi*(s*1e-4)**3 #mass of grains at location r with size s (to be used later)
  
            
            GTg = Tg_sr(s,rbins_bm).flatten() #grain temperature at location r
            BBfns = f_BBgT(GTg,wr) #black body function for grain of size s at location r with the temperature GTg


            BBfns_dNb = 8*pi**2*(s*10**-6)**2*(dpc*pc)**-2*np.multiply(Nrs,BBfns)  #Blackbody function * Nrs * constants (see Hengst et al. 2023)
#            print(Nrs)
#            print(BBfns)
            
            F_tw_mod = np.multiply(BBfns_dNb,Qabs_sw(s,wr))  #modified black blody due to Qabs see Hengst et al. 2023 to produce thermal emission as a function of wavelength: Flux(wave)
    
            F_tw_filt = np.multiply(F_tw_mod,np.vstack(filter)) #filter flux(wave) according to filter profile
            
            
            
            F_tnu = 10**26*F_tw_filt*((np.vstack(wr)*10**-6)**2)/(c)  #convert from Flux(wave) to Flux(frequency)
                        
            F_tnu_rs = np.sum(F_tnu,axis = 0)  #sum fluxes as a function location
            
#            plt.plot(wr,filter,'b')
#            fig, ax1 = plt.subplots()
#            ax2 = ax1.twinx()
##            ax1.plot(wr,F_tw_mod)
#            ax1.plot(wr,np.sum(F_tw_filt, axis = 1),'k')
#            ax1.plot(wr,np.sum(F_tw_mod, axis = 1), 'c')
#
#            ax2.plot(wr,np.sum(F_tnu, axis = 1), 'r')
##            ax2.plot(wr,filter, 'r')
#
##            ax2.plot(wr,np.sum(F_tw_filt, axis = 1),'r')
##            ax1.set_ylabel('F_wave', color = 'k')
#            ax2.set_ylabel('F_nu', color = 'r')
#            ax1.set_xlim([10,1000])
#            plt.xscale('log')
#            ax1.xaxis.set_major_formatter(formatter)
#            plt.show()
#            print(zed)
            
            
      
            flux_br = flux_br + F_tnu_rs
            
            
#            dNb = f_sd(s,rbins_bm)
#            dNb = dNb.flatten()
#            dNb = (dNb/np.sum(dNb))*(dNc*(s*1e-4)**-q)
#            df_rbins_dNg[s_colname] = dNb
#            df_rbins_dMg[s_colname] = dNb*rho*(4/3)*pi*(s*1e-4)**3
#
            
#            dNb = dNb/np.sum(dNb)
          
#            dNb_apert = dNb[0:rmax]
#            dNbr = np.trim_zeros(dNb_apert)  #remove zeroes from array
#            dNbr = np.trim_zeros(dNb)  #remove zeroes from array
#
#            #Summing mass values
#            dM_radial_b = dM_radial_b + np.sum(dNb*rho*(4/3)*pi*(s*1e-4)**3)
#
##            print('test 1')
#
##            dNbf = dNb.flatten()
#            glocID = dNb/dNb                       #Mask for grain locations
#            glocID[np.isnan(glocID)] = 0            #Remove NaN
#
##            print(glocID)
##            print(glocID*rbins_bm)
#
##            print('test 2')
#
#            gloc = np.trim_zeros(rbins_bm*glocID)   #radial locations where grains of size (s) are present
##
##            print(gloc)
##
##            print('test 3')
##            print(zed)
#
#            GTg = Tg_sr(s,gloc).flatten()           #Grain temperatures as a function of radial locations
#            BBfns = f_BBgT(GTg,wr)
#
#            #BBfns = np.multiply(np.multiply(BBfns,Qabs_sw(s,wr)),np.multiply(dNbr,gloc))    #multiply blackbody fns by corresponding number density * radial distance
#            BBfns = np.multiply(np.multiply(BBfns,Qabs_sw(s,wr)),np.multiply(dNbr[::-1],gloc[::-1]))    #multiply blackbody fns by corresponding number density * radial distance
#                            #BBfns = np.multiply(BBfns,dNbr)    #multiply blackbody fns by corresponding number density * radial distance
##            BBfns_sum = np.sum(BBfns, axis = 1)     #sum all BB functions together
##
##            print(BBfns)
##            print(BBfns.shape)
##            print(zed)
#
#            for i in range(len(gloc)):
#                fcolname = 'Flux_'+str(round(s,3))+'_'+str(gloc[i])
#
#
#            #Thermal Emission
#                Flam = BBfns[:,i]*8*pi**2*(s*10**-6)**3*(dpc*pc)**-2/hd_bs
#
##                #Scattered light
##                Alam = Qsca_sw(s,1:)/(Qsca_sw(s,1:)+Qabs_sw(s,1:)) #Albedo for scattered light
##                Flsca = dNb*((Rs*R_s)/(dpc*pc))**2*pi*Blam(Ts,wr*10**-6)*Alam*Qsca_sw(s,1:)*(s*10**-6/(2*rg*au))**2 #Scattered Emission
##
#            #Summing Fluxes
##                Flamtot = Flam + Flsca #Total flux as a function of wavelength
##                Fnu = 10**26*Flamtot*((wr*10**-6)**2)/(c) #Total flux as a function of frequency [Jankys]
#                Fnu = 10**26*Flam*((wr*10**-6)**2)/(c) #Total flux as a function of frequency [Jankys] for a specific sized grain (s)
#
#
#            #flux_b = flux_b + Fnu #Add flux
#
#                df_fluxsmbg[fcolname] = Fnu
#
#
#
#    #Summing values
    #Grav only
#    #Mass
#    print(f'Total Disc Mass | Grav Only Model (dmSum): {dM_radial_nb} g ')
#    MPer = round((dM_radial_nb/DiscMass)*100,3)
#    #print(f'Fraction of Mass for dM values (all au): {MPer}%')
#
#    #Beta model
#    #Mass
#    print(f'Total Disc Mass | Beta Model (dmSum): {dM_radial_b} g ')
#    MPerb = round((dM_radial_b/DiscMass)*100,3)
#    #print(f'Fraction of Mass for dM values (all au): {MPerb}%')
#
#    SED_total = np.add(flux_b,flux_sa)
##    SED_totalnb = np.add(flux_nb,flux_sa)

    
#    return df_fluxsmbg,df_rbins_dNg, df_rbins_dMg #returns SED Total for given set of initial conditions
#    plt.plot(rbins_bm,flux_br)
#    plt.xscale('log')
#    plt.show()
    return flux_br
#-----------------------------------------------------------------#
#Dust Migration model - FAST Method
#Revamping the original code to remove the reliance on dataframes
#Simply return flux values as a function of distance for filter passbands
def DustyMM_fdist(smin,smax,dfrac,q,rm,rw,rin,rout,Flux_Filter):
    ##Planestimal Belt Characteristics##
    print(f'Min. Grain Size: {smin}, dfrac: {dfrac}, q: {q}, rmean: {rm}, rsigma: {rw}')
    ##Total Grain Mass##
    DiscMass = dfrac*Me*10**3 #Total grain mass as a fraction of Earth Mass (converted to grams) because denisty(rho) is given in g/cm^3
#    print(f'Mass of Disc: {DiscMass} g') #inform human

#    Number of grain sizes
    no_s = 100

#    create grain size space where most grains are towards the smaller size (logarithmic spaced in linear fashion)
    s_gs = np.geomspace(smin,smax,no_s)
    print(f'Number of Grain Sizes: {len(s_gs)} ranging from {s_gs[0]} to {s_gs[-1]} microns')

    gr_blowout = [0]
    grainsizes = []

    #Initiate variables
    dM_radial_nb = 0   #Mass of non-beta grains
    flux_nb = np.zeros(len(wr))        #flux of non-beta grains
 
    dM_radial_b = 0   #mass of beta grains
    flux_b = np.zeros(len(rbins_bm))         #flux of beta grains


    ##grain size distribution dN \propto s^-q ds #Power law mass distribution + pre-fill dataframes for later summing of angle release
    ##Determine Constant to find exact number and mass of grains according to size##
    #print(f'Calculating Mass proportional constant...')
    dMs = (s_gs*1e-4)**-q*rho*(4/3)*pi*(s_gs*1e-4)**3
    Msum = np.sum(dMs)      #unscaled mass component
    dNc = DiscMass/Msum    #finding constant
    
    #Initial array for
    #dNsr = np.zeros(len(rbins_bm))
    
    #print(f'Proportional Constant for dN proto s**-q: {dNc} where grain size (s) in units of cm')

    for s in s_gs:
        

        flux_nb = 0

        #Beta-grains model
        #Determine beta-value | considering radiation pressure on grains
        be = 0.574*Ls*Qpr_s(s)/(Ms*rho*s) #beta-value in solar units.  Density of grain in g/cm^3
         
        if be == 0.5:
            #blow out limit
            print(f'Blowout grain size: {a}')
            #ignores grain
        if be > 0.5:
            print(f'Grains with size: {s} micron & beta: {be} are on unbound orbits are ignored, i.e. blown out of system (be > 0.5)')
            if gr_blowout[-1] != s:
                gr_blowout.append(s)
        if be < 0:
            print(f'something is screwed (be < 0), beta = {be}, Ls = {Ls}, Qpr = {Qpr_s(s)}.  Exiting...')
            print(zed)
        elif be >= 0 and be < 0.5:
            #Bound grains
            #Orbital values for 0 <= be < 0.5
            #print(f'Grain size: {a} is on a bound orbit with be: {be} (0 < be < 0.5)')
            
            dNb = f_sd(s,rbins_bm)
            dNb = dNb.flatten()
            dNb = (dNb/np.sum(dNb))*(dNc*(s*1e-4)**-q)
#            dNb = dNb/np.sum(dNb)
          

            dNbr = np.trim_zeros(dNb)  #remove zeroes from array
       
            #Summing mass values
            dM_radial_b = dM_radial_b + np.sum(dNb*rho*(4/3)*pi*(s*1e-4)**3)
            

            glocID = dNb/dNb                       #Mask for grain locations
            glocID[np.isnan(glocID)] = 0            #Remove NaN
            

            
            gloc = np.trim_zeros(rbins_bm*glocID)   #radial locations where grains of size (s) are present

            
            GTg = Tg_sr(s,rbins_bm).flatten()           #Grain temperatures as a function of radial locations
            
            
            BBfns = f_BBgT(GTg,wr)
         
            
            BBfns_mod1 = np.multiply(np.multiply(BBfns,Qabs_sw(s,wr)),np.multiply(dNb,rbins_bm))
            
            BBfns_mod2 = BBfns_mod1*8*pi**2*(s*10**-6)**3*(dpc*pc)**-2/hd_bs
            
            BBfns_filter = np.multiply(BBfns_mod2,np.vstack(Flux_Filter))  #applying filter Flux(wavelength)

            BBfns_nu = 10**26*BBfns_filter*((np.vstack(wr)*10**-6)**2)/(c)  #converting from Flux(wavelength) to Flux(frequency)
            

#            BBfns_filter = np.multiply(BBfns_mod3,np.vstack(Flux_Filter)) #applying filter Flux(Frequency)
#
            
#            BBfns = np.multiply(np.multiply(BBfns,Qabs_sw(s,wr)),np.multiply(dNbr[::-1],gloc[::-1]))    #multiply blackbody fns by corresponding number density * radial distance
                            #BBfns = np.multiply(BBfns,dNbr)    #multiply blackbody fns by corresponding number density * radial distance
            Fnu_r = np.sum(BBfns_nu, axis = 0)     #sum all BB all flux values as a function of distance together
            

             #Scattered light
#                Alam = Qsca_sw(s,1:)/(Qsca_sw(s,1:)+Qabs_sw(s,1:)) #Albedo for scattered light
#                Flsca = dNb*((Rs*R_s)/(dpc*pc))**2*pi*Blam(Ts,wr*10**-6)*Alam*Qsca_sw(s,1:)*(s*10**-6/(2*rg*au))**2 #Scattered Emission
#
            #Summing Fluxes
#
            flux_b = flux_b + Fnu_r #Add flux
    

    MPerb = round((dM_radial_b/DiscMass)*100,3)

    
    return flux_b #returns radial profile(s) for given set of initial conditions normalised for comparison
#    return SED_total
#-----------------------------------------------------------------#
#find nearest value in array
def find_nearest(array,value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#-----------------------------------------------------------------#
##Function:
#Produces a grain temperature profile (s,r)
#Produce radiation pressure effiencienes from Mie Theory: Qpr (s) | Qsca(s,wavelength) | Qabs(s,wavelength)

def radpressure(Ls,Ms,Rs,Ts,rho,composition):
    #Inputs:
    #Ls: Stellar luminosity (float) [Solar Luminosity units]
    #Ms: Stellar mass (float) [Solar mass unit]
    #Ts: Stellar photospheric temperature (float) [Kelvin]
    #rho: grain density (float) [g/cm^3]
    #composition: grain composition (string) e.g. silicate_d03, dirty_ice [user defined]

    #set grain size regime
    n_gs = 100
    srgs = np.geomspace(0.01,3000,n_gs) #Set sizes

    #set wavelength regime
    wrange = np.geomspace(0.01,3000,101)    #wavelength range used for this calculation | note this is different from the user defined for determining overall SED
    
    #Grain temperatures from Stellar Spectrum + Optical Contants gt(s,r) = grain temp
    ##Grain Temperature range##
    #Check if temperature grid profile exists
    Tg_file = 'df_Tg_'+composition+'_Ts'+str(Ts)+'_ns_'+str(n_gs)+'_master.csv'
    ##Check for 'Optical' efficiency terms##
    Qpr_file = 'df_Qpr_'+composition+'_Ts_'+str(Ts)+'_rho'+str(rho)+'_ns_'+str(n_gs)+'_master.csv'
    Qabs_file = 'df_Qabs_'+composition+'_Ts_'+str(Ts)+'_rho'+str(rho)+'_ns_'+str(n_gs)+'_master.csv'
    Qsca_file = 'df_Qsca_'+composition+'_Ts_'+str(Ts)+'_rho'+str(rho)+'_ns_'+str(n_gs)+'_master.csv'


   
    if path.exists(Tg_file) == True and path.exists(Qpr_file) == True and path.exists(Qabs_file) == True and path.exists(Qsca_file) == True:
        #Read in file
        print(f'All temperature and optical constants files were discovered.')
        df_Tg = pd.read_csv(Tg_file)
        df_Qpr = pd.read_csv(Qpr_file)
        df_Qabs = pd.read_csv(Qabs_file)
        df_Qsca = pd.read_csv(Qsca_file)
            
    else:
        #Note: Blackbody (Luminosity values) should be a function WAVELENGTH (not frequency
        print(f'One or more of the following files: {Tg_file},{Qpr_file},{Qabs_file},{Qsca_file} were not found.  Creating the temperature profile and optical constant (Qpr,Qsca, and Qabs) efficiencies. This may take several minutes...')
        gtr = np.geomspace(2.5,1500,101) #initial temperature range
        fact = 0.5*Rs*R_s/au #Rs is in solar units | constant to help determine grain temperature
        nv = fn(wrange) #n optical constant as a function of the wavelength space
        kv = fk(wrange) #k optical constant as a function of the wavelength space

        #Blackbody spectrum of Star | as recommended by Wolf&Hillanbrand 2003
        BStar = Blam(Ts,wrange*10**-6)
#
#        BlkStar = Blam(Ts,wrange*10**-6)
#        BlkStar_max = np.max(BlkStar)
#
#        #import flux values from model stellar spectrum for designated wavelengths values
#        f_10spek = fun_spek(np.log10(wrange))
#        BStar = BlkStar_max*f_10spek/np.max(f_10spek)

        #Temperature profile
        df_Tg = pd.DataFrame({'R': rbins_bm}) #temperature of the grains of size (s) for q to each bin: (Tg)

        #Qabs + Qsca profiles
        df_Qabs = pd.DataFrame({'Wavelength (um)': wrange}) #Dataframe for Qabs values
        df_Qsca = pd.DataFrame({'Wavelength (um)': wrange}) #Dataframe for Qsca values
        df_Qpr = pd.DataFrame(columns=['A'], index=range(1)) #Dataframe for Qpr values - summed over stellar spectrum
        

        for s in srgs:
            s_colname = str(round(s,4))
            #finding component for grain temperature (to determine later)
            x = 2*pi*s/wrange
            qext, qsca, qback, g = mpy.mie(nv-1.0j*kv,x)
            qabs = (qext - qsca)  #extinction - scattering coefficient
            qpr = qabs + qsca*(1-g)  #pressure coeficienct
            star = trapz(BStar*qabs,wrange)
#            print(f'star: {star}')
#            print(zed)
            
            #Fill dataframes
            df_Qpr[s_colname] = trapz(qpr*BStar,wrange)/trapz(BStar,wrange)
            df_Qabs[s_colname] = qabs
            df_Qsca[s_colname] = qsca

           
            #iterate through possible grain temperatures
            r_val = []
            for gt in gtr:
                BBFluxg = Blam(gt,wrange*10**-6) #blackbody temperature of grain
                dust = trapz(BBFluxg*qabs,wrange) #area underneath BB curve for grain
#                print(f'dust: {dust}')
                
                rgt = fact*np.sqrt(star/dust) #radial distance of grain [au]
                
                #print(f'temp: {gt}, distance: {rgt}')
                
                
                #print(dust)
                r_val.append(rgt) #append radial distance of grain at temperature
            
            #print(zed)

            frgt = interpolate.interp1d(r_val,gtr,kind = 'linear',fill_value='extrapolate')  #interpolate radial distance with grain temp
            df_Tg[s_colname] = frgt(rbins_bm) #Temperature profile as a function of radial distance for grain size

        #Save Temperature profile to csv
        df_Tg.to_csv(Tg_file,index=False)
        print(f'Temperature grid space created and saved as {Tg_file}')
        #Save Qpr file to csv
        df_Qpr.to_csv(Qpr_file,index=False)
        #Save Qsca + Qabs file to csv
        df_Qabs.to_csv(Qabs_file,index=False)
        df_Qsca.to_csv(Qsca_file,index=False)


    #Print results
    #Print Temperatures
    print(f'Temperature Grid space:')
    print(df_Tg)
    #Print Optical Efficiency values
    print(f'Absorption Coefficient (Qabs) values:')
    print(df_Qabs)
    print(f'Scattering Coefficent (Qsca) values:')
    print(df_Qsca)
    print(f'Pressure Coefficent (Qpr) values:')
    print(df_Qpr)
    
    #Interpolate 2D grid space for grain size regime and radial distance values
    f_Tg = interpolate.interp2d(srgs,rbins_bm,df_Tg.iloc[:,1:],kind='linear')  #f_Tg{s,r) s: grain size / r: radial distance

    #2D Grid for Qabs
    f_Qabs = interpolate.interp2d(srgs,wrange,df_Qabs.iloc[:,1:].to_numpy(), kind = 'linear')
    #2D Grid for Qsca
    f_Qsca = interpolate.interp2d(srgs,wrange,df_Qsca.iloc[:,1:].to_numpy(), kind = 'linear')
    #1D grid for Qsca
    f_Qpr = interpolate.interp1d(srgs,df_Qpr.iloc[0,1:].to_numpy(), kind = 'linear')

    
    
    ##Determine blow out size(s)
    
    beta = []
    for s in srgs:
        s_colname = str(round(s,4))
        Qpr = df_Qpr[s_colname][0]
        beta.append(0.574*Ls*Qpr/(Ms*rho*s))

    #new s_range
    s_range = np.geomspace(0.01,3000,3000)
    b_fn = interpolate.interp1d(srgs,beta,kind = 'linear',fill_value='extrapolate')
    b_interp = b_fn(s_range)
    
    blowlim = 0.5*np.ones(len(s_range))
    idx = np.argwhere(np.diff(np.sign(blowlim - b_interp))).flatten()
    sblow = []
    for i in idx:
        sblow.append(s_range[i])
    
    #Outputs
    #f_Tg: function of grain temperature space (s,r) s: grain size [micron], radial distance [au]
    #f_Qpr: function of presssure efficiency (s)
    #f_Qabs: function of absorption coeficient (s,wavelength)
    #f_Qsca: function of scatter efficiency (s,wavelength)
    #sblow: blowout size(s) [micron]
    #beta: beta values as a fucntion of grain size
    
    return [f_Tg,f_Qpr, f_Qabs, f_Qsca, sblow,beta]

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

    return c2

#-----------------------------------------------------------------#
##emcee functions###
##Inlike() function for emcee
#def lnlike(phot_wav,phot_flux,phot_unc,func_wav,func_flux):
def lnlike(theta,phot_wav,phot_flux,phot_unc):
    #sm,dfrac,qv,rm,rw = theta
#    sm,dfrac,qv,rm,rw = theta  #add time component???
    sm,dfrac,qv = theta  #add time component???
    func_flux,_,_ = DustyMM(sm,smax_r,dfrac,qv,rmean,rwidth,rin,rout)
    func_wf = interpolate.interp1d(wr,func_flux)
    flam = func_wf(phot_wav)
#    #print(flam)
    c2 = 0
    for i in range(len(phot_wav)):
        c2 = c2 + (((phot_flux[i]-flam[i]))/phot_unc[i])**2
#    lp = -0.5*np.sum( (phot_flux - flam) /phot_unc **2)
    chi2 = -0.5*c2
    print(chi2)
    return chi2

def lnprior(theta):
#    sm,dfrac,q,rm,rw = theta
    sm,dfrac,q = theta
#    if 0.01 < sm < 15.0 and 0.001 < dfrac < 0.1 and 3.0 < q < 4.0 and 125 < rm < 145 and 5 < rw < 15:
    if 0.01 < sm < 30.0 and 0.001 < dfrac < 0.5 and 3.0 < q < 4.0: #noting blowout for dirty ice is 2.69 micron for HD 105211 / astrosilicate: blowout 1.601 0.044 -> 0.711
        return 0.0
 
    else:
        return -np.inf


def lnprob(theta,wav,flx,unc):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,wav,flx,unc)

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
object = 'HD48682'
#Stellar Photometric values for comparison
#
lam = [3.4,    4.6,    9.0,    12.0,   22.0,  30, 32, 34, 70.0,  100.0,  160.0,  250.0,  350.0,  850.0]
flx = [8.6659, 4.5114, 1.3534, 0.7610, 0.2444, 0.148, 0.142, 0.136, 0.290,  0.275,  0.177,  0.09,   0.025,  0.0039]
unc = [2.7266, 1.1322, 0.0451, 0.0113, 0.005,  0.013, 0.017, 0.020, 0.038,  0.007,  0.024,  0.015,  0.008,  0.0008]
#
 
#IRS Spectrum
#HD 48682
irs_spec = ascii.read('IRS.txt')
irs_spec = np.loadtxt('IRS.txt', dtype=[('wavelength', float), ('flux', float), ('error', float)])
##Host Star properties in Solar units##
Ls = 1.83          #Luminosity of Star [Solar Luminosity]
Ms = 1.17        #Mass of Star [Solar Masses]
Rs = 1.23          #Radius of Star [Solar Radius]
Ts = 6086       #star temperature [K]
dpc = 16.65        #distance to star system [pc]

##Star Flux values: Blackbody + Spectrum (if available)
##Blackbody Stellar spectrum (default)
bb = BBody(temperature=Ts*u.K)
d = dpc*pc   #distance to Star [m]
Rad = Rs*R_s   #Star Radius: Sun [m]
As = pi*(Rad/d)**2 #Amplitude of Star's blackbody planck fn (i.e. for this case: cross-section of Sun) and convert to Jansky (as BBody units are given in: erg / (cm2 Hz s sr))
flux_s = As*Blam(Ts,wr*10**-6)*10**26*((wr*10**-6)**2)/(c)

##Stellar Spectrum##
#HD 48682 Stellar Spectrum#
f_spek = np.loadtxt('SM.txt', dtype=[('wave', float),('fspek', float)])

wave = f_spek["wave"]
fspek = f_spek["fspek"]

fun_spek = interpolate.interp1d(np.log10(wave),np.log10(fspek),kind = 'linear', fill_value = 'extrapolate')
f_10spek = fun_spek(np.log10(wr))
flux_sa = 10**(f_10spek)

#-----------------------------------------------------------------#
##Radial distance parameter space##
##distances are in au##
hd_bs = 1      #size of bin: recommened for debris discs: 1 au
hd_min = 0      #minimum radial distance
hd_max = 1600  #maximum radial distance: typically 1000 au
hd_minm = hd_min + hd_bs/2   #minimum mid space  value + hd_bs/2
hd_maxm = hd_max - hd_bs/2  #maximum mid space value - hd_bs/2
print(f'Maximum distance of grain used in model: {hd_max} au')  #note smallest grains may go beyond this distance (depending on r_min)
hd_steps = np.int64((hd_maxm-hd_minm)/hd_bs + 1)
rbins_bm = np.linspace(hd_minm, hd_maxm,hd_steps)

#bin space for histogram (binning radial values in orbit def)
bin_steps = np.int64(hd_max/hd_bs + 1)
rbins_bins = np.linspace(hd_min,hd_max,bin_steps)

#Integrate to a maximum heliocentric distance to save computational time for Ls = 1 r ~ 1000 au includes ~90% mass of disc
#rmax = b_rmax - 1 #to include the entire disc use: rmax = b_rmax - 1
rmax = 1600     #typically 1000 au
print(f'Limiting the flux integration up to a stellar distance of: {rmax} au to save computational time')
#-----------------------------------------------------------------#
##Grain properties##
rho = 3.3 #grain density - astro-silicate (typical)
#rho = 2.0 #grain density - dirty ice (typical)

###Grain Optical Constants
composition = 'silicate_d03'
comp = 'astro-silicate'
#composition = 'diice_hypv2'
#comp = 'dirty ice'
#read in nk (optical constants file): 'text' file Col1: Wavelength (um) range: ~10^-5 to ~10^5 / Col2: n Col3: k
nk = pd.read_csv(composition+'.lnk', header = None, delim_whitespace=True)

#extract column information
nkw = nk.iloc[:,0].to_numpy() #if given as a function of wavelength
n = nk.iloc[:,1].to_numpy() #n values optical constants for dirty ice
k = nk.iloc[:,2].to_numpy() #k values optical constants for dirty ice

#create functions for n & k
fn = interpolate.interp1d(nkw,n,fill_value = 'extrapolate')
fk = interpolate.interp1d(nkw,k,fill_value = 'extrapolate')
nv = fn(wr) #n optical constant as a function of the wavelength space
kv = fk(wr) #k optical constant as a function of the wavelength space
#Determine Temperature,Qpr,Qabs,Qsca profiles
#Checks if temperate profile and mie theory files are present for host star and grain composition & density. If not, creates new files.
[Tg_sr,Qpr_s,Qabs_sw,Qsca_sw,sblowA,beta] = radpressure(Ls,Ms,Rs,Ts,rho,composition)
#-----------------------------------------------------------------#
#Grab Filter Information from relevant observatory/instrument - transmission profiles
#Note: may need to normalise profile to obtain relative transmission
#Grab Herschel PACS Filter Information - blue (70) , green (100), & red (160): source: http://svo2.cab.inta-csic.es/theory/fps/index.php?id=Herschel/Pacs.green&&mode=browse&gname=Herschel&gname2=Pacs#filter

#Herschel/PACS 70 Blue Filter
FilterBlue70 = pd.read_csv('Herschel_Pacs.blue.dat', header = None, delim_whitespace = True)
FBlue70w = FilterBlue70.iloc[:,0].to_numpy()/1e4
FBlue70t = FilterBlue70.iloc[:,1].to_numpy()
#FBlue70t = FBlue70t/np.sum(FBlue70t)  #determine transmission profile by
fFBlue70 = interpolate.interp1d(FBlue70w,FBlue70t, fill_value = 'extrapolate')
fFBlue70wr = fFBlue70(wr)
fFBlue70wr[fFBlue70wr<0] = 0

#Herschel/PACS 100 Green Filter
FilterGreen100= pd.read_csv('Herschel_Pacs.green.dat', header = None, delim_whitespace = True)
FGreen100w = FilterGreen100.iloc[:,0].to_numpy()/1e4
FGreen100t = FilterGreen100.iloc[:,1].to_numpy()
#FBlue70t = FBlue70t/np.sum(FBlue70t)  #determine transmission profile by
fFGreen100 = interpolate.interp1d(FGreen100w,FGreen100t, fill_value = 'extrapolate')
fFGreen100wr = fFGreen100(wr)
fFGreen100wr[fFGreen100wr<0] = 0

#Herschel/PACS 160 Green Filter
FilterRed160= pd.read_csv('Herschel_Pacs.red.dat', header = None, delim_whitespace = True)
FRed160w = FilterRed160.iloc[:,0].to_numpy()/1e4
FRed160t = FilterRed160.iloc[:,1].to_numpy()
#FBlue70t = FBlue70t/np.sum(FBlue70t)  #determine transmission profile by
fFRed160 = interpolate.interp1d(FRed160w,FRed160t, fill_value = 'extrapolate')
fFRed160wr = fFRed160(wr)
fFRed160wr[fFRed160wr<0] = 0


#JWST/MIRI Filter Data
MIRIFilterData = pd.read_excel('MIRI_Filter_Profiles.xlsx', header = 0)
MIRIwr = MIRIFilterData.iloc[:,0].to_numpy() #Grab wavelengths [microns]
#Filter information
#5.6 micron
MIRI_F560 = MIRIFilterData.iloc[:,1].to_numpy()
fMIRI_F560 = interpolate.interp1d(MIRIwr,MIRI_F560, fill_value = 'extrapolate')
#7.7 micron
MIRI_F770 = MIRIFilterData.iloc[:,2].to_numpy()
fMIRI_F770 = interpolate.interp1d(MIRIwr,MIRI_F770, fill_value = 'extrapolate')
#10.0 micron
MIRI_F100 = MIRIFilterData.iloc[:,3].to_numpy()
fMIRI_F100 = interpolate.interp1d(MIRIwr,MIRI_F100, fill_value = 'extrapolate')
fMIRI_F100wr = fMIRI_F100(wr)
fMIRI_F100wr[fMIRI_F100wr<0] = 0
#11.3 micron
MIRI_F113 = MIRIFilterData.iloc[:,4].to_numpy()
fMIRI_F113 = interpolate.interp1d(MIRIwr,MIRI_F113, fill_value = 'extrapolate')
#12.8 micron
MIRI_F128 = MIRIFilterData.iloc[:,5].to_numpy()
fMIRI_F128 = interpolate.interp1d(MIRIwr,MIRI_F128, fill_value = 'extrapolate')
#15.0 micron
MIRI_F150 = MIRIFilterData.iloc[:,6].to_numpy()
fMIRI_F150 = interpolate.interp1d(MIRIwr,MIRI_F150, fill_value = 'extrapolate')
#18.0 micron
MIRI_F180 = MIRIFilterData.iloc[:,7].to_numpy()
fMIRI_F180 = interpolate.interp1d(MIRIwr,MIRI_F180, fill_value = 'extrapolate')
#21.0 micron
MIRI_F210 = MIRIFilterData.iloc[:,8].to_numpy()
fMIRI_F210 = interpolate.interp1d(MIRIwr,MIRI_F210, fill_value = 'extrapolate')
fMIRI_F210wr = fMIRI_F210(wr)
fMIRI_F210wr[fMIRI_F210wr<0] = 0
#25.5 micron
MIRI_F255 = MIRIFilterData.iloc[:,9].to_numpy()
fMIRI_F255 = interpolate.interp1d(MIRIwr,MIRI_F255, fill_value = 'extrapolate')

#plt.plot(wr,fMIRI_F100wr,'k',wr,fFBlue70wr,'b',wr,fFGreen100wr,'g',wr,fFRed160wr,'r' )
#plt.show()
#print(zed)



#-----------------------------------------------------------------#
#Filter flux according waveband/instrument/observatory
def FilterFunc(df_flux_sf,Filter):
#        Fluxvalsg = df_fluxsmbg.loc[df_fluxsmbg['Wavelength (um)'] == Wavelength]
#        print(Fluxvalsg) #check print to see if it has extracted appropriate wavelength values
    df_flux_sf['Filter'] = Filter
#    print(df_flux_sf)
    df_flux_sf.update(df_flux_sf.filter(regex='^Flux').mul(df_flux_sf['Filter'], axis=0))
#    df_flux = df_flux_sf[2:-2].multiply(df_flux_sf["Filter"], axis = "index")
#    df_flux_filter = df_flux_sf[:,2:-2].multiply(df_flux_sf[:,:-1], axis = "index")
#    print(df_flux_filter)
#    print(df_flux_sf.columns[2:-1])
#    print(zed)
    df_flux_sum = df_flux_sf.sum(axis = 0)
#    print(df_flux_sum)
#    print(df_flux_sum['Filter'])
##    print(list(df_flux_sum.rows.values))
##    print(df_flux_sum.columns[2:-1])
#    print(zed)
    bgs = [] #gs: grain grain  ('b' for beta?)
    bgd = [] #gd: grain distance
    bgf = [] #gf: grain flux

    for cval in df_flux_sf.columns[2:-1]:  #Extract all flux values
            val = df_flux_sum[cval]              #pulls all flux values at specific wavelength
            bvalues = re.findall('\d*\.?\d+',cval)    #finds all numbers in column name and places in a 2-element array of strings
            bgs.append(float(bvalues[0]))               #append grain size
            bgd.append(float(bvalues[1]))               #append grain distance
            bgf.append(val)                            #Append Flux value

    #set the zero flux values where there are no grains in the inner portion of the disc (note: it has already ignored zero flux values in the outer part of the disc)
    #helps in creating an ideal function: flux(distance) later
#    rinner = math.floor(bgd[0])
#    inDisc = np.linspace(0,rinner,rinner+1)
#
#    for i in inDisc:
#        bgs.append(0)   #append grain size
#        bgd.append(i)   #append grain distance
#        bgf.append(0)   #Append Flux value

    #Create Interpolated Function of Flux as a function of distance
    df_fldg = pd.DataFrame({'Distance': bgd, 'Flux': bgf})       #Grab all grain sizes at distances
    df_fld_ssg = df_fldg.groupby('Distance',as_index=False).sum()  #sorts flux values by distance and sum all flux values (hence all sizes) at said distance
    f_flux = interpolate.interp1d(df_fld_ssg['Distance'],df_fld_ssg['Flux'], fill_value = 'extrapolate')  #f_flux: flux as a function of distance: flux(distance)
    
    return f_flux
#-----------------------------------------------------------------#
#directories
direc = os.getcwd()
direcmain = '/DModelResults_'+object
main_direc = direc + direcmain
subprocess.run(['mkdir',main_direc])
#-----------------------------------------------------------------#
#Create Blackbody Profiles for temperature range
gtr = np.geomspace(2.5,1500,500)
df_BBprofiles = pd.DataFrame({'Wavelength (um)': wr})
for gt in gtr:
    df_BBprofiles[gt] = Blam(gt,wr*10**-6)

#print(df_BBprofiles)
#plt.plot(df_BBprofiles)
#plt.show()
#plt.xscale('log')
#plt.yscale('log')
#print(zed)
#Interpolate 2D grid space for blackbody profiles corresponding for temperatureo of grain size regime at radial distance values
f_BBgT = interpolate.interp2d(gtr,wr,df_BBprofiles.iloc[:,1:],kind='linear')  #f_BBgT (grain temperature, wavelength range)
#-----------------------------------------------------------------#
##Creating Density grid for grains released from 1au##
##To then be fitted any grain release at any distance
thr = np.linspace(0,pi,36000)  #angle space for radial function

#-----------------------------------------------------------------#

#Change directory to store subsequent files + images
os.chdir(main_direc)


#-----------------------------------------------------------------#
#Constant Input parameters not included in emcee analysis
smax_r = 3000
rin = 40
rout = 135
rmean = 89
rwidth = 36.5/2.2355
#rin = 120
#rout = 170
#rmean = 150
#rwidth = 5
#Tpr = 2.5e8 #1000 #1e9  #Fixed for now but may need to vary
#Tprs = f'{Tpr:.2E}'
#Tpr_range = [1000, 1e9, 3.2e9, 6.4e9, 8.9e9]
Tpr_range = [3.2e9]
tau = 7.2e-5


#set grain size regime
n_gs = 100
s_gs = np.geomspace(0.01,3000,n_gs) #Set sizes
#print(s_gs)
#print(np.geomspace(0.01,3000,n_gs))
#print(zed)

#Dataframe for probability of finding grain and distance
df_rbins_s = pd.DataFrame({'Distance (au)': rbins_bm})


#Dirty Ice
#sm_mcmc = [3.191,3.79,-1.834]
#dfrac_mcmc = [0.03383,0.0939,-0.00716]
#q_mcmc = [3.577,0.11,-0.088]

#astro-silicate
sm_mcmc = [3.612,0.59,-0.491]
dfrac_mcmc = [0.05696,0.00396,-0.00390]
q_mcmc = [3.855,0.07,-0.07]

#dNc
#q = 0   #default but this would need to implemented inside function

#dMs = (s_gs*1e-4)**-q*rho*(4/3)*pi*(s_gs*1e-4)**3
#Msum = np.sum(dMs)      #unscaled mass component
#dNc = DiscMass/Msum    #finding constant

r_steps = np.int64((rout-rin)/hd_bs + 1) #steps will be same size as rbins_bm values
r_belt = np.linspace(rin, rout, r_steps) #radial parametric space for belt

##Planetesimal Belt Model##
#print(f'Mean distance of grain release: {rm} au')

ddgauss = []
for r in rbins_bm:
    if r >= rin and r <= rout: #inside belt create belt value at rbin_bm location
        ddgauss.append(np.exp(-0.5*((r - rmean)/rwidth)**2))
    else:
        ddgauss.append(0)

#Normalise models to give a total sum of radial distribution to be unity
ddgauss = ddgauss/(np.sum(ddgauss))
#Final probability distribution value of planetesimals
f_gauss = interpolate.interp1d(rbins_bm,ddgauss)

for Tpr in Tpr_range:

    for s in s_gs:
        be = 0.574*Ls*Qpr_s(s)/(Ms*rho*s)
        s_col = str(round(s,3))
        den = np.zeros(len(rbins_bm))
        if be < 0.5:
    #        print(f'{be}, {s}')
            for r in r_belt:
                den = den + f_gauss(r)*prob_grain_rp_p(be,r,Tpr)
    #    df_rbins_s[s_col] = den*(be/0.25)**q
        df_rbins_s[s_col] = den


#    #Interpolate 2D grid space for grain distribution as a function of size and radial distance values
    f_sd = interpolate.interp2d(s_gs,rbins_bm,df_rbins_s.iloc[:,1:],kind='linear')  #f_sd (grain size, radial distance)
    
    
#    df_fluxsmbg,df_rbins_dNg, df_rbins_dMg = DustyMM_ext(1.5,3000,0.03,3.5,120,5,100,140)
#
#    print(df_fluxsmbg)
#    print(df_fluxsmbg.shape)
#    print(fMIRI_F210wr)
#
#    flux_f = FilterFunc(df_fluxsmbg,fMIRI_F210wr)
#    print(f)
#
###    print(zed)
#    Filter_70 = DustyMM_ext(5.3,3000,0.04,3.826,rmean,rwidth,rin,rout,fMIRI_F210wr)
##
#    Filter_100 = DustyMM_ext(5.3,3000,0.04,3.826,rmean,rwidth,rin,rout,fFGreen100wr)
#
#    Filter_160 = DustyMM_ext(5.3,3000,0.04,3.826,rmean,rwidth,rin,rout,fFRed160wr)
##
#    plt.plot(rbins_bm,Filter_70,'b')
#    plt.plot(rbins_bm,Filter_100,'g')
#    plt.plot(rbins_bm,Filter_160,'r')
#    plt.show()
#
#    print(rbins_bm[Filter_70.argmax()])
#    print(rbins_bm[Filter_100.argmax()])
#    print(rbins_bm[Filter_160.argmax()])
#    print(zed)
#
#    RadProfile = DustyMM_fdist(5,3000,0.9,3.5,150,5,120,180)
#
#    plt.plot(rbins_bm,RadProfile,'b')
#    plt.show()
#    print(zed)

    #print(f_sd(9,rbins_bm))
    ##print(f_BBgT(100,wr))
    #print(zed)
    #s_low = np.float64(df_rbins_s.columns[-1])
    #print(df_rbins_s)

    #testing flux values at specific wavelength
    #f_fluxsmbg,df_rbins_dNg, df_rbins_dMg = DustyMM_ext(8,smax_r,0.8,3.5,rmean,rwidth,rin,rout)
    #print(zed)

    ###Emcee Inputs##
    nwalkers = 30
    niter = 500
    #initial = np.array([5,0.05,3.5,133,10]) #Variables to be tested: sm,dfrac,qv,rm,rw
    initial = np.array([3,0.05,3.5]) #Variables to be tested: sm,dfrac,qv
    ndim = len(initial)
    p0 = [np.array(initial) + np.array([random.uniform(-2.99,5),random.uniform(-0.04,0.04),random.uniform(-0.5,0.5)],) for i in range(nwalkers)]
    #p0 = [np.array(initial) + np.array([random.uniform(-4.9,10.0),random.uniform(-0.03,0.03),random.uniform(-0.5,0.5),random.uniform(-3,3),random.uniform(-3,3)],) for i in range(nwalkers)]


    print('Walkers for simulation:')
    print(p0)
    data = lam,flx,unc


    def main(p0,nwalkers,niter,ndim,lnprob,data):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data) #, backend=backend)
        

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 300)
        burnin = sampler.get_chain()

        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter)

        return sampler, pos, prob, state, burnin

    #saving output???
    #fname = object+"_"+composition+str(Tpr)+".h5"
    #backend = emcee.backends.HDFBackend(fname)
    #
    #
    #if path.exists(fname) == True:
    #    sampler = emcee.backends.HDFBackend(fname)
    #else:
    #    #run main function to get sample
    #    sampler, pos, prob, state, burnin = main(p0,nwalkers,niter,ndim,lnprob,data)

#    temcee0 = time.time()
#
#    #run main function to get sample
#    sampler, pos, prob, state, burnin = main(p0,nwalkers,niter,ndim,lnprob,data)
#
#    temcee = time.time()
#    t = round((temcee - temcee0)/60,4)
#    print(f'Time to undergo emcee: {t} minutes')
#
#    samples = sampler.flatchain
#    np.savetxt("samples_"+object+"_"+composition+".csv", samples, delimiter = ",")
#
#    #sm_mcmc, dfrac_mcmc, q_mcmc, rm_mcmc, rw_mcmc = np.median(samples, axis=0)
#    #sm_mcmc, dfrac_mcmc, q_mcmc = np.median(samples, axis=0)
#    sm_mcmc, dfrac_mcmc, q_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84],axis=0)))
#
#    ##Fast function - Grab SED info and mass distribution by running DustyMM again
    [SED_total,SED_disc,Mperb] = DustyMM(sm_mcmc[0],3000,dfrac_mcmc[0],q_mcmc[0],rmean,rwidth,rin,rout)
    #
    #Fraction of disc mass
    dfracf = dfrac_mcmc[0]*Mperb/100

    #2 = Chi2(lam[8:-1],flx[8:-1],unc[8:-1],wr ,SED_total)
    c2 = Chi2(lam,flx,unc,wr ,SED_total)
    #chi2.append(c2)
    print(f'Chi2: {c2}')

    ##New Directory folder
    direcsav = '/'+object+'_smin'+str(round(sm_mcmc[0],2))+'_rm'+str(rmean)+'_dfrac'+str(round(dfrac_mcmc[0],2))+'_q'+str(round(q_mcmc[0],2))+'_rw'+str(round(rwidth,2))+'_rin'+str(rin)+'_rout'+str(rout)+'_chisq_'+str(round(c2,2))+'_tpr'+f'{Tpr:.2E}'
    model_direc = main_direc + direcsav
    subprocess.run(['mkdir',model_direc])
    #Change directory to store subsequent images
    os.chdir(model_direc)#labels = ['s_min','dfrac','q','R_m','R_sig']
#
#    np.savetxt("SED_total_"+object+"_"+composition+".csv", SED_total, delimiter = ",")
#    np.savetxt("SED_disc_"+object+"_"+composition+".csv", SED_disc, delimiter = ",")
#    #np.savetxt("Mperb_"+object+"_"+composition+".csv", Mperb, delimiter = ",")
#    print('Saved files')
#    #print(Mperb)
#
#
#    ##Plotting##
#    labels = ['s_min','d_mass[E]','q', 'r_mean', 'r_sigma']
#    fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84], title_fmt ='.3f')
#    cornerfig = object+'Emcee_corner_nwalkers'+str(nwalkers)+'_niter'+str(niter)+composition+'.pdf'
#    fig.savefig(cornerfig)
#
#    print("s_min: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) mu".format(sm_mcmc[0],sm_mcmc[1], sm_mcmc[2]))
#    print("dfrac: {0:1.5f} (+{1:1.5f}, -{2:1.5f}) MEarth".format(dfrac_mcmc[0],dfrac_mcmc[1], dfrac_mcmc[2]))
#    print("q: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) [-]".format(q_mcmc[0],q_mcmc[1], q_mcmc[2]))
#    #print("rmean: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) au".format(rm_mcmc[0],rm_mcmc[1], rm_mcmc[2]))
#    #print("rwidth: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) au".format(rw_mcmc[0],rw_mcmc[1], rw_mcmc[2]))
#
#    filename = 'DModel_emcee_results_'+str(nwalkers)+'_'+str(niter)+composition+'.txt'
#    f = open(filename,"w+")
#    f.write("s_min: {0:1.3f} (+{1:1.2f}, -{2:1.3f}) mu \n".format(sm_mcmc[0],sm_mcmc[1], sm_mcmc[2]))
#    f.write("dfrac: {0:1.5f} (+{1:1.5f}, -{2:1.5f}) MEarth \n".format(dfrac_mcmc[0],dfrac_mcmc[1], dfrac_mcmc[2]))
#    f.write("q: {0:1.3f} (+{1:1.2f}, -{2:1.3f}) [-] \n".format(q_mcmc[0],q_mcmc[1], q_mcmc[2]))
#    #f.write("rmean: {0:1.3f} (+{1:1.2f}, -{2:1.3f}) au \n".format(rm_mcmc[0],rm_mcmc[1], rm_mcmc[2]))
#    #f.write("rwidth: {0:1.3f} (+{1:1.2f}, -{2:1.3f}) au \n".format(rw_mcmc[0],rw_mcmc[1], rw_mcmc[2]))
#    f.close()
#
#    #print(f's_min: {sm_mcmc}')
#    #print(f'dfrac: {dfrac_mcmc}')
#    #print(f'q: {q_mcmc}')
#
#    #Plot chains
#    #fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
#    fig, axes = plt.subplots(len(initial), figsize=(10, 7), sharex=True)
#    samples = sampler.get_chain()
#    #s2 = burnin.append(samples)
#    s2 = np.concatenate((burnin, samples), axis=0)
#
#    for i in range(ndim):
#        ax = axes[i]
#        ax.plot(s2[:, :, i], "k", alpha=0.3)
#        ax.plot(burnin[:, :, i], "r", alpha=0.3)  #plot burn-in independently
#        ax.set_xlim(0, len(burnin) + len(samples))
#        ax.set_ylabel(labels[i])
#        ax.yaxis.set_label_coords(-0.1, 0.5)
#
#    axes[-1].set_xlabel("step number");
#
#    #b_time = np.ones(len(burnin))
#    #ax.axvline('k-.')#, label = '')
#
#    chains = object+'_Emcee_chains_nwalkers'+str(nwalkers)+'_niter'+str(niter)+composition+'.pdf'
#    fig.savefig(chains)
#
#
#
#    ##Produce output file with summary of results
#    f = open("DModel_summary.txt","w+")
#    f.write("Summary of Fitting SED for %s \n" % (object))
#    f.write("Grain properties: \n")
#    f.write("Composition: %s \n" % (composition))
#    #f.write("Blowout size: %0.3f microns \n" % (sblowA[0]))
#    #f.write("Initial Minimum Grain Size: %0.3f microns, Final minimum grain size: %0.3f microns\n" % (sm,grainsizes[0]))
#    f.write("Size distribution exponent: %0.3f \n" % (q_mcmc[0]))
#    f.write("Belt properties:\n")
#    f.write("Mean Belt Stellar distance: %0.3f au with a Gaussian width of %0.3f \n" % (rmean,rwidth))
#    f.write("Inner Edge: %0.3f au and Outer Edge %0.3f \n" % (rin,rout))
#    f.write("Initial Disc Mass: %0.4f M(Earth) Final Disc Mass: %0.4f M(Earth) \n" % (dfrac_mcmc[0],dfracf))  #Disc mass as a fraction of Earth mass
#    f.write("PR Time: %0.4f Years \n" % (Tpr))  #Disc mass as a fraction of Earth mass
#    f.write("Chi^2 value: %0.3f \n" % (c2))
#
#    f.close()

    #
    #####Plot beta + gaussian + power: SEDs
    plt.clf()
    fig, ax = plt.subplots(nrows = 1, ncols = 1)

    ##HD 48682 sorted into wavelength regimes
    ##For plotting puprposes
    onr_lam = [0.4400,0.5500,0.71,1.25,1.65,2.2200]
    onr_flx = [19.62,28.67,34.3,31.699,26.851,17.278]
    onr_unc = [0.84,0.24,1.39,1.494,1.266,0.868]

    mir_lam = [3.4, 4.6, 9.0,12.0,18.0,22.0]
    mir_flx = [8.6659,4.5114,1.3534,0.7610,0.4625,0.2444]
    mir_unc = [2.400,0.8678,0.0451,0.011,0.0631,0.005]

    mir_slam = [30,32,34]
    mir_sflx = [0.148,0.142,0.136]
    mir_sunc = [0.013,0.017,0.02]

    fir_lam = [70.0,100.0,160.0,250.0,350.0,850.0]
    fir_flx = [0.2894,0.275,0.177,0.09,0.025,0.0039]
    fir_unc = [0.038,0.007,0.024,0.015,0.008,0.0008]

    #comp = 'DI'
    #ax.plot(wr,SFlux_flxwr, 'k:', label = 'Star')
    ax.plot(wr,flux_sa, 'k:',linewidth = 1, label = 'Star:'+object, zorder = 1)
    ax.plot(wr,SED_disc, 'b-.', label = 'Disc: '+comp, zorder = 2)
    ax.plot(wr,SED_total, 'k', linewidth = 1, label = 'Star+Disc', zorder = 3)
    #ax.plot(irs_spec['wavelength'],irs_spec['flux'],'b',label = 'IRS Spectrum')
    ax.errorbar(irs_spec['wavelength'],irs_spec['flux'],yerr=irs_spec['error'],fmt='.',color='blue',ecolor='blue', label = 'IRS Spectrum',zorder = 4)
    ax.errorbar(onr_lam,onr_flx,yerr=onr_unc,fmt='o',mec='green',mfc='green',ecolor='black',capsize=4.,capthick=1, label = 'Optical',zorder=5)
    ax.errorbar(mir_lam,mir_flx,yerr=mir_unc,fmt='o',mec='skyblue',mfc='skyblue',ecolor='black',capsize=4.,capthick=1, label = 'Mid-IR',zorder=6)
    ax.errorbar(mir_slam,mir_sflx,yerr=mir_sunc,fmt='o',mec='cyan',mfc='cyan',ecolor='black',capsize=4.,capthick=1, label = 'Synth-Phot',zorder=7)
    ax.errorbar(fir_lam,fir_flx,yerr=fir_unc,fmt='o',mec='red',mfc='red',ecolor='black',capsize=4.,capthick=1, label = 'Far-IR/Sub-mm',zorder=8)
    #ax.errorbar(smm_lam,smm_flx,yerr=smm_unc,fmt='o',mec='y',mfc='y',ecolor='black',capsize=4.,capthick=1, label = 'Sub-mm',zorder=8)

    ax.set_xlabel('Wavelength [$\mu$m]',fontsize = 16)
    ax.set_ylabel('Flux Density [Jy]', fontsize = 16)
    ax.set_xlim([0.3, 3000])
    ax.set_ylim([10**-3, 150])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.legend(loc = 'lower left')
    #plt.show()
    #print(zed)
    #
    plt.savefig('SED_Star+disc_para_models_'+object+'_'+composition+'.pdf')
    ####

    print(zed)
    ###Plotting s probablity values as a function of distance
    #s_low = np.float64(df_rbins_s.columns[0])

    ##Grab grain information
    sgd = np.geomspace(sm_mcmc[0],smax_r,100)
    DiscSD = f_sd(sgd,rbins_bm)

    #Dataframe for probability of finding grain and distance
    df_rbins_smin = pd.DataFrame()

    #ind = 0
    #for i in np.range(len(s_gs))
    #    if s_gs[i] > sm_mcmc[0]:
    #        break


    
    i = 0
    for s in s_gs:
        s_col = str(round(s,3))
        den = np.zeros(len(rbins_bm)) #resets to zero very loop
        if s > sm_mcmc[0]:
            be = 0.574*Ls*Qpr_s(s)/(Ms*rho*s)
            
            if be < 0.5:
        #        print(f'{be}, {s}')
                for r in r_belt:
                    pdgrain = prob_grain_rp_p(be,r,Tpr)
                    den = den + f_gauss(r)*pdgrain/np.sum(pdgrain)*(be/0.25)**q_mcmc[0]
        #    df_rbins_s[s_col] = den*(be/0.25)**q
        df_rbins_smin[s_col] = den

    #    #Interpolate 2D grid space for grain distribution as a function of size and radial distance values
#    f_sd = interpolate.interp2d(s_gs,rbins_bm,df_rbins_smin.iloc[:,1:],kind='linear')  #f_sd (grain size, radial distance)

    df_rbins_smin = df_rbins_smin.transpose()  #transpose the dataframe diagonally

    #remove bad pixels from interpolation (i.e. zero probably will be black in colour)
    my_cmap = copy.copy(matplotlib.cm.get_cmap('gist_heat')) # copy the default cmap
    my_cmap.set_bad((0,0,0))

    #image show plot with extent, cmap = gist_hist
    #plt.imshow(df_rbins_s,extent=(hd_minm,hd_maxm,s_gs[-1],s_low),interpolation='nearest', cmap = my_cmap,norm=colors.LogNorm()) #, vmin = 0, vmax = 0.02)


    plt.clf()
    #plt.pcolor(rbins_bm, sgd, df_rbins_smin, cmap = my_cmap,norm=colors.LogNorm(),linewidth=0,rasterized=True)
    plt.pcolor(rbins_bm, s_gs, df_rbins_smin, cmap = my_cmap,norm=colors.LogNorm(),linewidth=0,rasterized=True) #,
    ax = plt.gca() #you first need to get the axis handle
    ax.set_aspect('auto') #sets the height to width ratio to 1.5.
    #    plt.aspect('auto')
    plt.xlim([0,250])
    #plt.ylim(3000,1500)
    plt.ylim([3000,0.01])
    #plt.ylim([s_low,3000])
    plt.yscale('log')
    plt.colorbar()
    #plt.ylim([1,0])
    plt.ylabel(r'Grain Size [$\mu m$]', fontsize = 16)
    plt.xlabel('Stellar distance [au]', fontsize = 16)

    #ax.tick_params(which = 'major', bottom = True, top = True, left = True, right = True,direction = 'inout' )
    ax.grid(which = 'major', linestyle = ':')##Decomment after this once I have fixed i
    #ax.yaxis.set_major_formatter(adj_log)
    ax.yaxis.set_major_formatter(formatter)


    title = 'Size_spatdist_timestep_'+str(Tpr)+'_q_'+str(round(q_mcmc[0],3))+'.pdf'
    plt.savefig(title)


    df_rbins_smin = df_rbins_smin.transpose()  #transpose the dataframe diagonally

    #




    #Index closest to 1 micron
    diff_array1 = np.absolute(s_gs - 1)
    ind1 = diff_array1.argmin()
    #Index closest to 10 micron
    diff_array10 = np.absolute(s_gs - 10)
    ind10 = diff_array1.argmin()


    #sum probability density values
    df_rbins_smin['Sum'] = df_rbins_smin.iloc[:,1:].sum(axis=1)
    df_rbins_smin['Small'] = df_rbins_smin.iloc[0:ind1,1:].sum(axis=1)
    df_rbins_smin['Medium'] = df_rbins_smin.iloc[ind1:ind10,1:].sum(axis=1)
    df_rbins_smin['Large'] = df_rbins_smin.iloc[ind10:,1:].sum(axis=1)


    np.savetxt("Number_Density"+object+"_"+composition+".csv", df_rbins_smin['Sum'], delimiter = ",")





    plt.clf()
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(rbins_bm,np.max(df_rbins_smin['Sum'])*f_gauss(rbins_bm)/np.max(f_gauss(rbins_bm)),'b-.',label = 'Belt')
    ax.plot(rbins_bm,df_rbins_smin['Sum'], 'k-', label = 'All dust grains')
    #ax.plot(rbins_bm,df_rbins_smin['Small'], 'm:', label = 'Small')
    #ax.plot(rbins_bm,df_rbins_smin['Medium'], 'g:', label = 'Medium')
    #ax.plot(rbins_bm,df_rbins_smin['Large'], 'r:', label = 'Large')

    ax.set_xlim([0,300])
    ax.set_ylabel('Number Density [N/au]')
    ax.set_xlabel('Stellar Distance [au]')
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.legend()
    plt.savefig('Numberdensity_distance_'+object+'_'+composition+'.pdf')


#    #Extended DustyMM def/function
#    df_fluxsmbg,df_rbins_dNg, df_rbins_dMg = DustyMM_ext(sm_mcmc[0],smax_r,dfrac_mcmc[0],q_mcmc[0],rmean,rwidth,rin,rout)
#
#    #print(df_fluxsmbg)
#
##    WRange = [find_nearest(wr,10), find_nearest(wr,70),find_nearest(wr,100),find_nearest(wr,160),find_nearest(wr,1300)]
#    WColour = ['m','k','b','g','r']
#    Ci = 0
#
#    plt.clf()
#    fig, ax = plt.subplots(nrows = 1, ncols = 1)
#
#    f_flux = FilterFunc(df_fluxsmbg,fMIRI_F100wr)
#    ax.plot(rbins_bm,f_flux(rbins_bm),str(WColour[Ci])+'-',label = 'JWST/MIRI-1000W')
#    Ci = Ci + 1
#    f_flux = FilterFunc(df_fluxsmbg,fMIRI_F210wr)
#    ax.plot(rbins_bm,f_flux(rbins_bm),str(WColour[Ci])+'-',label = 'JWST/MIRI-2100W')
#    Ci = Ci + 1
#
#    ax.set_xlim([1,300])
#    #ax.set_ylim([10**-5,10**-2])
##    ax.set_ylim([0,1.3])
##    ax.set_ylabel('Relative Flux Density [au$^{-1}$]')
#    ax.set_ylabel('Relative Flux Density [Jy au$^{-1}$]')
#    ax.set_xlabel('Stellar Distance [au]')
##    ax.set_xscale('log')
#    ax.set_yscale('log')
#    ax.legend()
#    ax.xaxis.set_major_formatter(formatter)
#    ax.yaxis.set_major_formatter(formatter)
#    plt.savefig('Flux_distance_JWST_'+object+'_'+composition+'.pdf')
#
#
#    plt.clf()
#    fig, ax = plt.subplots(nrows = 1, ncols = 1)
#
#    f_flux = FilterFunc(df_fluxsmbg,fFBlue70wr)
#    ax.plot(rbins_bm,f_flux(rbins_bm),str(WColour[Ci])+'-',label = 'Herschel/PACS_70')
#    Ci = Ci + 1
#    f_flux = FilterFunc(df_fluxsmbg,fFGreen100wr)
#    ax.plot(rbins_bm,f_flux(rbins_bm),str(WColour[Ci])+'-',label = 'Herschel/PACS_100')
#    Ci = Ci + 1
#    f_flux = FilterFunc(df_fluxsmbg,fFRed160wr)
#    ax.plot(rbins_bm,f_flux(rbins_bm),str(WColour[Ci])+'-',label = 'Herschel/PACS_160')
#    Ci = Ci + 1
#
#
#    ax.set_xlim([1,300])
#    #ax.set_ylim([10**-5,10**-2])
##    ax.set_ylim([0,1.3])
##    ax.set_ylabel('Relative Flux Density [au$^{-1}$]')
#    ax.set_ylabel('Relative Flux Density [Jy au$^{-1}$]')
#    ax.set_xlabel('Stellar Distance [au]')
##    ax.set_xscale('log')
#    ax.set_yscale('log')
#    ax.legend()
#    ax.xaxis.set_major_formatter(formatter)
#    #ax.yaxis.set_major_formatter(formatter)
#    plt.savefig('Flux_distance_HerschelPACS_'+object+'_'+composition+'.pdf')
    
    
    #Plot relative flux using filter information
    print('Determine Filter Values for MIRI-100 Filter band...')
#    MIRI_10 = DustyMM_fdist(sm_mcmc[0],smax_r,dfrac_mcmc[0],q_mcmc[0],rmean,rwidth,rin,rout,fMIRI_F100wr)
    MIRI_10 = DustyMM_ext(sm_mcmc[0],smax_r,dfrac_mcmc[0],q_mcmc[0],rmean,rwidth,rin,rout,fMIRI_F100wr)
    print('Determine Filter Values for MIRI-210 Filter band...')
#    MIRI_21 = DustyMM_fdist(sm_mcmc[0],smax_r,dfrac_mcmc[0],q_mcmc[0],rmean,rwidth,rin,rout,fMIRI_F210wr)
    MIRI_21 = DustyMM_ext(sm_mcmc[0],smax_r,dfrac_mcmc[0],q_mcmc[0],rmean,rwidth,rin,rout,fMIRI_F210wr)


    
    
    plt.clf()
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    ax.plot(rbins_bm,MIRI_10,'m:', label = 'JWST/MIRI-1000W')
    ax.plot(rbins_bm,MIRI_21,'k-.', label = 'JWST/MIRI-2100W')

    
    ax.set_xlim([1,200])
    #ax.set_ylim([10**-5,10**-2])
#    ax.set_ylim([0,1.3])
    ax.set_ylabel('Relative Flux Density [Jy au$^{-1}$]')
    ax.set_xlabel('Stellar Distance [au]')
#    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    plt.savefig('Flux_Filter_Distance_'+object+'_'+composition+'_MIRI.pdf')
    
    
    

    print('Determine Filter Values for Herschel/PACS 70 Filter band...')
#    PACS_70 = DustyMM_fdist(sm_mcmc[0],smax_r,dfrac_mcmc[0],q_mcmc[0],rmean,rwidth,rin,rout,fFBlue70wr)
    PACS_70 = DustyMM_ext(sm_mcmc[0],smax_r,dfrac_mcmc[0],q_mcmc[0],rmean,rwidth,rin,rout,fFBlue70wr)
    
    print('Determine Filter Values for Herschel/PACS 100 Filter band...')
#    PACS_100 = DustyMM_fdist(sm_mcmc[0],smax_r,dfrac_mcmc[0],q_mcmc[0],rmean,rwidth,rin,rout,fFGreen100wr)
    PACS_100 = DustyMM_ext(sm_mcmc[0],smax_r,dfrac_mcmc[0],q_mcmc[0],rmean,rwidth,rin,rout,fFGreen100wr)
    
    print('Determine Filter Values for Herschel/PACS 70 Filter band...')
#    PACS_160 = DustyMM_fdist(sm_mcmc[0],smax_r,dfrac_mcmc[0],q_mcmc[0],rmean,rwidth,rin,rout,fFRed160wr)
    PACS_160 = DustyMM_ext(sm_mcmc[0],smax_r,dfrac_mcmc[0],q_mcmc[0],rmean,rwidth,rin,rout,fFRed160wr)
    
    PACS_max = [np.max(PACS_70),np.max(PACS_100),np.max(PACS_160)]
    
    plt.clf()
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    
    ax.plot(rbins_bm,PACS_70,'b', label = 'Herschel/PACS_70')
    ax.plot(rbins_bm,PACS_100,'g', label = 'Herschel/PACS_100')
    ax.plot(rbins_bm,PACS_160,'r', label = 'Herschel/PACS_160')


    ax.set_xlim([1,200])
#    ax.set_ylim([10**-4,10**-1])
#    ax.set_ylim([np.floor(np.log10(PACS_max)),np.ceil(np.log10(PACS_max))])
    ax.set_ylabel('Relative Flux Density [Jy au$^{-1}$]')
    ax.set_xlabel('Stellar Distance [au]')
#    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    plt.savefig('Flux_Filter_Distance_'+object+'_'+composition+'_PACS.pdf')
    
    PACS_70_peakloc = rbins_bm[PACS_70.argmax()]
    PACS_100_peakloc = rbins_bm[PACS_100.argmax()]
    PACS_160_peakloc = rbins_bm[PACS_160.argmax()]


    f = open("DModel_FluxPeak.txt","w+")
    f.write("Peak Flux Values for %s \n" % (object))
    f.write("PACS_70 | Total Flux: %s Jy , Peak: %s au \n" % (np.sum(PACS_70), PACS_70_peakloc))
    f.write("PACS_100 | Total Flux: %s Jy , Peak: %s au \n" % (np.sum(PACS_100), PACS_100_peakloc))
    f.write("PACS_160 | Total Flux: %s Jy , Peak: %s au \n" % (np.sum(PACS_160), PACS_160_peakloc))
    
    f.close()
    
  
#save run
#fname = "testrun.h5"
#emcee.backends.HDFBackend(fname)


t1 = time.time()
t = round((t1 - t0)/60,4)
print(f'Total Time: {t} minutes')



