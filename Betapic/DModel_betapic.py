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
    #Luminosity as function of wavelength [SI Units]
    
    return L

#-----------------------------------------------------------------#
#Main Dusty Model code
#Using dataframes: easier to sort out grain regimes
def BetaGrains(smin,smax,dfrac,q,rm,rw,rin,rout,rho,no_s,hd_bs,TGrid,Qpr,QscaG,QabsG,wr):
#def BetaGrains(s_gs,r_belt,f_belt,rbins_bm,rbins_bins,q,df_Qpr,df_Qabs,wr):
#def BetaGrains(smin,smax,dfrac,q,rmean,rw,rin,rout,rho,no_s,hd_bs,TGrid,Qpr,QscaG,QabsG,wr): #minimum grain size, max grain size, disc frac [Me], q, rmean, rsig, rin, rou, rho,no_s,bin size,  [au], wavelength range
    
    ##Planestimal Belt Characteristics##
    ##Total Grain Mass##
    DiscMass = dfrac*Me*10**3 #Total grain mass as a fraction of Earth Mass (converted to grams) because denisty(rho) is given in g/cm^3
    print(f'Mass of Disc: {DiscMass} g') #inform human

    #create grain size space where most grains are towards the smaller size (logarithmic spaced in linear fashion)
    s_gs = np.geomspace(smin,smax,no_s)
    print(f'Number of Grain Sizes: {len(s_gs)} ranging from {s_gs[0]} to {s_gs[-1]} microns')

    ##defining the timestep from largest grain (or planetesimal releasing grains???)
    nts = 3600 #number of timesteps per orbit
    no = 1 #number of orbits
    dt = rm/nts #timestep for all grains

    #Planetesimal density distribution model#
    #These models will be created regardless
    #This will be used to determine where all grains are or where they are released on for planetesimal orbit
    #Release point (initial at Keplerian velocity) / #note: intial location of grain or shortest distance for grain | also could be considered the radius of the circular orbit with radiation forces present
    #Determine radial values for belt
    r_steps = np.int64((rout-rin)/hd_bs + 1) #steps will be same size as rbins_bm values
    r_belt = np.linspace(rin, rout, r_steps) #radial parametric space for belt

    ##Planetesimal Belt Model##
    print(f'Mean distance of grain release: {rm} au')

    ddgauss = []
    for r in rbins_bm:
        if r >= rin and r <= rout: #inside belt create belt value at rbin_bm location
            ddgauss.append(np.exp(-0.5*((r - rm)/rw)**2))
        else:
            ddgauss.append(0)

    #Normalise models to give a total sum of radial distribution to be unity
    ddgauss = ddgauss/(np.sum(ddgauss))
    #Final probability distribution value of planetesimals
    f_belt = interpolate.interp1d(rbins_bm,ddgauss)
    
    gr_blowout = [0]
    grainsizes = []
    #Create dataframes
    #gravitation forces only
    df_rbins_p = pd.DataFrame({'R': rbins_bm}) #probability of grains of size (s) to each bin: (p)
    df_rbins_dN = pd.DataFrame({'R': rbins_bm}) #number of grains of size (s) for a given a power law exponent (q) in each bin: (dN)
    df_rbins_dM = pd.DataFrame({'R': rbins_bm}) #mass of grains of size (s) in each bin: (dM)
 
    #beta activated
    df_rbins_pb = pd.DataFrame({'R': rbins_bm}) #probability of grains of size (s) to each bin: (p)
    df_rbins_dNb = pd.DataFrame({'R': rbins_bm}) #number of grains of size (s) for a given a power law exponent (q) in each bin: (dN)
    df_rbins_dMb = pd.DataFrame({'R': rbins_bm}) #mass of grains of size (s) in each bin: (dM)

    #Dataframe for flux values corresponding to each model
    df_fluxsm = pd.DataFrame({'Wavelength (um)': wr}) #Wavelength values
    df_fluxsm['Frequency (Hz)'] = c/(df_fluxsm['Wavelength (um)']*1e-6)#*2.99792458e+14 |frequency values
    df_fluxsmb = pd.DataFrame({'Wavelength (um)': wr}) #Wavelength values
    df_fluxsmb['Frequency (Hz)'] = c/(df_fluxsm['Wavelength (um)']*1e-6)#*2.99792458e+14 |frequency values
    
    #Y-column names intiate
    y_values_nb = []
    y_values_b = []

    ##grain size distribution dN \propto s^-q ds #Power law mass distribution + pre-fill dataframes for later summing of angle release
    dMs = []
    

    for s in s_gs:
        dMs.append((s*1e-4)**-q*rho*(4/3)*pi*(s*1e-4)**3) #Note: s is converted to cm to match 'rho' and have mass in grams
        s_colname = str(round(s,4)) #grain size column name
        #Setting up initial values in dataframes
        #Grav only
        df_rbins_p[s_colname] = 0
        df_rbins_dN[s_colname] = 0
        df_rbins_dM[s_colname] = 0

        #Beta model
        df_rbins_pb[s_colname] = 0
        df_rbins_dNb[s_colname] = 0
        df_rbins_dMb[s_colname] = 0
    

    ##Determine Constant to find exact number and mass of grains according to size##
    Msum = np.sum(dMs)      #unscaled mass component
    dNc = DiscMass/Msum    #finding constant
    print(f'Proportional Constant for dN proto s**-q: {dNc} where grain size (s) in units of cm')

    #Iterate through grain size to determine dust distribition (future: may turn this into a 'def' function to create multiple rings)
    #Determine location of sized grains that are "produced" by a planetesimal on a circular orbit
    print(f'Starting interating through grain size to determine dust distribution and corresponding SED...')
    for s in s_gs:
        t_1 = time.time()
        #grain size column name
        s_colname = str(round(s,4))
        y_values_nb.append(str(s_colname)) #store y value names in dataframe for plotting later
        y_values_b.append(str(s_colname)) #stort y values + sum values later

        #Setting initial value grain column
        #Grav
        df_rbins_p[s_colname] = 0
        df_rbins_dN[s_colname] = 0
        df_rbins_dM[s_colname] = 0

        #Beta model
        df_rbins_pb[s_colname] = 0
        df_rbins_dNb[s_colname] = 0
        df_rbins_dMb[s_colname] = 0

        for rg in r_belt:
            #Calculate heliocentric distances for models with non-radiation forcces
            P = rg**(3/2)            #Period for one orbit
            dtm = dt*(r_steps+1)          #increase size of timestep by the number of release locations to reduce computational time
            #Orbital Values for a bound grain at heliocentric distance (r) and size (s)
            [r1,th1,ti1] = orbit(rg,rg,0,P,dtm)
#            e = 0
#            a = rg
#            r1 = a*(2-(1-e**2)/(1 + e*np.cos(thr)))
            
            #binning heliocentric (orbital) distances in binsizes = 1 au and finding "density" or normalised values
            [den,bins] = np.histogram(r1,bins = rbins_bins,density=True)
                
            #Gaussian distribution of grains
            ngfs = f_belt(rg) #fraction of grains on radial release location for gaussian distribution
            dN = (dNc*(s*1e-4)**-q)*ngfs  #number of grains corresponding to grain size release point (corresponding to Guassian distribution)
            #Fill DataFrames: Probablity, Number of Grains, Mass of Grain in each bin
            df_rbins_p[s_colname] = df_rbins_p[s_colname] + den #fill column with "density" values from histogram
            df_rbins_dN[s_colname] = df_rbins_dN[s_colname] + dN*den#  #relative number of grains corresponding to the grain size, q and distance
            df_rbins_dM[s_colname] = df_rbins_dM[s_colname] + dN*rho*(4/3)*pi*(s*1e-4)**3*den # #mass of grain in bin (assuming spherical grains)
            #Power law distribution of grains


           
            #Beta-grains model
            #Determine beta-value | considering radiation pressure on grains
            be = 0.574*Ls*Qpr_s(s)/(Ms*rho*s) #beta-value in solar units.  Density of grain in g/cm^3
             
            if be == 0.5:
                #blow out limit
                print(f'Blowout grain size: {a}')
                #ignores grain
            if be > 0.5:
                print(f'Grains with size: {s} micron & beta: {be} at a distance: {rg} au are on unbound orbits are ignored, i.e. blown out of system (be > 0.5)')
                if gr_blowout[-1] != s:
                    gr_blowout.append(s)
            if be < 0:
                print(f'something is screwed (be < 0)')
            elif be >= 0 and be < 0.5:
                #Bound grains
                grainsizes.append(s)
                #Orbital values for 0 <= be < 0.5
                #print(f'Grain size: {a} is on a bound orbit with be: {be} (0 < be < 0.5)')
                e = be/(1-be)           #relation between eccentricity & beta - source: Augereau & Beust (2006) - On the AU Microscopii debris disk (A&B2006)
                a = rg/(1-e)#semi-major axis being release at r(theta)
                b = a*math.sqrt(1-e**2) #semi-minor axis
                P = a**(3/2)            #Period for one orbit
                dtm = dt*(r_steps*s+1)          #increase size of timestep by the number of release locations x grain size to reduce computational time
                #Orbital Values for bound grains
                [r1,th1,ti1] = orbit(a,b,e,P,dtm)
#                r1 = a*(2-(1-e**2)/(1 + e*np.cos(thr)))
                
                #binning heliocentric (orbital) distances in binsizes = 1 au and finding "density" or normalised values
                [den,bins] = np.histogram(r1,bins = rbins_bins,density=True)
                
               
                #Fill DataFrames: Probablity, Number of Grains, Mass of Grain in each bin
                df_rbins_pb[s_colname] = df_rbins_pb[s_colname] + den #fill column with density distribution values from histogram
                df_rbins_dNb[s_colname] = df_rbins_dNb[s_colname] + dN*den#  #relative number of grains corresponding to the grain size, q and distance
                df_rbins_dMb[s_colname] = df_rbins_dMb[s_colname] + dN*rho*(4/3)*pi*(s*1e-4)**3*den # #mass of grain in bin (assuming spherical grains)
    
    #Calculate Albedo
    
    
    #Calculating Flux Vales (SEDs)
    print(f'Calculating SED Values...')
    for s in s_gs:
        s_colname = str(round(s,4))
        #print(s)
        for i in range(len(rbins_bm)):
            if df_rbins_dN.at[i,s_colname] > 0:
                #Create column name based on radius and grain size to fill in the flux values at r
                fcolname = 'Flux_'+str(round(s,3))+'_'+str(rbins_bm[i])
                #GTg = TGrid[s,rbins_bm[i]]      #Grain Temperture at r
                GTg = Tg_sr(s,rbins_bm[i])
                dNR = df_rbins_dN.at[i,s_colname]    #Number of grains at r
                BBfdm = Blam(GTg,wr*10**-6)  #Planck function
                #Alam = QscaG[s,1:]/(QscaG[s,1:]+QabsG[s,1:])
                Flam = 8*pi**2*dNR*(s*10**-6)**3*(dpc*pc)**-2*QabsG(s,wr).flatten()*BBfdm*(rbins_bm[i]/hd_bs) #Thermal Emission
                #Flsca = dNR*((Rs*R_s)/(dpc*pc))**2*pi*Blam(Ts,wr*10**-6)*Alam*QscaG[s,1:]*(s*10**-6/(2*rbins_bm[i]*au))**2 #Scattered Emission
                #Flamtot = Flam + Flsca
                Flamtot = Flam
                Fnu = 10**26*Flamtot*((wr*10**-6)**2)/(c)
                df_fluxsm[fcolname] = Fnu

                
            if df_rbins_dNb.at[i,s_colname] > 0:
                fcolname = 'Flux_'+str(round(s,3))+'_'+str(rbins_bm[i])
                #GTg = df_rbins_Tg.at[i,s_colname]        #Grain Temperture at r
                GTg = Tg_sr(s,rbins_bm[i])
                dNR = df_rbins_dNb.at[i,s_colname]       #Number of grains at r
                BBf = Blam(GTg,wr*10**-6)
#                print(np.max(BBf))
#                print(zed)
#                Alam = QscaG[s,1:]/(QscaG[s,1:]+QabsG[s,1:]) #Albedo
                Flamb = 8*pi**2*dNR*(s*10**-6)**3*(dpc*pc)**-2*QabsG(s,wr).flatten()*BBf*(rbins_bm[i]/hd_bs)  #Thermal Emission
                #Flscab = dNR*((Rs*R_s)/(dpc*pc))**2*pi*Blam(Ts,wr*10**-6)*Alam*QscaG[s,1:]*(s*10**-6/(2*rbins_bm[i]*au))**2 #Scattered Emission
                #Flamtotb = Flamb + Flscab
                Flamtotb = Flamb
                Fnub = 10**26*Flamtotb*((wr*10**-6)**2)/(c)
                df_fluxsmb[fcolname] =  Fnub
                y_values_b.append(fcolname)
    
#        plt.plot(wr,df_fluxsmb[fcolname])
#        plt.xscale('log')
#        plt.yscale('log')
#        plt.ylim([10**-10,10**-4.5])
#        plt.xlim([10,1000])
#        plt.savefig('Old Code Test.eps')
#        print(zed)
        

    #Summing values
    #Grav only
    #Mass
    dMSum = np.sum(df_rbins_dM.iloc[:,1:].sum(axis=1))
    print(f'Total Disc Mass | Grav Only Model (dmSum): {dMSum} g ')
    MPer = round((dMSum/DiscMass)*100,3)
    print(f'Fraction of Mass for dM values (all au): {MPer}%')
    df_rbins_dM['Sum'] = df_rbins_dM.iloc[:,1:].sum(axis=1)
    #Number
    df_rbins_dN['Sum'] = df_rbins_dN.iloc[:,1:].sum(axis=1)
    print(np.sum(df_rbins_dN['Sum']))
    #Flux
    y_values_nb.append('Sum')
    df_fluxsm['Sum'] = df_fluxsm.iloc[:,2:].sum(axis=1)

    #Beta model
    #Mass
    dMSumb = np.sum(df_rbins_dMb.iloc[:,1:].sum(axis=1))
    print(f'Total Disc Mass | Beta Model (dmSum): {dMSum} g ')
    MPerb = round((dMSumb/DiscMass)*100,3)
    print(f'Fraction of Mass for dM values (all au): {MPerb}%')
    df_rbins_dMb['Sum'] = df_rbins_dMb.iloc[:,1:].sum(axis=1)
    #Number
    df_rbins_dNb['Sum'] = df_rbins_dNb.iloc[:,1:].sum(axis=1)
    print(np.sum(df_rbins_dNb['Sum']))
    #Flux
    y_values_b.append('Sum')
    df_fluxsmb['Sum'] = df_fluxsmb.iloc[:,2:].sum(axis=1)

    #Fraction of disc mass
    dfracf = dfrac*MPerb/100
    
    disc_func = interpolate.interp1d(wr,df_fluxsmb['Sum'].to_numpy(), fill_value = 'extrapolate')
    #df_fluxdb = disc_func(wa)
    
    SED_total_nb = np.add(df_fluxsm['Sum'].to_numpy(),flux_sa)
    SED_total = np.add(df_fluxsmb['Sum'].to_numpy(),flux_sa)
#    SED_total_nb = np.add(df_fluxsm['Sum'].to_numpy(),flux_s)
#    SED_total = np.add(df_fluxsmb['Sum'].to_numpy(),flux_s)


    return [df_rbins_p, df_rbins_pb, df_rbins_dN, df_rbins_dNb, df_rbins_dM,df_rbins_dMb, df_fluxsm, df_fluxsmb, y_values_nb, y_values_b, grainsizes,gr_blowout,MPerb,SED_total,SED_total_nb,df_fluxsmb['Sum'].to_numpy(),df_fluxsm['Sum'].to_numpy()]

#-----------------------------------------------------------------#
#Dust Migration model
#Revamping the original code to remove the reliance on dataframes
#Simply return a SED flux
def DustyMM(smin,smax,dfrac,q,rm,rw,rin,rout):
    ##Planestimal Belt Characteristics##
    print(f'Min. Grain Size: {smin}, dfrac: {dfrac}, q: {q}, rmean: {rm}, rsigma: {rw}')
    ##Total Grain Mass##
    DiscMass = dfrac*Me*10**3 #Total grain mass as a fraction of Earth Mass (converted to grams) because denisty(rho) is given in g/cm^3
#    print(f'Mass of Disc: {DiscMass} g') #inform human

    #Number of grain sizes
    no_s = 100

    #create grain size space where most grains are towards the smaller size (logarithmic spaced in linear fashion)
    s_gs = np.geomspace(smin,smax,no_s)
#    print(f'Number of Grain Sizes: {len(s_gs)} ranging from {s_gs[0]} to {s_gs[-1]} microns')

    ##defining the timestep from largest grain (or planetesimal releasing grains???)
    nts = 3600 #number of timesteps per orbit
    no = 1 #number of orbits
    dt = rm/nts #timestep for all grains

    #Planetesimal density distribution model#
    #These models will be created regardless
    #This will be used to determine where all grains are or where they are released on for planetesimal orbit
    #Release point (initial at Keplerian velocity) / #note: intial location of grain or shortest distance for grain | also could be considered the radius of the circular orbit with radiation forces present
    #Determine radial values for belt
    r_steps = np.int64((rout-rin)/hd_bs + 1) #steps will be same size as rbins_bm values
    r_belt = np.linspace(rin, rout, r_steps) #radial parametric space for belt

    ##Planetesimal Belt Model##
    #print(f'Mean distance of grain release: {rm} au')

    ddgauss = []
    for r in rbins_bm:
        if r >= rin and r <= rout: #inside belt create belt value at rbin_bm location
            ddgauss.append(np.exp(-0.5*((r - rm)/rw)**2))
        else:
            ddgauss.append(0)

    #Normalise models to give a total sum of radial distribution to be unity
    ddgauss = ddgauss/(np.sum(ddgauss))
    #Final probability distribution value of planetesimals
    f_gauss = interpolate.interp1d(rbins_bm,ddgauss)
    
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
    dMs = (s_gs*1e-4)**-q*rho*(4/3)*pi*(s_gs*1e-4)**3
    Msum = np.sum(dMs)      #unscaled mass component
    dNc = DiscMass/Msum    #finding constant
    #print(f'Proportional Constant for dN proto s**-q: {dNc} where grain size (s) in units of cm')

    for s in s_gs:
        
#        for rg in r_belt:
        #Calculate heliocentric distances for models with non-radiation forcces
#            P = rg**(3/2)                 #Period for one circular orbit | under the assumption planetesimals orbit on circular orbits (eccentric orbits to be implemented)
#            dtm = dt*(r_steps+1)          #increase size of timestep by the number of release locations to reduce computational time
        #Orbital Values for a bound grain at heliocentric distance (r) and size (s)
        #[r1,th1,ti1] = orbit(rg,rg,0,P,dtm)
#            r1 = f_BeRad(0,BeIn) #note: you need to multiply the r1 by rg in the histrogram function
#        e = 0
#        a = r_belt/(1-e)
##        r1 = (2-(1-e**2)/(1 + e*np.cos(thr)))
##        a1 = np.reshape(r_belt,(len(r_belt),1))
##        rnb = np.ravel(a1*r1)
##
##
##        #binning heliocentric (orbital) distances in binsizes = 1 au and finding "density" or normalised values
##        [den,bins] = np.histogram(rnb,bins = rbins_bins,density=True)
#
#        rb = (2-(1-e**2)/(1 + e*np.cos(thr)))
##            ab = np.reshape(a,(len(a),1))
##            rb = ab*rb
#
#        den = np.zeros(len(rbins_bm))
#        #print(rb)
#        for i in range(len(r_belt)):
#        #binning heliocentric (orbital) distances in binsizes = 1 au and finding "density" or normalised values
#            [denb,bins] = np.histogram(rb*a[i],bins = rbins_bins,density=True)
#            den = den + denb*f_gauss(r_belt[i])
        
#
#        plt.plot(rbins_bm,den)
#        plt.show()
#        print(zed)
        
#        #Gaussian distribution of grains
#        #ngfs = f_gauss(rg) #fraction of grains on radial release location for gaussian distribution
##        ngfs = f_gauss(r_belt) #fraction of grains on radial release location for gaussian distribution
#        dNnb = den*(dNc*(s*1e-4)**-q)#*ngfs  #number of grains corresponding to grain size release point (corresponding to Guassian distribution)
#        #Summing mass values
#        dM_radial_nb = dM_radial_nb + np.sum(dNnb*rho*(4/3)*pi*(s*1e-4)**3) #summing total mass
#
#        dNnb = np.trim_zeros(dNnb)
#
#
#        glocID = den/den                        #Mask for grain locations
#        glocID[np.isnan(glocID)] = 0            #Remove NaN
#
#        gloc = np.trim_zeros(rbins_bm*glocID)   #radial locations where grains of size (s) are present
#        GTg = Tg_sr(s,gloc).flatten()           #Grain temperatures as a function of radial locations
#        BBfns = f_BBgT(GTg,wr)                  #Blackbody functions as a function of grain temperatures
#        BBfns = np.multiply(BBfns,dNnb*gloc)    #multiply blackbody fns by corresponding number density * radial distance
#        BBfns_sum = np.sum(BBfns, axis = 1)     #sum all BB functions together
#
#        #Thermal Emission
#        Flam = BBfns_sum*32*pi**3*(s*10**-6)**3*(dpc*pc)**-2*Qabs_sw(s,wr).flatten()/hd_bs
#
##                #Scattered light
##                Alam = Qsca_sw(s,1:)/(Qsca_sw(s,1:)+Qabs_sw(s,1:)) #Albedo for scattered light
##                Flsca = dNb*((Rs*R_s)/(dpc*pc))**2*pi*Blam(Ts,wr*10**-6)*Alam*Qsca_sw(s,1:)*(s*10**-6/(2*rg*au))**2 #Scattered Emission
##
#        #Summing Fluxes
##                Flamtot = Flam + Flsca #Total flux as a function of wavelength
##                Fnu = 10**26*Flamtot*((wr*10**-6)**2)/(c) #Total flux as a function of frequency [Jankys]
#        Fnu = 10**26*Flam*((wr*10**-6)**2)/(c) #Total flux as a function of frequency [Jankys]
#        flux_nb = flux_nb + Fnu #Add flux
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
            print(f'something is screwed (be < 0)')
        elif be >= 0 and be < 0.5:
            #Bound grains
            #Orbital values for 0 <= be < 0.5
            #print(f'Grain size: {a} is on a bound orbit with be: {be} (0 < be < 0.5)')
            e = be/(1-be)           #relation between eccentricity & beta - source: Augereau & Beust (2006) - On the AU Microscopii debris disk (A&B2006)
            a = r_belt/(1-e)            #semi-major axis being release at r(theta)
#            b = a*math.sqrt(1-e**2) #semi-minor axis
#            P = a**(3/2)            #Period for one orbit
#            dtm = dt*(r_steps*s+1)/5  #timestep

            #Orbital Values for bound grains
            #[r1,th1,ti1] = orbit(a,b,e,P,dtm)
            #r1 = f_BeRad(be,BeIn)
            rb = (2-(1-e**2)/(1 + e*np.cos(thr)))

            
            dNb = np.zeros(len(rbins_bm))
            #print(rb)
            for i in range(len(r_belt)):
            #binning heliocentric (orbital) distances in binsizes = 1 au and finding "density" or normalised values
#                [denb,bins] = np.histogram(rb*a[i],bins = rbins_bins,density=True)
                denb = histogram1d(rb*a[i], range = [rbins_bins[0],rbins_bins[-1]], bins=len(rbins_bins)-1)
                dNb = dNb + (denb/np.sum(denb))*f_gauss(r_belt[i])*(dNc*(s*1e-4)**-q)
            
#            dNb_apert = dNb[0:rmax]
#            dNbr = np.trim_zeros(dNb_apert)  #remove zeroes from array
            dNbr = np.trim_zeros(dNb)  #remove zeroes from array

            #Summing mass values
            dM_radial_b = dM_radial_b + np.sum(dNb*rho*(4/3)*pi*(s*1e-4)**3)
            
            glocID = dNb/dNb                        #Mask for grain locations
            glocID[np.isnan(glocID)] = 0            #Remove NaN
            
            gloc = np.trim_zeros(rbins_bm*glocID)   #radial locations where grains of size (s) are present
            GTg = Tg_sr(s,gloc).flatten()           #Grain temperatures as a function of radial locations
            BBfns = f_BBgT(GTg,wr)
            
            #BBfns = np.multiply(np.multiply(BBfns,Qabs_sw(s,wr)),np.multiply(dNbr,gloc))    #multiply blackbody fns by corresponding number density * radial distance
            BBfns = np.multiply(np.multiply(BBfns,Qabs_sw(s,wr)),np.multiply(dNbr[::-1],gloc[::-1]))    #multiply blackbody fns by corresponding number density * radial distance
                            #BBfns = np.multiply(BBfns,dNbr)    #multiply blackbody fns by corresponding number density * radial distance
            BBfns_sum = np.sum(BBfns, axis = 1)     #sum all BB functions together
            
           
            #Thermal Emission
            Flam = BBfns_sum*8*pi**2*(s*10**-6)**3*(dpc*pc)**-2/hd_bs
      
#                #Scattered light
#                Alam = Qsca_sw(s,1:)/(Qsca_sw(s,1:)+Qabs_sw(s,1:)) #Albedo for scattered light
#                Flsca = dNb*((Rs*R_s)/(dpc*pc))**2*pi*Blam(Ts,wr*10**-6)*Alam*Qsca_sw(s,1:)*(s*10**-6/(2*rg*au))**2 #Scattered Emission
#
            #Summing Fluxes
#                Flamtot = Flam + Flsca #Total flux as a function of wavelength
#                Fnu = 10**26*Flamtot*((wr*10**-6)**2)/(c) #Total flux as a function of frequency [Jankys]
            Fnu = 10**26*Flam*((wr*10**-6)**2)/(c) #Total flux as a function of frequency [Jankys]


#
            flux_b = flux_b + Fnu #Add flux
        
#        plt.plot(wr,flux_b)
#        plt.xscale('log')
#        plt.yscale('log')
#        plt.ylim([10**-10,10**-4.5])
#        plt.xlim([10,1000])
#        plt.savefig('New Code Test.eps')
#        print(zed)


    
    
    
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
#    MPerb = round((dM_radial_b/DiscMass)*100,3)
#    #print(f'Fraction of Mass for dM values (all au): {MPerb}%')
#
#    SED_total = np.add(flux_b,flux_nu)
##    SED_totalnb = np.add(flux_nb,flux_sa)

    
    #return SED_total,flux_b,MPerb #returns SED Total for given set of initial conditions
#    return SED_total
    return flux_b
#-----------------------------------------------------------------#



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

    #flux for radiation pressure co-efficients
    f_10spek = fun_spek(np.log10(wrange))
    flux_sa = 10**(f_10spek)
    BB_star = Blam(Ts,wrange*10**-6)
    BStar = np.max(BB_star)*(flux_sa/np.max(flux_sa))
    
    
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

    return [c2]

#-----------------------------------------------------------------#
##emcee functions###
##Inlike() function for emcee
#def lnlike(phot_wav,phot_flux,phot_unc,func_wav,func_flux):
def lnlike(theta,phot_wav,phot_flux,phot_unc):
#    sm,dfrac,qv,rm,rw = theta
    sm,dfrac,qv,rm,rw, sm2,dfrac2,qv2,rm2,rw2 = theta
#    func_flux = DustyMM(sm,smax_r,dfrac,qv,rm,rw,rin,rout)
    func_flux1 = DustyMM(sm,smax_r,dfrac,qv,rm,rw,rin,rout)
    func_flux2 = DustyMM(sm2,smax_r,dfrac2,qv2,rm2,rw2,rin2,rout2)
    f_belts = np.add(func_flux1,func_flux2)
    func_flux = np.add(flux_nu,f_belts)
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
    sm,dfrac,qv,rm,rw, sm2,dfrac2,qv2,rm2,rw2 = theta
#    if 1.83 < sm < 20 and 0.001 < dfrac < 2.0 and 3.0 < q < 4.0 and 15 < rm < 150 and 10 < rw < 200:
    if 1.9 < sm < 20 and 0.001 < dfrac < 2.0 and 3.0 < qv < 4.0 and 60 < rm < 130 and 2 < rw < 110 and 1.9 < sm2 < 20 and 0.001 < dfrac2 < 2.0 and 3.0 < qv2 < 4.0 and 1 < rm2 < 15 and 1 < rw2 < 10:
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
object = 'Beta-pic'
#Stellar Photometric values for comparison

#lam = [12.0, 18.3,    23.67,  24.6,    25.00,   60,   70, 100.0,  160.0,  250.0,  350.0, 500.0,  850.0,1200]
#flx = [2.296,4.316,   7.847,  8.807,   10.2,    18.5, 16, 9.8,    5.1,  1.9,   0.72,   0.38, 0.058,0.036]
#unc = [0.2771, 0.432, 0.392, 0.881, 2.00, 3.7,  0.8, 1.058, 0.5, 0.1,  0.05,  0.03,  0.006,0.01]
#unc = 3*unc
#BB ~82K Laureijs et. al. 2002 | correction K(nu) for 12 micron IRAS ~ 1.48 | 3.395 K -> ~2.296 https://irsa.ipac.caltech.edu/IRASdocs/archives/colorcorr.html
##Uncertainty 3*per. rel. unc. 4% -> 12%
##Import Photometric values with warm component subtract
#Phot_bpic = np.loadtxt('df_betapic_cold.txt', dtype=[('wavelength', float),('flux', float),('uncertainty', float) ])
#lam = Phot_bpic["wavelength"]
#flx = Phot_bpic["flux"]
#unc = Phot_bpic["uncertainty"]
lam = [3.4, 4.6, 12.0, 18.3,    23.67,  24.6,    25.00,   60,   70, 100.0,  160.0,  250.0,  350.0, 500.0,  850.0,1200]
flx = [12.39,9.11, 2.66,4.316,   7.847,  8.807,   10.2,    18.5, 16, 9.8,    5.1,  1.9,   0.72,   0.38, 0.058,0.036]
unc = [5.06,2.35,0.2771, 0.432, 0.392, 0.881, 2.00, 3.7,  0.8, 1.058, 0.5, 0.1,  0.05,  0.03,  0.006,0.01]

##Host Star properties in Solar units##
Ls = 8.7         #Luminosity of Star [Solar Luminosity]
Ms = 1.75        #Mass of Star [Solar Masses]
Rs = 1.54          #Radius of Star [Solar Radius]
Ts = 8200       #star temperature [K]
dpc = 19.44       #distance to star system [pc]

##Star Flux values: Blackbody + Spectrum (if available)
##Blackbody Stellar spectrum (default)
bb = BBody(temperature=Ts*u.K)
d = dpc*pc   #distance to Star [m]
Rad = Rs*R_s   #Star Radius: Sun [m]
As = pi*(Rad/d)**2 #Amplitude of Star's blackbody planck fn (i.e. for this case: cross-section of Sun) and convert to Jansky (as BBody units are given in: erg / (cm2 Hz s sr))
flux_s = As*Blam(Ts,wr*10**-6)*10**26*((wr*10**-6)**2)/(c)

##Stellar Spectrum##
#Import Model Atmosphere values
f_spek = np.loadtxt('BetaPic-8200.txt', dtype=[('wave', float),('fspek', float)])

#Convert, Extrapolate and Calibrate with Blackbody Spectrum of similar temperature
wave = f_spek["wave"]*0.0001 #Convert from Angstroms to Microns
fspek = f_spek["fspek"]

#Extrapolate/Interplate F(wave)
fun_spek = interpolate.interp1d(np.log10(wave),np.log10(fspek),kind = 'linear', fill_value = 'extrapolate')

#Convert flux (wave) to flux (frequency)
sc_flux = 100.12
f_10spek = fun_spek(np.log10(wr))
flux_nu = (10**f_10spek)*((wr*10**-6)**2)/(c)  #unscaled flux (frequency)
flux_nu = sc_flux*flux_nu/np.max(flux_nu)

#plt.plot(lam,flx,'o')
#plt.plot(wr,flux_sa,'k')
#plt.xscale('log')
#plt.yscale('log')
#plt.ylim([0.001,1000])
#plt.xlim([0.3,3000])
#plt.show()


#print(len(f_spek2))
#print(len(10**(wr)))
#
#plt.plot(wr,flux_sa,'k',wr,f_spek2,'c')
##plt.plot(np.log10(wr),f_10spek)
#plt.xscale('log')
#plt.yscale('log')
#plt.show()
#print(zed)
#
#print(len(SFlux_flxwr))
#print(len(flux_s))
#print(len(flux_s[1281:3000])+len(SFlux_flxwr[0:1281]))
#

#
#print(zed)


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
#rho = 3.3 #grain density
rho = 2.0
###Grain Optical Constants
#composition = 'silicate_d03'
composition = 'diice_hypv2'
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
#Interpolate 2D grid space for grain size regime and radial distance values
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
rin = 5
rout = 200
rin2 = 1
rout2 = 10

##Emcee Inputs##
nwalkers = 25
niter = 220
#initial = np.array([1.85,1.5,3.5,90,50]) #Variables to be tested: sm,dfrac,qv,rm,rw
#initial = np.array([2.1,1.0,3.86,100,40,7.85,0.06,3.16,7.34,2.35]) #Variables to be tested: sm,dfrac,qv,rm,rw
initial = np.array([3.5,0.42,3.6,88.5,50,8.5,0.06,3.11,10.4,2.2]) #Variables to be tested: sm,dfrac,qv,rm,rw
ndim = len(initial)
#p0 = [np.array(initial) + np.random.randn(ndim) for i in range(nwalkers)]
#p0 = [np.array(initial) + np.array([np.random.randn(),1e-2*np.random.randn(),np.random.randn(),np.random.randn(),np.random.randn()]) for i in range(nwalkers)]
p0 = [np.array(initial) + np.array([random.uniform(-0.2,1),random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),random.uniform(-10,10),random.uniform(-10,10),random.uniform(-1,1),random.uniform(-0.01,0.01),random.uniform(-0.05,0.05),random.uniform(-1,1),random.uniform(-1,1)]) for i in range(nwalkers)]
print('Walkers for simulation:')
print(p0)
#print(zed)
data = lam,flx,unc


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

#labels = ['s_min','dfrac','q','R_m','R_sig']
labels = ['s_min','dfrac','q','R_m','R_sig','s_min2','dfrac2','q2','R_m2','R_sig2']
fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84])
cornerfig = object+composition+'_Emcee_corner_nwalkers_'+str(nwalkers)+'_niter'+str(niter)+'_2temp.eps'
#fig.savefig('Emcee_corner_30_500.eps')
fig.savefig(cornerfig)

#sm_mcmc, dfrac_mcmc, q_mcmc, rm_mcmc, rw_mcmc = np.median(samples, axis=0)
sm_mcmc, dfrac_mcmc, q_mcmc, rm_mcmc, rw_mcmc, sm_mcmc2, dfrac_mcmc2, q_mcmc2, rm_mcmc2, rw_mcmc2 = np.median(samples, axis=0)

#print(f's_min: {sm_mcmc}')
#print(f'dfrac: {dfrac_mcmc}')
#print(f'q: {q_mcmc}')
#print(f'r_mean: {rm_mcmc}')
#print(f'r_width: {rw_mcmc}')

#sm_mcmcup, dfrac_mcmcup, q_mcmcup, rm_mcmcup, rw_mcmcup = np.percentile(samples,[84], axis=0)
#sm_mcmclo, dfrac_mcmclo, q_mcmclo, rm_mcmclo, rw_mcmclo = np.percentile(samples,[16], axis=0)
#
#print(f's_min: {sm_mcmc} +{sm_mcmcup} -{sm_mcmclo}')
#print(f'dfrac: {dfrac_mcmc} +{dfrac_mcmcup} -{dfrac_mcmclo}')
#print(f'q: {q_mcmc} +{q_mcmcup} -{q_mcmclo}')
#print(f'r_mean: {rm_mcmc} +{rm_mcmcup} -{rm_mcmclo}')
#print(f'r_width: {rw_mcmc} +{rw_mcmcup} -{rw_mcmclo}')

#sampler[:, 2] = np.exp(sampler[:, 4])
#sm_mcmc, dfrac_mcmc, q_mcmc, rm_mcmc, rw_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84],axis=0)))
sm_mcmc, dfrac_mcmc, q_mcmc, rm_mcmc, rw_mcmc, sm_mcmc2, dfrac_mcmc2, q_mcmc2, rm_mcmc2, rw_mcmc2  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84],axis=0)))


print("s_min: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) mu".format(sm_mcmc[0],sm_mcmc[1], sm_mcmc[2]))
print("dfrac: {0:1.5f} (+{1:1.5f}, -{2:1.5f}) MEarth".format(dfrac_mcmc[0],dfrac_mcmc[1], dfrac_mcmc[2]))
print("q: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) [-]".format(q_mcmc[0],q_mcmc[1], q_mcmc[2]))
print("rmean: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) au".format(rm_mcmc[0],rm_mcmc[1], rm_mcmc[2]))
print("rwidth: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) au".format(rw_mcmc[0],rw_mcmc[1], rw_mcmc[2]))
print("s_min2: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) mu".format(sm_mcmc2[0],sm_mcmc2[1], sm_mcmc2[2]))
print("dfrac2: {0:1.5f} (+{1:1.5f}, -{2:1.5f}) MEarth".format(dfrac_mcmc2[0],dfrac_mcmc2[1], dfrac_mcmc2[2]))
print("q2: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) [-]".format(q_mcmc2[0],q_mcmc2[1], q_mcmc2[2]))
print("rmean2: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) au".format(rm_mcmc2[0],rm_mcmc2[1], rm_mcmc2[2]))
print("rwidth2: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) au".format(rw_mcmc2[0],rw_mcmc2[1], rw_mcmc2[2]))



DModel_emcee_values = 'DModel_emcee_values_'+composition+'_'+str(nwalkers)+'_'+str(niter)+'_2temp.txt'
f = open(DModel_emcee_values,"w+")
f.write("s_min: {0:1.3f} (+{1:1.2f}, -{2:1.3f}) mu \n".format(sm_mcmc[0],sm_mcmc[1], sm_mcmc[2]))
f.write("dfrac: {0:1.5f} (+{1:1.5f}, -{2:1.5f}) MEarth \n".format(dfrac_mcmc[0],dfrac_mcmc[1], dfrac_mcmc[2]))
f.write("q: {0:1.3f} (+{1:1.2f}, -{2:1.3f}) [-] \n".format(q_mcmc[0],q_mcmc[1], q_mcmc[2]))
f.write("rmean: {0:1.3f} (+{1:1.2f}, -{2:1.3f}) au \n".format(rm_mcmc[0],rm_mcmc[1], rm_mcmc[2]))
f.write("rwidth: {0:1.3f} (+{1:1.2f}, -{2:1.3f}) au \n".format(rw_mcmc[0],rw_mcmc[1], rw_mcmc[2]))
print("s_min2: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) mu".format(sm_mcmc2[0],sm_mcmc2[1], sm_mcmc2[2]))
print("dfrac2: {0:1.5f} (+{1:1.5f}, -{2:1.5f}) MEarth".format(dfrac_mcmc2[0],dfrac_mcmc2[1], dfrac_mcmc2[2]))
print("q2: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) [-]".format(q_mcmc2[0],q_mcmc2[1], q_mcmc2[2]))
print("rmean2: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) au".format(rm_mcmc2[0],rm_mcmc2[1], rm_mcmc2[2]))
print("rwidth2: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) au".format(rw_mcmc2[0],rw_mcmc2[1], rw_mcmc2[2]))
f.close()
   
   
   


t1 = time.time()
t = round((t1 - t0)/60,4)
print(f'Total Time: {t} minutes')



