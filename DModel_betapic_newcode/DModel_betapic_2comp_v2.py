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
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import shift
from scipy.integrate import simps, quad, trapz
#from photutils.centroids import fit_2dgaussian
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
    #Luminosity as function of wavelength [SI Units]
    
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

#Main Dusty Model code
#Using dataframes: easier to sort out grain regimes
#def BetaGrains(smin,smax,dfrac,q,rm,rw,rin,rout,rho,no_s,hd_bs,TGrid,Qpr,QscaG,QabsG,wr):
#def BetaGrains(s_gs,r_belt,f_belt,rbins_bm,rbins_bins,q,df_Qpr,df_Qabs,wr):
#def BetaGrains(smin,smax,dfrac,q,rmean,rw,rin,rout,rho,no_s,hd_bs,TGrid,Qpr,QscaG,QabsG,wr): #minimum grain size, max grain size, disc frac [Me], q, rmean, rsig, rin, rou, rho,no_s,bin size,  [au], wavelength range
def BetaGrains(smin,smax,dfrac,q,rm,rw,rin,rout,no_s,composition):

    if composition == 'diice_hypv2':
        Tg_sr = Tg_sr_di
        Qpr_s = Qpr_s_di
        QabsG = Qabs_sw_di
        QscaG = Qsca_sw_di
        rho = 2.0
    if composition == 'silicate_d03':
        Tg_sr = Tg_sr_as
        Qpr_s = Qpr_s_as
        QabsG = Qabs_sw_as
        QscaG = Qsca_sw_as
        rho = 3.3

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
#        plt.savefig('Old Code Test.pdf')
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
    
    SED_total_nb = np.add(df_fluxsm['Sum'].to_numpy(),flux_nu)
    SED_total = np.add(df_fluxsmb['Sum'].to_numpy(),flux_nu)
#    SED_total_nb = np.add(df_fluxsm['Sum'].to_numpy(),flux_s)
#    SED_total = np.add(df_fluxsmb['Sum'].to_numpy(),flux_s)


    return [df_rbins_p, df_rbins_pb, df_rbins_dN, df_rbins_dNb, df_rbins_dM,df_rbins_dMb, df_fluxsm, df_fluxsmb, y_values_nb, y_values_b, grainsizes,gr_blowout,MPerb,SED_total,SED_total_nb,df_fluxsmb['Sum'].to_numpy(),df_fluxsm['Sum'].to_numpy()]

#-----------------------------------------------------------------#
#Dust Migration model
#Revamping the original code to remove the reliance on dataframes
#Simply return a SED flux
def DustyMM(smin,smax,dfrac,q,rm,rw,rin,rout,composition,f_sd):

    if composition == 'diice_hypv2':
        Tg_sr = Tg_sr_di
        Qpr_s = Qpr_s_di
        Qabs_sw = Qabs_sw_di
        Qsca_sw = Qsca_sw_di
        rho = 2.0
    if composition == 'silicate_d03':
        Tg_sr = Tg_sr_as
        Qpr_s = Qpr_s_as
        Qabs_sw = Qabs_sw_as
        Qsca_sw = Qsca_sw_as
        rho = 3.3

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
        
 #Scattered Emission
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
    
    SED_total = np.add(flux_b,flux_nu)
    
    MPerb = round((dM_radial_b/DiscMass)*100,3)

    return SED_total
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
    sm,dfrac,qv, sm2,dfrac2,qv2 = theta

    #Outer Belt
    composition = comp_outer
    f_sd = f_sd1

    func_flux1 = DustyMM(sm,smax_r,dfrac,qv,rm1,rw1,rin,rout,composition,f_sd)
    #Inner Belt

    f_sd = f_sd2
    composition = comp_inner
    func_flux2 = DustyMM(sm2,smax_r,dfrac2,qv2,rm2,rw2,rin2,rout2, composition,f_sd)
    
    #Add belt fluxes + stellar
    f_belts = np.add(func_flux1,func_flux2) #add flux contributions from each belt
    func_flux = np.add(flux_nu,f_belts) #add belt contributions to stellar flux
    func_wf = interpolate.interp1d(wr,func_flux)
    flam = func_wf(phot_wav)
#    #print(flam)
    c2 = 0
    for i in range(len(phot_wav)):
        c2 = c2 + (((phot_flux[i]-flam[i]))/phot_unc[i])**2
#    lp = -0.5*np.sum( (phot_flux - flam) /phot_unc **2)
    chi2 = -0.5*c2
    print(chi2)
#    chi2chk = Chi2(phot_wav, phot_flux, phot_unc,wr,func_flux)
#    print(chi2chk)

    return chi2

def lnprior(theta):
#    sm,dfrac,q,rm,rw = theta
#    sm,dfrac,qv,rm,rw, sm2,dfrac2,qv2,rm2,rw2 = theta
    sm,dfrac,qv,sm2,dfrac2,qv2 = theta
#    if 1.83 < sm < 20 and 0.001 < dfrac < 2.0 and 3.0 < q < 4.0 and 15 < rm < 150 and 10 < rw < 200:
#    if 1.9 < sm < 20 and 0.001 < dfrac < 2.0 and 3.0 < qv < 4.0 and 60 < rm < 130 and 2 < rw < 110 and 1.9 < sm2 < 20 and 0.001 < dfrac2 < 2.0 and 3.0 < qv2 < 4.0 and 1 < rm2 < 15 and 0.1 < rw2 < 10:
    if 0.01 < sm < 20 and 0.001 < dfrac < 2.0 and 3.0 < qv < 4.0 and 0.01 < sm2 < 20 and 0.001 < dfrac2 < 2.0 and 3.0 < qv2 < 4.0:
        return 0.0
    else:
        return -np.inf


def lnprob(theta,lam,flx,unc):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,lam,flx,unc)

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
[Tg_sr_di,Qpr_s_di,Qabs_sw_di,Qsca_sw_di,sblowA_di,beta_di] = radpressure(Ls,Ms,Rs,Ts,2.0,'diice_hypv2')

##Grain properties##
rho = 3.3 #grain density
#rho = 2.0
###Grain Optical Constants
composition = 'silicate_d03'
#composition = 'diice_hypv2'
#comp = DI
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
[Tg_sr_as,Qpr_s_as,Qabs_sw_as,Qsca_sw_as,sblowA_as,beta_as] = radpressure(Ls,Ms,Rs,Ts,3.3,'silicate_d03')
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
##Scenarios
Scen = 1#0: inner: astro-silicate outer: dirty ice / 1: inner/outer: astro-silicate 2: inner/outer: dirty ice
if Scen == 0:
    comp_inner = 'silicate_d03'
    comp_outer = 'diice_hypv2'
    ci = 'AS'
    co = 'DI'
    sm_outer = [5.279,2.15,-1.477]
    dfrac_outer = [0.46989,0.05444, -0.05537]
    q_outer = [3.684,0.06, -0.049]
    sm_inner = [2.78,1.25, -0.99]
    dfrac_inner = [0.09389,0.01964, -0.00642]
    q_inner = [3.87,0.10, -0.16]
if Scen == 1:
    comp_inner = 'silicate_d03'
    comp_outer = comp_inner
    ci = 'AS'
    co = ci
    sm_outer = [2.907,1.06, -0.611]
    dfrac_outer = [0.97381,0.20385, -0.19533]
    q_outer = [3.874,0.05, -0.051]
    sm_inner = [3.70,1.90, -1.39]
    dfrac_inner = [0.08830,0.00762, -0.00569]
    q_inner = [3.82,0.12, -0.21]

if Scen == 2:
    comp_inner = 'diice_hypv2'
    comp_outer = comp_inner
    ci = 'DI'
    co = ci
    sm_outer = [5.004,2.28, -1.903]
    dfrac_outer = [0.43768,0.09714, -0.06295]
    q_outer = [3.660,0.06, -0.052]
    sm_inner = [17.33,1.92, -4.18]
    dfrac_inner = [0.05216,0.00383, -0.00276]
    q_inner = [3.62,0.26, -0.33]

print(f'Composition of Inner and Outer Belts are {comp_inner} and {comp_outer} respectivetly.')


#-----------------------------------------------------------------#

#Constant Input parameters not included in emcee analysis
smax_r = 3000
rin = 5  ##inner edge of outer cool belt
rout = 200 ##outer edge of outer cool belt
rm1 = 106.34  #mean value from Matra et al. ALMA 1.3mm
rw1 = 35.98 #

#test = DustyMM(2,smax_r,0.5,3.5,rm1,rw1,rin,rout,'silicate_d03')

#Inner Belt
rin2 = 1  ##inner edge of inner warm belt
rout2 = 20 ##outer edge of inner warm belt
#rm2 = 5.6  #2012 Li & Telesco
#rw2 = 3.0
rm2 = 6.4 #Okamoto et al. 2004
rw2 = 1.5 #Estimate where the planetesimal are???

Tpr_range = [23e6]
Tpr = 23e6
tau = 24.3e-4

#set grain size regime
n_gs = 100
s_gs = np.geomspace(0.01,3000,n_gs) #Set sizes
#print(s_gs)
#print(np.geomspace(0.01,3000,n_gs))
#print(zed)

#Dataframe for probability of finding grain and distance
df_rbins_s = pd.DataFrame({'Distance (au)': rbins_bm})

#dNc
#q = 0   #default but this would need to implemented inside function

#dMs = (s_gs*1e-4)**-q*rho*(4/3)*pi*(s_gs*1e-4)**3
#Msum = np.sum(dMs)      #unscaled mass component
#dNc = DiscMass/Msum    #finding constant

r_steps = np.int64((rout-rin)/hd_bs + 1) #steps will be same size as rbins_bm values
r_belt = np.linspace(rin, rout, r_steps) #radial parametric space for belt

##Planetesimal Belt Model##
#print(f'Mean distance of grain release: {rm} au')

#outer belt
ddgauss1 = []
for r in rbins_bm:
    if r >= rin and r <= rout: #inside belt create belt value at rbin_bm location
        ddgauss1.append(np.exp(-0.5*((r - rm1)/rw1)**2))
    else:
        ddgauss1.append(0)
#Normalise models to give a total sum of radial distribution to be unity
ddgauss1 = ddgauss1/(np.sum(ddgauss1))

#Final probability distribution value of planetesimals
f_gauss1 = interpolate.interp1d(rbins_bm,ddgauss1)

#inner belt
ddgauss2 = []
for r in rbins_bm:
    if r >= rin2 and r <= rout2: #inside belt create belt value at rbin_bm location
        ddgauss2.append(np.exp(-0.5*((r - rm2)/rw2)**2))
    else:
        ddgauss2.append(0)
#Normalise models to give a total sum of radial distribution to be unity
ddgauss2 = ddgauss2/(np.sum(ddgauss2))

#Final probability distribution value of planetesimals
f_gauss2 = interpolate.interp1d(rbins_bm,ddgauss2)

##sum two belt profiles
#ddgauss = np.add(ddgauss1, ddgauss2)
#print(len(ddgauss1))
#print(len(ddgauss2))
#print(len(ddgauss))
##ddgauss = ddgauss/np.sum(ddgauss)
#
#plt.plot(rbins_bm,ddgauss,'k')
##plt.plot(rbins_bm,ddgauss1, 'r')
##plt.plot(rbins_bm,ddgauss2, 'b')
#plt.xlim([0,220])
#plt.show()
#print(zed)



if Scen == 0:
    rho = 2
    Qpr_s = Qpr_s_di
    for s in s_gs:
        be = 0.574*Ls*Qpr_s(s)/(Ms*rho*s)
        s_col = str(round(s,3))
        den = np.zeros(len(rbins_bm))
        if be < 0.5:
    #        print(f'{be}, {s}')
            for r in r_belt:
                den = den + f_gauss1(r)*prob_grain_rp_p(be,r,Tpr)
    #    df_rbins_s[s_col] = den*(be/0.25)**q
        df_rbins_s[s_col] = den


#    #Interpolate 2D grid space for grain distribution as a function of size and radial distance values
    f_sd1 = interpolate.interp2d(s_gs,rbins_bm,df_rbins_s.iloc[:,1:],kind='linear')  #f_sd (grain size, radial distance)

    rho = 3.3
    Qpr_s = Qpr_s_as
    for s in s_gs:
        be = 0.574*Ls*Qpr_s(s)/(Ms*rho*s)
        s_col = str(round(s,3))
        den = np.zeros(len(rbins_bm))
        if be < 0.5:
    #        print(f'{be}, {s}')
            for r in r_belt:
                den = den + f_gauss2(r)*prob_grain_rp_p(be,r,Tpr)
    #    df_rbins_s[s_col] = den*(be/0.25)**q
        df_rbins_s[s_col] = den


#    #Interpolate 2D grid space for grain distribution as a function of size and radial distance values
    f_sd2 = interpolate.interp2d(s_gs,rbins_bm,df_rbins_s.iloc[:,1:],kind='linear')  #f_sd (grain size, radial distance)
if Scen == 1:
    rho = 3.3
    Qpr_s = Qpr_s_as
    for s in s_gs:
        be = 0.574*Ls*Qpr_s(s)/(Ms*rho*s)
        s_col = str(round(s,3))
        den = np.zeros(len(rbins_bm))
        if be < 0.5:
    #        print(f'{be}, {s}')
            for r in r_belt:
                den = den + f_gauss1(r)*prob_grain_rp_p(be,r,Tpr)
    #    df_rbins_s[s_col] = den*(be/0.25)**q
        df_rbins_s[s_col] = den


#    #Interpolate 2D grid space for grain distribution as a function of size and radial distance values
    f_sd1 = interpolate.interp2d(s_gs,rbins_bm,df_rbins_s.iloc[:,1:],kind='linear')  #f_sd (grain size, radial distance)

    for s in s_gs:
        be = 0.574*Ls*Qpr_s(s)/(Ms*rho*s)
        s_col = str(round(s,3))
        den = np.zeros(len(rbins_bm))
        if be < 0.5:
    #        print(f'{be}, {s}')
            for r in r_belt:
                den = den + f_gauss2(r)*prob_grain_rp_p(be,r,Tpr)
    #    df_rbins_s[s_col] = den*(be/0.25)**q
        df_rbins_s[s_col] = den

#    #Interpolate 2D grid space for grain distribution as a function of size and radial distance values
    f_sd2 = interpolate.interp2d(s_gs,rbins_bm,df_rbins_s.iloc[:,1:],kind='linear')  #f_sd (grain size, radial

if Scen == 2:
    rho = 2
    Qpr_s = Qpr_s_di
    for s in s_gs:
        be = 0.574*Ls*Qpr_s(s)/(Ms*rho*s)
        s_col = str(round(s,3))
        den = np.zeros(len(rbins_bm))
        if be < 0.5:
    #        print(f'{be}, {s}')
            for r in r_belt:
                den = den + f_gauss1(r)*prob_grain_rp_p(be,r,Tpr)
    #    df_rbins_s[s_col] = den*(be/0.25)**q
        df_rbins_s[s_col] = den


#    #Interpolate 2D grid space for grain distribution as a function of size and radial distance values
    f_sd1 = interpolate.interp2d(s_gs,rbins_bm,df_rbins_s.iloc[:,1:],kind='linear')  #f_sd (grain size, radial distance)

    for s in s_gs:
        be = 0.574*Ls*Qpr_s(s)/(Ms*rho*s)
        s_col = str(round(s,3))
        den = np.zeros(len(rbins_bm))
        if be < 0.5:
    #        print(f'{be}, {s}')
            for r in r_belt:
                den = den + f_gauss2(r)*prob_grain_rp_p(be,r,Tpr)
    #    df_rbins_s[s_col] = den*(be/0.25)**q
        df_rbins_s[s_col] = den
        
#    #Interpolate 2D grid space for grain distribution as a function of size and radial distance values
    f_sd2 = interpolate.interp2d(s_gs,rbins_bm,df_rbins_s.iloc[:,1:],kind='linear')  #f_sd (grain size, radial




###Emcee Inputs##
nwalkers = 60
niter = 200
#initial = np.array([5,0.5,3.5,5,0.1,3.5]) #Variables to be tested: sm,dfrac,qv (outer belt) sm2,dfrac2,qv2 (inner belt)
#
#
#ndim = len(initial)
#p0 = [np.array(initial) + np.array([random.uniform(-4.5,10),random.uniform(-0.1,0.1),random.uniform(-0.5,0.5),random.uniform(-4.5,10),random.uniform(-0.05,0.1),random.uniform(-0.5,0.5)]) for i in range(nwalkers)]
#
#print('Walkers for simulation:')
#print(p0)
##print(zed)
#data = lam,flx,unc
#
#
#def main(p0,nwalkers,niter,ndim,lnprob,data):
#    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
#
#    print("Running burn-in...")
#    p0, _, _ = sampler.run_mcmc(p0, 150)
#    burnin = sampler.get_chain()
#    sampler.reset()
#
#    print("Running production...")
#    pos, prob, state = sampler.run_mcmc(p0, niter)
#
#    return sampler, pos, prob, state, burnin
    
#sampler, pos, prob, state, burnin = main(p0,nwalkers,niter,ndim,lnprob,data)
#
#samples = sampler.flatchain
#
##sm_mcmc, dfrac_mcmc, q_mcmc, rm_mcmc, rw_mcmc = np.median(samples, axis=0)
##sm_mcmc, dfrac_mcmc, q_mcmc, rm_mcmc, rw_mcmc, sm_mcmc2, dfrac_mcmc2, q_mcmc2, rm_mcmc2, rw_mcmc2 = np.median(samples, axis=0)
##sm_mcmc, dfrac_mcmc, q_mcmc, sm_mcmc2, dfrac_mcmc2, q_mcmc2 = np.median(samples, axis=0)
#
#sm_outer, dfrac_outer, q_outer, sm_inner, dfrac_inner, q_inner  = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16,50,84],axis=0)))



##Print values
#print(f's_min (outer): {sm_outer}')
#print(f'dfrac (outer): {dfrac_outer}')
#print(f'q (outer): {q_outer}')
#print(f's_min (inner): {sm_inner}')
#print(f'dfrac (inner): {dfrac_inner}')
#print(f'q (inner): {q_inner}')


#New Directory folder
direcsav = '/'+object+'_rm1_'+str(rm1)+'_rm2_'+str(rm2)+'_sm1_'+str(round(sm_outer[0],3))+'_sm2_'+str(round(sm_inner[0],3))+'_inner'+ci+'_outer'+co+'_tpr'+str(Tpr)
model_direc = main_direc + direcsav
subprocess.run(['mkdir',model_direc])



##Plotting##
#Change directory to store subsequent images
os.chdir(model_direc)


#labels = ['s_min-out','dfrac-out','q-out','s_min-in','dfrac-in','q-in']
#fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84])
#cornerfig = object+'_Emcee_corner_nwalkers_'+str(nwalkers)+'_niter'+str(niter)+'_2temp_inner'+ci+'_outer'+co+'.pdf'
#fig.savefig(cornerfig)
#
##Plot chains
#fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
#samplesgc = sampler.get_chain()
#s2 = np.concatenate((burnin, samplesgc), axis=0)
#
#for i in range(ndim):
#    ax = axes[i]
#    ax.plot(s2[:, :, i], "k", alpha=0.3)
#    ax.plot(burnin[:, :, i], "r", alpha=0.3)  #plot burn-in independently
#    ax.set_xlim(0, len(burnin) + len(samplesgc))
#    ax.set_ylabel(labels[i])
#    ax.yaxis.set_label_coords(-0.1, 0.5)
#
#axes[-1].set_xlabel("step number");
#
#chains = object+'_Emcee_chains_nwalkers'+str(nwalkers)+'_niter'+str(niter)+'_2Comps'+ci+co+'.pdf'
#fig.savefig(chains)

#outer belt
sm = sm_outer[0]
dfrac = dfrac_outer[0]
qv = q_outer[0]
rin = rin
rout = rout
smax = smax_r

[df_rbins_p, df_rbins_pb, df_rbins_dN, df_rbins_dNb, df_rbins_dM,df_rbins_dMb, df_fluxsm, df_fluxsmb, y_values_nb, y_values_b, grainsizes,gr_blowout,MPerb,SED_total2,SED_total_nb2,SED_outerbelt,SED_disc_nb2] = BetaGrains(sm,smax,dfrac,qv,rm1,rw1,rin,rout,100,comp_outer)


#inner belt
sm = sm_inner[0]
dfrac = dfrac_inner[0]
qv = q_inner[0]
rin = rin2
rout = rout2

#Slow function
[df_rbins_p2, df_rbins_pb2, df_rbins_dN2, df_rbins_dNb2, df_rbins_dM2,df_rbins_dMb2, df_fluxsm2, df_fluxsmb2, y_values_nb2, y_values_b2, grainsizes2,gr_blowout2,MPerb2,SED_total22,SED_total_nb22,SED_innerbelt,SED_disc_nb22] = BetaGrains(sm,smax_r,dfrac,qv,rm2,rw2,rin,rout,100,comp_inner)


SED_belts = np.add(SED_outerbelt,SED_innerbelt)
SED_starbelts = np.add(flux_nu,SED_belts)

c2 = Chi2(lam,flx,unc,wr ,SED_starbelts)
print(f'Chi2(fast): {c2}')

#DModel_emcee_values = 'DModel_emcee_values_'+str(nwalkers)+'_'+str(niter)+'_2temp_Inner'+ci+'_Outer'+co+'.txt'
#f = open(DModel_emcee_values,"w+")
#f.write("s_min: {0:1.3f} (+{1:1.2f}, -{2:1.3f}) mu \n".format(sm_outer[0],sm_outer[1], sm_outer[2]))
#f.write("dfrac: {0:1.5f} (+{1:1.5f}, -{2:1.5f}) MEarth \n".format(dfrac_outer[0],dfrac_outer[1], dfrac_outer[2]))
#f.write("q: {0:1.3f} (+{1:1.2f}, -{2:1.3f}) [-] \n".format(q_outer[0],q_outer[1], q_outer[2]))
#f.write("s_min2: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) mu \n".format(sm_inner[0],sm_inner[1], sm_inner[2]))
#f.write("dfrac2: {0:1.5f} (+{1:1.5f}, -{2:1.5f}) MEarth \n".format(dfrac_inner[0],dfrac_inner[1], dfrac_inner[2]))
#f.write("q2: {0:1.2f} (+{1:1.2f}, -{2:1.2f}) [-] \n".format(q_inner[0],q_inner[1], q_inner[2]))
##f.write("chi2: {0:1.3f} [-] \n".format(c2))
#f.write("Chi^2 value: %0.3f \n" % (c2[0]))
#f.close()
#
#f = open("DModel_summary.txt","w+")
#f.write("Summary of Fitting SED for %s \n" % (object))
#f.write("Grain properties: \n")
#f.write("Compositions: Inner: %s, Outer: %s\n" % (comp_inner,comp_outer))
##f.write("Blowout size: %0.3f microns \n" % (sblowA[0]))
##f.write("Initial Minimum Grain Size: %0.3f microns, Final minimum grain size: %0.3f microns\n" % (sm,grainsizes[0]))
#f.write("Size distribution exponent: %0.3f \n" % (qv))
#f.write("Inner Belt properties:\n")
#f.write("Mean Belt Stellar distance: %0.3f au with a Gaussian width of %0.3f \n" % (rm2,rw2))
#f.write("Outer Belt properties:\n")
#f.write("Mean Belt Stellar distance: %0.3f au with a Gaussian width of %0.3f \n" % (rm1,rw1))
#f.write("Inner Edge: %0.3f au and Outer Edge %0.3f \n" % (rin2,rout1))
##f.write("Initial Disc Mass: %0.4f M(Earth) Final Disc Mass: %0.4f M(Earth) \n" % (dfrac,dfracf))  #Disc mass as a fraction of Earth mass
#f.write("Chi^2 value: %0.3f \n" % (c2[0]))
#f.write("tpr value: %0.3f \n" % (Tpr))
#f.close()

###Plot beta + gaussian + power: SEDs
plt.clf()
fig, ax = plt.subplots(nrows = 1, ncols = 1)


plt.plot(wr,flux_nu, 'k:', label = 'Star:'+object, zorder = 0)
#plt.plot(lam,flx,'o')
plt.plot(wr,SED_starbelts,'k', label = 'Star+belts', zorder = 3)
plt.plot(wr,SED_innerbelt, 'r:', label = 'Warm Belt'+' '+ci, zorder = 1)
plt.plot(wr,SED_outerbelt, 'c:', label = 'Cold Belt'+' '+co, zorder = 2)


#near infrared
onr_lam = [1.235, 1.662, 2.159]
onr_flx = [54.3, 39.1, 25.9]
onr_unc = [13.2, 8, 5.9]

#mir infrared
mir_lam = [3.4, 4.6, 12.0, 18.3,23.67,  24.6,    25.00, ]
mir_flx = [12.39,9.11, 2.66,4.316, 7.847,  8.807,   10.2]
mir_unc = [5.06,2.35,0.2771, 0.432, 0.392, 0.881, 2.00]

#far infrared
fir_lam = [60, 70, 100.0,  160.0,250.0,  350.0]
fir_flx = [18.5, 16, 9.8,    5.1,1.9,   0.72]
fir_unc = [3.7, 0.8, 1.058, 0.5,0.1,  0.05]

#sub-mm/mm
smm_lam = [500.0,  850.0,1200]
smm_flx = [0.38, 0.058,0.036]
smm_unc = [0.03,  0.006,0.01]

#
plt.errorbar(onr_lam,onr_flx,yerr=onr_unc,fmt='o',mec='green',mfc='green',ecolor='black',capsize=4.,capthick=1, label = 'Near-IR',zorder = 4)
plt.errorbar(mir_lam,mir_flx,yerr=mir_unc,fmt='o',mec='skyblue',mfc='skyblue',ecolor='black',capsize=4.,capthick=1, label = 'Mid-IR', zorder = 5)
plt.errorbar(fir_lam,fir_flx,yerr=fir_unc,fmt='o',mec='red',mfc='red',ecolor='black',capsize=4.,capthick=1, label = 'Far-IR', zorder = 6)
plt.errorbar(smm_lam,smm_flx,yerr=smm_unc,fmt='o',mec='y',mfc='y',ecolor='black',capsize=4.,capthick=1, label = 'Sub-mm/mm', zorder = 7)


plt.xlabel('Wavelength [$\mu$m]', fontsize = 16)
plt.ylabel('Flux Density [Jy]', fontsize = 16)
plt.xlim([0.3, 3000])
plt.ylim([10**-3, 150])
plt.xscale('log')
plt.yscale('log')
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)

ax.legend(loc = 'lower left')
figname = 'SED_'+object+'_disc_para_models_2temp_2comp_combine'+str(niter)+'_'+ci+co+'.pdf'
plt.savefig(figname)
#plt.show()
print(model_direc)
print(zed)
#
#plt.clf()
#fig, ax = plt.subplots(nrows = 1, ncols = 1)
#ax.plot(rbins_bm,df_rbins_dNb['Sum']/np.max(df_rbins_dNb['Sum']),'b-.',label = 'Cool RP+Grav grains')
#ax.plot(rbins_bm,df_rbins_dN['Sum']/np.max(df_rbins_dN['Sum']), 'b', label = 'Cool Grav-only grains')
#ax.plot(rbins_bm,df_rbins_dNb2['Sum']/np.max(df_rbins_dNb2['Sum']),'r-.',label = 'Warm RP+Grav grains')
#ax.plot(rbins_bm,df_rbins_dN2['Sum']/np.max(df_rbins_dNb2['Sum']), 'r', label = 'Warm Grav-only grains')
#ax.set_xlim([0,300])
#ax.set_ylabel('Number Density [N/au]')
#ax.set_xlabel('Stellar Distance [au]')
##ax.set_xscale('log')
##ax.set_yscale('log')
#ax.legend()
#figname = 'SED_'+object+'_Numberdensity_distance_2temp_normalise_2comp_'+ci+co+'.pdf'
#plt.savefig(figname)
#
###Grab grain information
##    sgd = np.geomspace(sm_mcmc[0],smax_r,100)
##    DiscSD = f_sd(sgd,rbins_bm)
#
##Dataframe for probability of finding grain and distance
#df_rbins_smin_o = pd.DataFrame()
#df_rbins_smin_i = pd.DataFrame()
#
#
#if Scen == 0:
#    #outer disc
#    rho = 2
#    for s in s_gs:
#        s_col = str(round(s,3))
#        den = np.zeros(len(rbins_bm)) #resets to zero very loop
#        if s > sm_outer[0]:
#            be = 0.574*Ls*Qpr_s_di(s)/(Ms*rho*s)
#
#            if be < 0.5:
#                for r in r_belt:
#                    pdgrain = prob_grain_rp_p(be,r,Tpr)
#                    den = den + f_gauss1(r)*pdgrain/np.sum(pdgrain)*(be/0.25)**q_outer[0]
#        #    df_rbins_s[s_col] = den*(be/0.25)**q
#        df_rbins_smin_o[s_col] = den
#
#
#    #inner disc
#    rho = 3.3
#    for s in s_gs:
#        s_col = str(round(s,3))
#        den = np.zeros(len(rbins_bm)) #resets to zero very loop
#        if s > sm_inner[0]:
#            be = 0.574*Ls*Qpr_s_di(s)/(Ms*rho*s)
#
#            if be < 0.5:
#        #        print(f'{be}, {s}')
#                for r in r_belt:
#                    pdgrain = prob_grain_rp_p(be,r,Tpr)
#                    den = den + f_gauss2(r)*pdgrain/np.sum(pdgrain)*(be/0.25)**q_inner[0]
#        #    df_rbins_s[s_col] = den*(be/0.25)**q
#        df_rbins_smin_i[s_col] = den
#
#
#if Scen == 1:
#    #outer disc
#    rho = 3.3
#    for s in s_gs:
#        s_col = str(round(s,3))
#        den = np.zeros(len(rbins_bm)) #resets to zero very loop
#        if s > sm_outer[0]:
#            be = 0.574*Ls*Qpr_s_as(s)/(Ms*rho*s)
#
#            if be < 0.5:
#                for r in r_belt:
#                    pdgrain = prob_grain_rp_p(be,r,Tpr)
#                    den = den + f_gauss1(r)*pdgrain/np.sum(pdgrain)*(be/0.25)**q_outer[0]
#        #    df_rbins_s[s_col] = den*(be/0.25)**q
#        df_rbins_smin_o[s_col] = den
#
#
#    #inner disc
#    for s in s_gs:
#        s_col = str(round(s,3))
#        den = np.zeros(len(rbins_bm)) #resets to zero very loop
#        if s > sm_inner[0]:
#            be = 0.574*Ls*Qpr_s_di(s)/(Ms*rho*s)
#
#            if be < 0.5:
#        #        print(f'{be}, {s}')
#                for r in r_belt:
#                    pdgrain = prob_grain_rp_p(be,r,Tpr)
#                    den = den + f_gauss2(r)*pdgrain/np.sum(pdgrain)*(be/0.25)**q_inner[0]
#        #    df_rbins_s[s_col] = den*(be/0.25)**q
#        df_rbins_smin_i[s_col] = den
#
#if Scen == 2:
#    #outer disc
#    rho = 2
#    for s in s_gs:
#        s_col = str(round(s,3))
#        den = np.zeros(len(rbins_bm)) #resets to zero very loop
#        if s > sm_outer[0]:
#            be = 0.574*Ls*Qpr_s_di(s)/(Ms*rho*s)
#
#            if be < 0.5:
#                for r in r_belt:
#                    pdgrain = prob_grain_rp_p(be,r,Tpr)
#                    den = den + f_gauss1(r)*pdgrain/np.sum(pdgrain)*(be/0.25)**q_outer[0]
#        #    df_rbins_s[s_col] = den*(be/0.25)**q
#        df_rbins_smin_o[s_col] = den
#
#
#    #inner disc
#    for s in s_gs:
#        s_col = str(round(s,3))
#        den = np.zeros(len(rbins_bm)) #resets to zero very loop
#        if s > sm_inner[0]:
#            be = 0.574*Ls*Qpr_s_di(s)/(Ms*rho*s)
#
#            if be < 0.5:
#        #        print(f'{be}, {s}')
#                for r in r_belt:
#                    pdgrain = prob_grain_rp_p(be,r,Tpr)
#                    den = den + f_gauss2(r)*pdgrain/np.sum(pdgrain)*(be/0.25)**q_inner[0]
#        #    df_rbins_s[s_col] = den*(be/0.25)**q
#        df_rbins_smin_i[s_col] = den
#
#
#df_rbins_smin = np.add(df_rbins_smin_i,df_rbins_smin_o)
#
#
#
#
#df_rbins_smin = df_rbins_smin.transpose()  #transpose the dataframe diagonally
#
#
##remove bad pixels from interpolation (i.e. zero probably will be black in colour)
#my_cmap = copy.copy(matplotlib.cm.get_cmap('gist_heat')) # copy the default cmap
#my_cmap.set_bad((0,0,0))
#
##image show plot with extent, cmap = gist_hist
##plt.imshow(df_rbins_s,extent=(hd_minm,hd_maxm,s_gs[-1],s_low),interpolation='nearest', cmap = my_cmap,norm=colors.LogNorm()) #, vmin = 0, vmax = 0.02)
#
#
#plt.clf()
##plt.pcolor(rbins_bm, sgd, df_rbins_smin, cmap = my_cmap,norm=colors.LogNorm(),linewidth=0,rasterized=True)
#plt.pcolor(rbins_bm, s_gs, df_rbins_smin, cmap = my_cmap,norm=colors.LogNorm(),linewidth=0,rasterized=True) #,
#ax = plt.gca() #you first need to get the axis handle
#ax.set_aspect('auto') #sets the height to width ratio to 1.5.
##    plt.aspect('auto')
#plt.xlim([0,250])
##plt.ylim(3000,1500)
#plt.ylim([3000,0.01])
##plt.ylim([s_low,3000])
#plt.yscale('log')
#plt.colorbar()
##plt.ylim([1,0])
#plt.ylabel(r'Grain Size [$\mu m$]', fontsize = 16)
#plt.xlabel('Stellar distance [au]', fontsize = 16)
#
##ax.tick_params(which = 'major', bottom = True, top = True, left = True, right = True,direction = 'inout' )
#ax.grid(which = 'major', linestyle = ':')##Decomment after this once I have fixed i
##ax.yaxis.set_major_formatter(adj_log)
#ax.yaxis.set_major_formatter(formatter)
#
#
#title = 'Size_spatdist_timestep_'+str(Tpr)+'_q_'+str(round(q_outer[0],3))+'.pdf'
#plt.savefig(title)
#
#
#
#
#
#
#
#t1 = time.time()
#t = round((t1 - t0)/60,4)
#print(f'Total Time: {t} minutes')



