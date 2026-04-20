import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from pathlib import Path

from soilice.src_soil import MakeDictFloat, MakeDictArray
from soilice.src_constitutiveFunctions import GCEFun, thetaFun, thermalKfun
from soilice import model
from soilice import save, load

from soilice import writeDefaultPars
writeDefaultPars()

def getPars(sim):
    pars0=sim.pars.copy()
    pars0['alpha']=0.839
    pars0['n']= 1.5205
    pars0['m']=1-1/pars0['n']
    pars0['thetaS']= 0.482
    pars0['thetaR'] = 0.09
    pars0['Ks']=9.4e-19 
    pars0['cp_org'] = 1920.
    pars0['theta_mineral'] = 1-pars0['thetaS']
    pars0['theta_org'] = 0.

    return pars0

def thetaFun(psi,pars):
    Se=(1+(psi*-pars['alpha'])**pars['n'])**(-pars['m'])
    #Se[psi>0.]=1.0
    return pars['thetaR']+(pars['thetaS']-pars['thetaR'])*Se

def GCEfun(T,pars,const):
    psi=T*const['lambda_f']/(const['g']*const['T0'])
    psi[psi>0]=0.
    return psi

def psiFun(theta, pars):
    m = pars['m']
    alpha = pars['alpha']
    n = pars['n']
    thetaR = pars['thetaR']
    thetaS = pars['thetaS']
    Se = (theta - thetaR) / (thetaS - thetaR)
    psi = - (1 / alpha) * ((Se ** (-1/m) - 1) ** (1/n))
    return psi

def psicalc(theta1,pars0):
    minl = psiFun(theta1,pars0)
    return minl

def auto(theta1,nYears):
    sim=model()

    opts={}
    opts['infiltration']=0.   
    opts['gravity']=0.       # 0. for horizontal; 1. for vertical
    opts['cryoK']=0.      # 0. flow based on psie; 1. flow based on psif 
    opts['cryoGradient']=0.      # 0. flow based on psie; 1. flow based on psif 
    opts['withadv']=0.       # 0. turn off advection; 1. turn on advection
    opts['conductionTop']=1. # 0. no conduction on upper BC; 1. conduction based on TTop
    opts['conductionBot']=0. # 0. no conduction on lower BC; 1. conduction based on TBot
    opts['simulateFlow'] = False
    opts['simulateTransport'] = False
    opts['freeDrainage']=0.          # 0. no flow lower BC, 1.0 free draining lowerBC
    sim.opts=opts


    nz=1500
    dz=np.zeros(nz)+0.01
    dz[0:1000] = 0.01 # 0 -5
    dz[1000:1500] = 0.02 # 5m - 10m
    bz=np.hstack([0, np.cumsum(dz)])
    z=(bz[:-1]+bz[1:])/2
    zMax=bz[-1]
    sim.zGrid(bz)

    layers=np.zeros(nz)

    # ## Time grid

    y=4
    t=np.arange(0,y*365,1) # days
    nt=len(t)
    dt=t[1]-t[0]
    sim.tGrid(0,y*365,dt)


    # ## Soil Parameters

    # Mineral Layer (Silty Clay Loam  (Layer 0)
    sim.readPars()

    pars0=getPars(sim)
  to be modified:
    parsD={}

    for key in sim.pars:
        parsD[key]=np.zeros(nz)+sim.pars[key]
        

    sim.pars=parsD
    
    # Set water content for all layers
    minl = psicalc(theta1,pars0)

    # Initial condition
    # Temperature
    T0=np.zeros(nz)

    # Soil Moisture
    psi0 = np.ones(nz)*minl

    sim.setICs(T0=T0,psi0=psi0)

    # Winter modified sine wave
    TTop = -np.sin(2* np.pi * (t) / 365) * 13.5 - 1.5
    dummy= -np.sin(2* np.pi * (t) / 365) * 7- 1.5
    TTop[TTop<-1]=dummy[TTop<-1]

    sim.setBCs(TTop=TTop)

    # --- Run model ---
    years=int(nYears/4)
    for year in range(years):
        out = sim.run() #dt, t, dz, nz,T0, psi0,qI, TTop, TBot, TInf, jTopBC,parsD, const, opts,rtol=1e-7)
        sim.setICs(T0=out.T[-1,:],psi0=out.psie[-1,:])

    # --- Save to pickle ---
    outdir = Path.home() / "Desktop" / "SOILICE" / "TotalMineral_Output" / "Outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    fname = (
        f"VWC_th1_{theta1:.2f}.dill"
    )

    save(outdir / fname, out)

    return
