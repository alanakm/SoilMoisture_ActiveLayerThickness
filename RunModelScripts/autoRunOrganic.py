import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from pathlib import Path

from soilice.src_soil import MakeDictFloat, MakeDictArray
# from soilice.src_constitutiveFunctions import GCEFun, thetaFun, thermalKfun
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
    

    #Organic Layer 1 (Layer 1)
    pars1=sim.pars.copy()
    pars1['thetaR']=0.04
    pars1['thetaS']=0.93
    pars1['alpha']=8
    pars1['n']= 1.9
    pars1['m']=1-1/pars1['n']
    pars1['theta_org']=1.0*(1-pars1['thetaS'])
    pars1['theta_mineral'] = (1-pars1['thetaS'])-pars1['theta_org']
    pars1['Ks']= 9.4e-19

    # Organic Layer 2 (Layer 2)
    pars2=sim.pars.copy()
    pars2['thetaR']=0.15
    pars2['thetaS']=0.83
    pars2['alpha']=2
    pars2['n']= 1.7
    pars2['m']=1-1/pars2['n']
    pars2['theta_org']=1.0*(1-pars2['thetaS'])
    pars2['theta_mineral'] = (1-pars2['thetaS'])-pars2['theta_org']
    pars2['Ks']= 9.4e-19

    return pars0,pars1,pars2

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

def psicalc(theta1,theta2,theta3,pars0,pars1,pars2):
    org1 = psiFun(theta1,pars1)
    org2 = psiFun(theta2,pars2)
    minl = psiFun(theta3,pars0)
    return org1,org2,minl

def auto(theta1,theta2,theta3,nYears):
    sim=model()

    opts={}
    opts['infiltration']=0.   
    opts['gravity']=1.       # 0. for horizontal; 1. for vertical
    opts['cryoK']=0.      # 0. flow based on psie; 1. flow based on psif 
    opts['cryoGradient']=0.      # 0. flow based on psie; 1. flow based on psif 
    opts['withadv']=0.       # 0. turn off advection; 1. turn on advection
    opts['conductionTop']=1. # 0. no conduction on upper BC; 1. conduction based on TTop
    opts['conductionBot']=0. # 0. no conduction on lower BC; 1. conduction based on TBot
    opts['simulateFlow'] = False
    opts['simulateTransport'] = True
    opts['freeDrainage']=0.          # 0. no flow lower BC, 1.0 free draining lowerBC
    sim.opts=opts

    nz=1250
    dz=np.zeros(nz)+0.01
    dz[:500] = 0.01 # 0 -5
    dz[500:] = 0.02 # 5m - 20m
    bz=np.hstack([0, np.cumsum(dz)])
    z=(bz[:-1]+bz[1:])/2
    zMax=bz[-1]
    sim.zGrid(bz)
    
    layers=np.zeros(nz)
    layers[0:30]=2. # first 30 cm
    layers[0:15]=1. # first 15 cm

    # ## Time grid
    y=4
    t=np.arange(0,y*365,1) # days
    # t=t*86400. # Convert to seconds
    nt=len(t)
    dt=t[1]-t[0]
    sim.tGrid(0,y*365,dt)


    # ## Soil Parameters

    # Mineral Layer (Silty Clay Loam  (Layer 0)
    sim.readPars()

    pars0,pars1,pars2=getPars(sim)
    # Distribute parameters by layer:

    # This way we have one unique parameter for each depth, which is assigned in this loop.
    # Here I am just making them all the same, so this part of the code would need to be modified:
    parsD={}
    #parsD=MakeDictArray()
    for key in sim.pars:
        parsD[key]=np.zeros(nz)+sim.pars[key]
        #assign parameters based on layer
        parsD[key][layers == 0] = pars0[key]
        parsD[key][layers == 1] = pars1[key]
        parsD[key][layers == 2] = pars2[key]

    sim.pars=parsD
    
    # Set water content for all layers
    org1, org2, minl = psicalc(theta1,theta2,theta3,pars0,pars1,pars2)

    # Initial condition
    # Temperature
    T0=np.zeros(nz)

    # Soil Moisture
    psi0 = np.ones(nz)*minl
    psi0[layers == 2]= org2
    psi0[layers == 1]= org1 

    sim.setICs(T0=T0,psi0=psi0)

    # Winter modified sine wave
    TTop = -np.sin(2* np.pi * (t) / 365) * 13.5 - 1.5
    sin2= -np.sin(2* np.pi * (t) / 365) * 7- 1.5
    TTop[TTop<-1]=sin2[TTop<-1]

    sim.setBCs(TTop=TTop, jBotBC = -4579.2)
 
    # --- Run model ---
    years=int(nYears/4)
    for year in range(years):
        out = sim.run() #dt, t, dz, nz,T0, psi0,qI, TTop, TBot, TInf, jTopBC,parsD, const, opts,rtol=1e-7)
        sim.setICs(T0=out.T[-1,:],psi0=out.psie[-1,:])

    # --- Save to pickle ---
    outdir = Path.home() / "Desktop" / "SOILICE_P2" / "TotalOrganic_Output" / "Output"/ "FixedK_Geothermal_20m"
    outdir.mkdir(parents=True, exist_ok=True)

    fname = (
        f"VWC_th1_{theta1:.2f}_"
        f"th2_{theta2:.2f}_"
        f"th3_{theta3:.2f}.dill"
    )

    save(outdir / fname, out)

    return
