# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:09:23 2020

@author: HEKIMOGL
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:06:54 2020

@author: HEKIMOGL
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 00:46:19 2020

@author: Administrator
"""
from scipy.stats import ncx2, ks_2samp, multivariate_normal,kstest
import numpy as np
import scipy.stats as scs
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.special import kv
import scipy as sc
import statsmodels.api as sm
import seaborn as sns
from scipy.optimize import fmin_bfgs,fmin_powell,fmin_slsqp,minimize,newton
from scipy.integrate import quad,trapz,quadrature,fixed_quad,simpson
from yahoo_finance import Share
import yfinance as yf
from pandas_datareader import data as pdr
from joblib import Parallel, delayed
from multiprocessing import Pool,cpu_count
import pandas_datareader as pdrd
import quandl as qdl
import datetime as dt
import numexpr as ne
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
import math
from functions import functions as ff
from functools import partial
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
path0="C://Users//Burak//OneDrive - BAKIRÇAY ÜNİVERSİTESİ//Research//Product Normal//codes//python//Results//"

# path0="D://OneDrive - BAKIRÇAY ÜNİVERSİTESİ//Research//Product Normal//codes//python//Results//"
worker_size=30# How many workers to use.
simsize=1000

onesample=True

for s in range(1,2):
    if s==1:
        ms,ss=[0],[1]
    if s==2:
        ms,ss=np.arange(0.5,3.5,0.5),np.arange(0.5,2.5,0.5)
    if s==3:
        ms,ss=np.arange(0,3.5,0.5),np.arange(0.5,2.5,0.5)
    rf_all=pd.DataFrame(None,columns=['T','s','v','m','r','ss','Pert','mm1','mm2','ss1','ss2','rho','rf-nChi','mks-nChi','rf-VG','mks-VG'
                                                      'rf-NIG','mks-NIG'])
    # rf_all=pd.DataFrame(None,columns=['T','s','v','m','r','ss','Pert','mm1','mm2','ss1','ss2','rho','rf-nChigk','mks-nChigk','rf-nChi','mks-nChi'])
    for r in np.arange(-0.9,0.92,0.2):
        for m in ms:
            for sss in ss:
                if s==1:
                    mm1,mm2,ss1,ss2,rho=m,m,sss,sss,r
                    optpar=(mm1,mm2,ss1,ss2,ff.rho_to_rr(rho))
                if s==2:
                    mm1,mm2,ss1,ss2,rho=-m,m,sss,sss,r
                    optpar=(mm1,mm2,ss1,ss2,ff.rho_to_rr(rho))
                if s==3:
                    mm1,mm2,ss1,ss2,rho=m,m,sss,2*sss,r
                    optpar=(mm1,mm2,ss1,ss2,ff.rho_to_rr(rho))
                
                for ssize in [100,500,1000]:
                    for v in range(1,5):
                        if v==1:
                            perts=np.arange(0,2.2,0.2)
                            rf=np.zeros((perts.shape[0],18))      
                            # rf=np.zeros((perts.shape[0],16)) 
                        else:
                            perts=np.arange(0.2,2.2,0.2)
                            rf=np.zeros((perts.shape[0],18))      
                            # rf=np.zeros((perts.shape[0],16))        
                        pp=0
                        for pert in perts:
                            pert1=0
                            pert2=0
                            pert3=0
                            pert4=0
                            pert5=0
                            globals()['pert'+str(v)] = pert   
                            rf[pp,:-6]=[ssize,s,v,m,r,sss,pert,mm1+pert1,mm2+pert2,ss1+pert3,ss2+pert4,rho]
                            # rf[pp,:-4]=[ssize,s,v,m,r,sss,pert,mm1+pert1,mm2+pert2,ss1+pert3,ss2+pert4,rho]
                            optpar=(mm1+pert1,mm2+pert2,ss1+pert3,ss2+pert4,ff.rho_to_rr(rho))
                            if __name__=='__main__':
                                with Pool(processes=worker_size) as pool:
                                    N = pool.map(partial(ff.NP_sim_gk,optpar=optpar,ssize=ssize,
                                                        mm1=mm1,mm2=mm2,ss1=ss1,ss2=ss2,rho=rho,
                                                        onesample=onesample),range(simsize))
                                pool.close()
                                pool.join()
                            ks_pval0=np.array([N[x][1] for x in range(simsize)]).reshape((-1,1))
                            ks_stat0=np.array([N[x][0] for x in range(simsize)]).reshape((-1,1))
                            ks_pval1=np.array([N[x][3] for x in range(simsize)]).reshape((-1,1))
                            ks_stat1=np.array([N[x][2] for x in range(simsize)]).reshape((-1,1))
                            ks_pval2=np.array([N[x][5] for x in range(simsize)]).reshape((-1,1))
                            ks_stat2=np.array([N[x][4] for x in range(simsize)]).reshape((-1,1)) 
                            rf[pp,-2]=np.mean(ks_pval2<=0.05)
                            rf[pp,-1]=np.mean(ks_stat2)
                            rf[pp,-4]=np.mean(ks_pval1<=0.05)
                            rf[pp,-3]=np.mean(ks_stat1)
                            rf[pp,-6]=np.mean(ks_pval0<=0.05)
                            rf[pp,-5]=np.mean(ks_stat0)
                            # rf[pp,-2]=np.mean(ks_pval1<=0.05)
                            # rf[pp,-1]=np.mean(ks_stat1)
                            # rf[pp,-4]=np.mean(ks_pval0<=0.05)
                            # rf[pp,-3]=np.mean(ks_stat0)
                            pp=pp+1
                        mat0=pd.DataFrame(rf,columns=['T','s','v','m','r','ss','Pert','mm1','mm2','ss1','ss2','rho','rf-nChi','mks-nChi','rf-VG','mks-VG',
                                                      'rf-NIG','mks-NIG'])
                        # mat0=pd.DataFrame(rf,columns=['T','s','v','m','r','ss','Pert','mm1','mm2','ss1','ss2','rho','rf-nChigk','mks-nChigk','rf-nChi','mks-nChi'])
                        rf_all=pd.concat((rf_all,mat0),axis=0) 
                        with pd.ExcelWriter(path0+'Results for Scenario '+str(s)+' new comparison.xlsx',engine='openpyxl') as writer:
                            rf_all.to_excel(writer,sheet_name='Main',index=False,
                                            header=True,engine='openpyxl')
