# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 16:39:14 2025

@author: aheki
"""

import os
os.chdir('C:/Users/aheki/Downloads/pairs/pairs')
from scipy.stats import ncx2
import numpy as np
import scipy.stats as scs
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import multivariate_normal as mvn
import scipy as sc
import statsmodels.api as sm
import seaborn as sns
from scipy.optimize import fmin_bfgs,fmin_powell,fmin_slsqp,minimize
from scipy.integrate import quad,trapz,quadrature,fixed_quad,simpson
import yfinance as yf
from pandas_datareader import data as pdr
from joblib import Parallel, delayed
from multiprocessing import Pool
#import pandas_datareader as pdrd
import quandl as qdl
import datetime as dt
import numexpr as ne
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
import math
from scipy.stats import kruskal,anderson,anderson_ksamp,wilcoxon,multivariate_normal,ks_2samp,ks_1samp
from GKintegrate import *
from numpy import arccos as acos
#stocklist=['DB','ENI.MI','^XU100']
#os.chdir('C:\\Users\\Administrator\\Desktop\\abdl\\calibre_HCJ')
def datagetyahoo(stock,start,end):
    ydr = pdrd.get_data_yahoo(stock,start,end)
    #YahooDailyReader(stock,start,end)
    df= ydr['Adj Close']
    return df

def NIGmom(x):
    K=scs.kurtosis(x)+3.0
    S=scs.skew(x)
    V=np.var(x)
    E=np.mean(x)
    k=K/3.0-1
    A=S/(3.0*K)
    sgm=np.sqrt(V/(1+A**2*k))
    theta=sgm*A
    mu=E-theta
    return  abs(sgm),abs(k),theta,mu

def NIGddnew(x,sgm,k,theta,mu):
    kp=1./k
    A=theta/sgm**2
    B=np.sqrt(theta**2+sgm**2*kp)/sgm**2
    C=1.0/np.pi*np.exp(kp)*np.sqrt(theta**2*kp/sgm**2+kp**2)
    f=C*(np.exp(A*(x-mu))*sc.special.kv(1,B*np.sqrt((x-mu)**2+kp*sgm**2))/np.sqrt((x-mu)**2+kp*sgm**2))
    #(np.sqrt(2)/(np.sqrt(np.pi)*sgm*sc.special.gamma(nu)))*(np.abs(x-mus)/np.sqrt(theta**2+2*sgm**2))**(nu-0.5)*np.exp((x-mus)*theta/sgm**2)*sc.special.kv(nu-0.5,np.abs(x-mus)*np.sqrt(theta**2+2*sgm**2)/sgm**2)
    return f
def brloglikNIG(pars,x):
    sgm=pars[0];k=pars[1];theta=pars[2];mu=pars[3];
    Lv=np.sum(np.log(NIGddnew(x,sgm,k,theta,mu)))
    LL=-Lv
    return LL
def VGmom(x):
    K=scs.kurtosis(x)+3.0
    S=scs.skew(x)
    V=np.var(x)
    E=np.mean(x)
    #theta=(E*S**2)**(1./3.);
    #nu=E/theta;
    #sgm=V/nu;
    nu=3./(K-3.)
    sgm=np.sqrt(V*(K-3.)/3.)
    theta=S*np.sqrt(V)/3.
    mus=E-theta*nu
    return  abs(sgm),abs(nu),theta,mus
def VGmomdt(x,d):
    K=(scs.kurtosis(x)+3.0)*d
    S=scs.skew(x)*d
    V=np.var(x)*d
    E=np.mean(x)*d
    #theta=(E*S**2)**(1./3.);
    #nu=E/theta;
    #sgm=V/nu;
    nu=3./((K-3.))
    sgm=np.sqrt(V*(K-3.)/3.)
    theta=S*np.sqrt(V)/3.
    mus=E-theta*nu
    return  abs(sgm),abs(nu),theta,mus

def VGmomFull(x):
    d=1
    #T/n
    sgm0,nu0,theta0,mus0=VGmom(x)
    eps0=(theta0**2*nu0)/sgm0**2
    
    K=scs.kurtosis(x)+3.0
    S=scs.skew(x)
    V=np.var(x)
    E=np.mean(x)
    feps=lambda eps:eps*(3+2*eps)**2/((1+4*eps+2*eps**2)*(1+eps))-2*S**2/K
    #theta=(E*S**2)**(1./3.);
    #nu=E/theta;
    #sgm=V/nu;
    epstar=newton(feps,eps0)
    sgm=np.sqrt(V/(1+epstar)/d)
    nu=K*d/3*((1+epstar)**2/(1+4*epstar+2*epstar**2))
    theta=S/(sgm**2*nu*d)*(1/(3+2*epstar))
    mus=E-theta*nu
    return abs(sgm),abs(nu),theta,mus



def VGddnew(x,sgm,nu,theta,mus):
    f=(np.sqrt(2)/(np.sqrt(np.pi)*sgm*sc.special.gamma(nu)))*(np.abs(x-mus)/np.sqrt(theta**2+2*sgm**2))**(nu-0.5)*np.exp((x-mus)*theta/sgm**2)*sc.special.kv(nu-0.5,np.abs(x-mus)*np.sqrt(theta**2+2*sgm**2)/sgm**2)
    return f


#@jit(nopython=False,parallel=True)
def loglikVGnum(pars,x):
    sgm=pars[0];nu=pars[1];theta=pars[2];mus=pars[3];
#mus=pars(1);
#mus=0;
    T=np.shape(x)[0];
    nL=-(0.5*T*np.log(2/np.pi)+np.sum((x-mus)*theta/sgm**2)-T*np.sum(np.log(sc.special.gamma(nu)*sgm))+np.sum(np.log(sc.special.kv(nu-0.5,(np.sqrt(2*sgm**2+theta**2)*np.abs(x-mus)/sgm**2))))+np.sum((nu-0.5)*(np.log(np.abs(x-mus)-0.5*np.log(2.*sgm**2+theta**2)))))
    return nL
def brloglikVG(pars,x):
    sgm=pars[0];nu=pars[1];theta=pars[2];mus=pars[3];
    Lv=np.sum(np.log(VGddnew(x,sgm,nu,theta,mus)))
    LL=-Lv
    return LL
@jit(nopython=True,parallel=False)
def appNormCDF(x):
    p=1.0/(np.exp(-358.0*x/23.0+111*np.arctan(37.0*x/294))+1.0)
    return p
@jit(nopython=True,parallel=False,nogil=True,fastmath=True)
def fastNpdf(x):
    C=1.0/np.sqrt(2.0*np.pi)
    return np.exp(-x*x*0.5)*C

def getnchix2rvs(mm1,mm2,ss1,ss2,rr,nsim):
    rr2=np.sqrt(ss1*ss2*(1-rr)/2)
    rr1=np.sqrt(rr*ss1*ss2+rr2**2)
    ll1=((mm1**2*ss2**2+mm2**2*ss1**2-2*rr*mm1*mm2*ss1*ss2)+4*mm1*mm2*rr1**2)/(4*rr1**2+4*rr2**2)
    ll2=ll1-mm1*mm2
    d1=ncx2(1,((ll2/rr2**2)),loc=0.0,scale=rr2**2)
    d2=ncx2(1,((ll1/rr1**2)),loc=0.0,scale=rr1**2)
    diffnch2rv=d2.rvs(size=nsim)-d1.rvs(size=nsim)
    return diffnch2rv

def simNormProd(pars,nsim):
    mm1,mm2,ss1,ss2,rr=pars
    rr=1.0/(1.0+np.exp(-rr))
    bivr=multivariate_normal.rvs([mm1,mm2],cov=[[ss1**2,rr*ss1*ss2],[rr*ss1*ss2,ss2**2.]],size=nsim)
    return bivr[:,0]*bivr[:,1]

def NIGcdf(ub,sgmn,k,thetan,mun):
    if ub<0.0:
        I=gk(lambda x:NIGddnew(x,sgmn,k,thetan,mun),a=-7.0,b=ub)
    else:
        I=gk(lambda x:NIGddnew(x,sgmn,k,thetan,mun),a=-7.0,b=0.0)+gk(lambda x:NIGddnew(x,sgmn,k,thetan,mun),a=0.0,b=ub)
    return I

def VGcdf(ub,sgm,nu,theta,mu):
    if ub<0.0:
        I=gk(lambda x:VGddnew(x,sgm,nu,theta,mu),a=-7.0,b=ub)
    else:
        I=gk(lambda x:VGddnew(x,sgm,nu,theta,mu),a=-7.0,b=0.0)+gk(lambda x:VGddnew(x,sgm,nu,theta,mu),a=0.0,b=ub)
    return I



def Pnormprodpdfbynchi2(z,pars,ispdf,scalar):
    mm1,mm2,ss1,ss2,rr=pars
    ss1=abs(ss1)
    ss2=abs(ss2)
    # rr=np.sign(rr)*1.0/(1.0+np.exp(-rr))
    rr2=np.sqrt(ss1*ss2*(1-rr)/2)
    rr1=np.sqrt(rr*ss1*ss2+rr2**2)
    ll1=abs((mm1**2*ss2**2+mm2**2*ss1**2-2*rr*mm1*mm2*ss1*ss2)+4*mm1*mm2*rr1**2)/(4*rr1**2+4*rr2**2)
    ll2=abs(ll1-mm1*mm2)
    nchix2=lambda x,l:(fastNpdf(np.sqrt(x)+np.sqrt(l))+fastNpdf(np.sqrt(x)-np.sqrt(l)))*0.5/np.sqrt(x)
    cnchix2=lambda x,l:(appNormCDF(np.sqrt(x)+np.sqrt(l))+appNormCDF(np.sqrt(x)-np.sqrt(l)))-1.0
    crmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)
    nrmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*nchix2((x+z)/sgm1,ll1)/(sgm1*sgm2)
    if ispdf:
        if scalar:
            out=quad(nrmprdnchicorr, abs(np.minimum(z,0.0)),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
        else:
            out=Parallel(n_jobs=8)(delayed(lambda zz:quad(nrmprdnchicorr, abs(np.minimum(zz,0.0)),np.inf,epsabs=1e-5,args=(zz,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0])(zz)  for zz in z)
        #quad(nrmprdnchicorr, abs(np.minimum(z,0.0)),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
        
    else:
        crmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)/sgm2
        #np.sqrt(sgm1*sgm2)
        
        if scalar:
            out=quad(crmprdnchicorr, abs(np.minimum(z,0.0)),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
            # out2=quad(lambda x,sgm2:nchix2(x/sgm2,ll2),abs(np.minimum(z,0.0)),z,args=(rr2**2))[0]
            # print("Conv.Singl.",out2)
        else:
            out=[quad(lambda x:crmprdnchicorr(x,zz,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2), abs(np.minimum(zz,0.0)),np.inf,epsabs=1e-5)[0]  for zz in z]
            #Parallel(n_jobs=8)(delayed(lambda zz:quad(crmprdnchicorr, abs(np.minimum(zz,0.0)),np.inf,epsabs=1e-5,args=(zz,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0])(zz)  for zz in z)
    return out

def PnormprodCdfByFast(z, pars, scalar=True):
    mu1, mu2, sigma1, sigma2, rho = pars

    # Compute scaling terms
    s_x = np.sqrt((1 + rho) * sigma1 * sigma2 / 2)
    s_y = np.sqrt((1 - rho) * sigma1 * sigma2 / 2)

    # Compute non-centrality parameters
    lambda_x = abs((mu1**2 * sigma2**2 + mu2**2 * sigma1**2 - 2 * rho * mu1 * mu2 * sigma1 * sigma2
                    + 4 * mu1 * mu2 * s_x**2) / (4 * sigma1 * sigma2))
    lambda_y = abs(lambda_x - mu1 * mu2)

    def compute_cdf_single(zz):
        lower = abs(min(zz, 0))

        def integrand(x):
            sqrtx = np.sqrt(x)
            term1 = fastNpdf(sqrtx / s_y - np.sqrt(lambda_y) / s_y)
            term2 = fastNpdf(sqrtx / s_y + np.sqrt(lambda_y) / s_y)
            density_part = (term1 + term2) / (2 * sqrtx * s_y)

            sqrtxz = np.sqrt(x + zz)
            cdf_part = (
                appNormCDF(sqrtxz / s_x - np.sqrt(lambda_x) / s_x) +
                appNormCDF(sqrtxz / s_x + np.sqrt(lambda_x) / s_x)
            )

            return density_part * cdf_part

        result, _ = quad(integrand, lower, np.inf, epsabs=1e-6)

        sqrtz_term = np.sqrt(max(0, -zz)) / s_y
        sqrt_ly_term = np.sqrt(lambda_y) / s_y
        tail_correction = (
            appNormCDF(-sqrtz_term + sqrt_ly_term) +
            appNormCDF(-sqrtz_term - sqrt_ly_term)
        )

        return result - tail_correction

    if scalar:
        return compute_cdf_single(z)
    else:
        return np.array(Parallel(n_jobs=-1)(delayed(compute_cdf_single)(zz) for zz in z))

def Pnormprodpdfbynchi2BC(z,pars,ispdf,scalar):
    mm1,mm2,ss1,ss2,rr=pars
    ss1=abs(ss1)
    ss2=abs(ss2)
    # rr=np.sign(rr)*1.0/(1.0+np.exp(-rr))
    rr2=np.sqrt(ss1*ss2*(1-rr)/2)
    rr1=np.sqrt(rr*ss1*ss2+rr2**2)
    ll1=abs((mm1**2*ss2**2+mm2**2*ss1**2-2*rr*mm1*mm2*ss1*ss2)+4*mm1*mm2*rr1**2)/(4*rr1**2+4*rr2**2)
    ll2=abs(ll1-mm1*mm2)
    nchix2=lambda x,l:(fastNpdf(np.sqrt(x)+np.sqrt(l))+fastNpdf(np.sqrt(x)-np.sqrt(l)))*0.5/np.sqrt(x)
    cnchix2=lambda x,l:(appNormCDF(np.sqrt(x)+np.sqrt(l))+appNormCDF(np.sqrt(x)-np.sqrt(l)))-1.0
    crmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2((x-z)/sgm2,ll2)*cnchix2((x)/sgm1,ll1)
    nrmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2((x-z)/sgm2,ll2)*nchix2((x)/sgm1,ll1)/(sgm1*sgm2)
    if ispdf:
        if scalar:
            out=quad(nrmprdnchicorr, np.maximum(z,0.0),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
        else:
            out=Parallel(n_jobs=8)(delayed(lambda zz:quad(nrmprdnchicorr, abs(np.minimum(zz,0.0)),np.inf,epsabs=1e-5,args=(zz,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0])(zz)  for zz in z)
        #quad(nrmprdnchicorr, abs(np.minimum(z,0.0)),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
        
    else:
        crmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)/sgm2
        #np.sqrt(sgm1*sgm2)
        
        if scalar:
            out=quad(crmprdnchicorr, np.maximum(z,0.0),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
        else:
            out=[quad(lambda x:crmprdnchicorr(x,zz,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2), np.maximum(z,0.0),np.inf,epsabs=1e-5)[0]  for zz in z]
            #Parallel(n_jobs=8)(delayed(lambda zz:quad(crmprdnchicorr, abs(np.minimum(zz,0.0)),np.inf,epsabs=1e-5,args=(zz,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0])(zz)  for zz in z)
    return out


def NPcorrConst(s1,s2,r):
    c=1/((1.0-r**2)*(s1*s2))
    p=r/((1.0-r**2)*(s1*s2))
    #A=np.exp(-0.5*m1**2/(s1**2*(1-r**2))-0.5*m2**2/(s2**2*(1-r**2))+m1*m2*r/(1-r**2))*1.0/(np.sqrt(1.0-r**2)*(np.pi*s1*s2))
    return (acos(p/c)/np.sqrt(c**2-p**2)+acos(-p/c)/np.sqrt(c**2-p**2))



@jit(nopython=True)
def nchix2(x,l):
    return (fastNpdf(np.sqrt(x)+np.sqrt(l))+fastNpdf(np.sqrt(x)-np.sqrt(l)))*0.5/np.sqrt(x)

@jit(nopython=True)
def nchix2u(u,l):
    return (fastNpdf(u+np.sqrt(l))+fastNpdf(u-np.sqrt(l)))


@jit(nopython=True)
def cnchix2(x,l):
    return (appNormCDF(np.sqrt(x)+np.sqrt(l))+appNormCDF(np.sqrt(x)-np.sqrt(l)))-1.0

@jit(nopython=True)
def cnchix2u(u,l):
    return (appNormCDF(np.sqrt(x)+np.sqrt(l))+appNormCDF(np.sqrt(x)-np.sqrt(l)))-1.0



@jit(nopython=True)
def crmprdnchicorr(x,z,ll1,ll2,sgm1,sgm2):
    return nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)/sgm2

@jit(nopython=True)
def crmprdnchicorru(u,z,ll1,ll2,sgm1,sgm2):
    return nchix2u(u/np.sqrt(sgm2),ll2)*cnchix2u(u/np.sqrt(sgm1)+z/sgm1,ll1)/np.sqrt(sgm2)


@jit(nopython=True)
def nrmprdnchicorr(x,z,ll1,ll2,sgm1,sgm2):
    return nchix2(x/sgm2,ll2)*nchix2((x+z)/sgm1,ll1)/(sgm1*sgm2+1e-12)

@jit(nopython=True)
def nrmprdnchicorru(u,z,ll1,ll2,sgm1,sgm2):
    x=u**2
    return nchix2u(u/np.sqrt(sgm2),ll2)*nchix2u(np.sqrt(x+z)/np.sqrt(sgm1),ll1)/((sgm1*sgm2)+1e-12)



def Pnormprodpdfbynchi2gk(z,pars,ispdf,scalar,bb):
    mm1,mm2,ss1,ss2,rr=pars
    mm1*=np.sign(z)
    # mm2*=np.sign(z)
    z*=np.sign(z)
    # rr*=np.sign(z)
    ss1=abs(ss1)
    ss2=abs(ss2)
    # rr=1.0/(1.0+np.exp(-rr))
    rr2=np.sqrt(ss1*ss2*(1-rr+1e-12)/2)
    rr1=np.sqrt(rr*ss1*ss2+rr2**2)
    ll1=abs((mm1**2*ss2**2+mm2**2*ss1**2-2*rr*mm1*mm2*ss1*ss2)+4*mm1*mm2*rr1**2)/(4*rr1**2+4*rr2**2)
    ll2=abs(ll1-mm1*mm2)
    
    
    # nchix2=lambda x,l:(fastNpdf(np.sqrt(x)+np.sqrt(l))+fastNpdf(np.sqrt(x)-np.sqrt(l)))*0.5/np.sqrt(x)
    # cnchix2=lambda x,l:(appNormCDF(np.sqrt(x)+np.sqrt(l))+appNormCDF(np.sqrt(x)-np.sqrt(l)))-1.0
    # crmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)
    # nrmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*nchix2((x+z)/sgm1,ll1)/(sgm1*sgm2)
    if ispdf:
        if scalar:
            out=adGKIntegrate(f=nrmprdnchicorr, a=0.0,b=bb,tol=1e-9,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0]
            #abs(np.minimum(z,0.0))
        else:
            out=[adGKIntegrate(f=nrmprdnchicorr, a=0.0,b=bb,tol=1e-9,args=(zz+1e-15,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0]  for zz in z]
            #Parallel(n_jobs=8)(delayed(lambda zz:adGKIntegrate(f=nrmprdnchicorr, a=abs(np.minimum(zz,0.0)),b=bb,tol=1e-9,args=(zz,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0])(zz)  for zz in z)
        #quad(nrmprdnchicorr, abs(np.minimum(z,0.0)),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
        
    else:
    
        
        if scalar:
            out=adGKIntegrate(f=crmprdnchicorr, a=0.0,b=bb,tol=1e-9,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0]
        else:
            out=[adGKIntegrate(f=crmprdnchicorr, a=0.0,b=bb,tol=1e-9,args=(zz+1e-15,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0]  for zz in z]
            #Parallel(n_jobs=8)(delayed(lambda zz:adGKIntegrate(f=crmprdnchicorr, a=abs(np.minimum(zz,0.0)),b=bb,tol=1e-9,args=(zz,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0])(zz)  for zz in z)
    return out



def getfnF(z,s1,s2,m1,m2,rr):
    lx=getparsnchi([m1,m2,s1,s2,rr])[0]
    ly=getparsnchi([m1,m2,s1,s2,rr])[1]
    r=getparsnchi([m1,m2,s1,s2,rr])[2]
    c=2.0/((1-r**2)*s1*s2)
    fn=lambda c,z,lx,ly,r:np.exp(z*r/((1-r**2))-c*abs(z)*s1*s2/2.0-c*(lx*(1-r)*s1*s2+ly*(1+r)*s1*s2)/2.0)*quad(lambda t:np.exp(-c*abs(z)*t*s1*s2-c*np.sqrt(abs(z)*t)*(np.sqrt(ly)*(1+r)*s1*s2+np.sqrt(lx)*(1-r)*s1*s2*(np.sqrt(1.0+1.0/t))))*(t*(1+t))**-0.5,a=0.0,b=np.inf)[0]/(np.sqrt(1.0-r**2)*np.pi*s1*s2)
    fn1=lambda c,z,lx,ly,r:np.exp(z*r/((1-r**2))-c*abs(z)*s1*s2/(2.0)-c*(lx*(1-r)*s1*s2+ly*(1+r)*s1*s2)/2.0)*quad(lambda t:np.exp(-c*abs(z)*t*s1*s2-c*np.sqrt(abs(z)*t)*(np.sqrt(ly)*(1+r)*s1*s2-np.sqrt(lx)*(1-r)*s1*s2*(np.sqrt(1.0+1.0/t))))*(t*(1+t))**-0.5,a=0.0,b=np.inf)[0]/(np.sqrt(1.0-r**2)*np.pi*s1*s2)
    fn2=lambda c,z,lx,ly,r:np.exp(z*r/((1-r**2))-c*abs(z)*s1*s2/(2.0)-c*(lx*(1-r)*s1*s2+ly*(1+r)*s1*s2)/2.0)*quad(lambda t:np.exp(-c*abs(z)*t*s1*s2-c*np.sqrt(abs(z)*t)*(-np.sqrt(ly)*(1+r)*s1*s2+np.sqrt(lx)*(1-r)*s1*s2*(np.sqrt(1.0+1.0/t))))*(t*(1+t))**-0.5,a=0.0,b=np.inf)[0]/(np.sqrt(1.0-r**2)*np.pi*s1*s2)
    fn3=lambda c,z,lx,ly,r:np.exp(z*r/((1-r**2))-c*abs(z)*s1*s2/(2.0)-c*(lx*(1-r)*s1*s2+ly*(1+r)*s1*s2)/2.0)*quad(lambda t:np.exp(-c*abs(z)*t*s1*s2+c*np.sqrt(abs(z)*t)*(np.sqrt(ly)*(1+r)*s1*s2+np.sqrt(lx)*(1-r)*s1*s2*(np.sqrt(1.0+1.0/t))))*(t*(1+t))**-0.5,a=0.0,b=np.inf)[0]/(np.sqrt(1.0-r**2)*np.pi*s1*s2)
    I1,I2,I3,I4=fn3(c,z,lx,ly,r),fn2(c,z,lx,ly,r),fn1(c,z,lx,ly,r),fn(c,z,lx,ly,r)
    #Is=[I1,I2,I3,I4]
    return (I1+I2+I3+I4)/4.


def getparsnchi(pars):
    mm1,mm2,ss1,ss2,rr=pars
    ss1=abs(ss1)
    ss2=abs(ss2)
    # rr=1.0/(1.0+np.exp(-rr))
    rr2=np.sqrt(ss1*ss2*(1-rr+1e-12)/2)
    rr1=np.sqrt(rr*ss1*ss2+rr2**2)
    ll1=abs((mm1**2*ss2**2+mm2**2*ss1**2-2*rr*mm1*mm2*ss1*ss2)+4*mm1*mm2*rr1**2)/(4*rr1**2+4*rr2**2)
    ll2=abs(ll1-mm1*mm2)
    return ll1,ll2,rr

def getfnFsxsy(z,s1,s2,m1,m2,rr):
    lx=getparsnchi([m1,m2,s1,s2,rr])[0]
    ly=getparsnchi([m1,m2,s1,s2,rr])[1]
    r=getparsnchi([m1,m2,s1,s2,rr])[2]
    c=2.0/((1-r**2)*s1*s2)
    #abs(np.minimum(z,0.0))
    # pz=(8.209536151601387*s1+m1)*(8.209536151601387*s2+m2)
    fn=lambda c,z,lx,ly,r:np.exp(z*r/((1-r**2)*s1*s2)-c*abs(z)/2.0-c*(lx*(1-r)+ly*(1+r))/2.0)*quad(lambda t:np.exp(-c*abs(z)*t-c*np.sqrt(abs(z)*t)*(np.sqrt(ly)*(1+r)+np.sqrt(lx)*(1-r)*(np.sqrt(1.0+1.0/t))))*(t*(1+t))**-0.5,a=abs(np.minimum(z,0.0)),b=np.inf)[0]/(np.sqrt(1.0-r**2)*np.pi*s1*s2)
    fn1=lambda c,z,lx,ly,r:np.exp(z*r/((1-r**2)*s1*s2)-c*abs(z)/(2.0)-c*(lx*(1-r)+ly*(1+r))/2.0)*quad(lambda t:np.exp(-c*abs(z)*t-c*np.sqrt(abs(z)*t)*(np.sqrt(ly)*(1+r)-np.sqrt(lx)*(1-r)*(np.sqrt(1.0+1.0/t))))*(t*(1+t))**-0.5,a=abs(np.minimum(z,0.0)),b=np.inf)[0]/(np.sqrt(1.0-r**2)*np.pi*s1*s2)
    fn2=lambda c,z,lx,ly,r:np.exp(z*r/((1-r**2)*s1*s2)-c*abs(z)/(2.0)-c*(lx*(1-r)+ly*(1+r))/2.0)*quad(lambda t:np.exp(-c*abs(z)*t-c*np.sqrt(abs(z)*t)*(-np.sqrt(ly)*(1+r)+np.sqrt(lx)*(1-r)*(np.sqrt(1.0+1.0/t))))*(t*(1+t))**-0.5,a=abs(np.minimum(z,0.0)),b=np.inf)[0]/(np.sqrt(1.0-r**2)*np.pi*s1*s2)
    fn3=lambda c,z,lx,ly,r:np.exp(z*r/((1-r**2)*s1*s2)-c*abs(z)/(2.0)-c*(lx*(1-r)+ly*(1+r))/2.0)*quad(lambda t:np.exp(-c*abs(z)*t+c*np.sqrt(abs(z)*t)*(np.sqrt(ly)*(1+r)+np.sqrt(lx)*(1-r)*(np.sqrt(1.0+1.0/t))))*(t*(1+t))**-0.5,a=abs(np.minimum(z,0.0)),b=np.inf)[0]/(np.sqrt(1.0-r**2)*np.pi*s1*s2)
    I1,I2,I3,I4=fn3(c,z,lx,ly,r),fn2(c,z,lx,ly,r),fn1(c,z,lx,ly,r),fn(c,z,lx,ly,r)
    Is=[I1,I2,I3,I4]
    return (I1+I2+I3+I4)/4.,Is

def getfnFsxsyN(z,s1,s2,m1,m2,rr):
    sign=np.sign(z)
    m1*=sign
    rr*=sign
    lx=getparsnchi([m1,m2,s1,s2,rr])[0]
    ly=getparsnchi([m1,m2,s1,s2,rr])[1]
    r=getparsnchi([m1,m2,s1,s2,rr])[2]
    c=2.0/((1-r**2)*s1*s2)
    
    #abs(np.minimum(z,0.0))
    # pz=(8.209536151601387*s1+m1)*(8.209536151601387*s2+m2)
    fn=lambda c,z,lx,ly,r:np.exp(sign*z*r/((1-r**2)*s1*s2)-c*abs(z)/2.0-c*(lx*(1-r)+ly*(1+r))/2.0)*quad(lambda t:np.exp(-c*abs(z)*t-c*np.sqrt(abs(z)*t)*(np.sqrt(ly)*(1+r)+np.sqrt(lx)*(1-r)*(np.sqrt(1.0+1.0/t))))*(t*(1+t))**-0.5,a=0.0,b=np.inf)[0]/(np.sqrt(1.0-r**2)*np.pi*s1*s2)
    fn1=lambda c,z,lx,ly,r:np.exp(sign*z*r/((1-r**2)*s1*s2)-c*abs(z)/(2.0)-c*(lx*(1-r)+ly*(1+r))/2.0)*quad(lambda t:np.exp(-c*abs(z)*t-c*np.sqrt(abs(z)*t)*(np.sqrt(ly)*(1+r)-np.sqrt(lx)*(1-r)*(np.sqrt(1.0+1.0/t))))*(t*(1+t))**-0.5,a=0.0,b=np.inf)[0]/(np.sqrt(1.0-r**2)*np.pi*s1*s2)
    fn2=lambda c,z,lx,ly,r:np.exp(sign*z*r/((1-r**2)*s1*s2)-c*abs(z)/(2.0)-c*(lx*(1-r)+ly*(1+r))/2.0)*quad(lambda t:np.exp(-c*abs(z)*t-c*np.sqrt(abs(z)*t)*(-np.sqrt(ly)*(1+r)+np.sqrt(lx)*(1-r)*(np.sqrt(1.0+1.0/t))))*(t*(1+t))**-0.5,a=0.0,b=np.inf)[0]/(np.sqrt(1.0-r**2)*np.pi*s1*s2)
    fn3=lambda c,z,lx,ly,r:np.exp(sign*z*r/((1-r**2)*s1*s2)-c*abs(z)/(2.0)-c*(lx*(1-r)+ly*(1+r))/2.0)*quad(lambda t:np.exp(-c*abs(z)*t+c*np.sqrt(abs(z)*t)*(np.sqrt(ly)*(1+r)+np.sqrt(lx)*(1-r)*(np.sqrt(1.0+1.0/t))))*(t*(1+t))**-0.5,a=0.0,b=np.inf)[0]/(np.sqrt(1.0-r**2)*np.pi*s1*s2)
    I1,I2,I3,I4=fn3(c,z,lx,ly,r),fn2(c,z,lx,ly,r),fn1(c,z,lx,ly,r),fn(c,z,lx,ly,r)
    Is=[I1,I2,I3,I4]
    return (I1+I2+I3+I4)/4.,Is


def Pnormprodpdfbynchi2gku(z,pars,ispdf,scalar,bb):
    mm1,mm2,ss1,ss2,rr=pars
    ss1=abs(ss1)
    ss2=abs(ss2)
    rr=1.0/(1.0+np.exp(-rr))
    rr2=np.sqrt(ss1*ss2*(1-rr+1e-12)/2)
    rr1=np.sqrt(rr*ss1*ss2+rr2**2)
    ll1=abs((mm1**2*ss2**2+mm2**2*ss1**2-2*rr*mm1*mm2*ss1*ss2)+4*mm1*mm2*rr1**2)/(4*rr1**2+4*rr2**2)
    ll2=abs(ll1-mm1*mm2 )
    
    
    if ispdf:
        if scalar:
            out=adGKIntegrate(f=nrmprdnchicorru, a=np.sqrt(abs(np.minimum(z,0.0))),b=np.sqrt(bb),tol=1e-9,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0]
        else:
            out=[adGKIntegrate(f=nrmprdnchicorru, a=np.sqrt(abs(np.minimum(zz,0.0))),b=np.sqrt(bb),tol=1e-9,args=(zz+1e-15,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0]  for zz in z]
        
    else:
    
        
        if scalar:
            out=adGKIntegrate(f=crmprdnchicorru, a=np.sqrt(abs(np.minimum(z,0.0))),b=np.sqrt(bb),tol=1e-9,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0]
        else:
            out=[adGKIntegrate(f=crmprdnchicorru, a=np.sqrt(abs(np.minimum(zz,0.0))),b=np.sqrt(bb),tol=1e-9,args=(zz+1e-12,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0]  for zz in z]
            #Parallel(n_jobs=8)(delayed(lambda zz:adGKIntegrate(f=crmprdnchicorr, a=abs(np.minimum(zz,0.0)),b=bb,tol=1e-9,args=(zz,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0])(zz)  for zz in z)
    return out



def nchidirect(z,pars,ispdf):
    mm1,mm2,ss1,ss2,rr=pars
    rr=1.0/(1.0+np.exp(-rr))
    ss1=abs(ss1)
    ss2=abs(ss2)
    rr2=np.sqrt(ss1*ss2*(1-rr)/2)
    rr1=np.sqrt(rr*ss1*ss2+rr2**2)
    ll1=((mm1**2*ss2**2+mm2**2*ss1**2-2*rr*mm1*mm2*ss1*ss2)+4*mm1*mm2*rr1**2)/(4*rr1**2+4*rr2**2)
    ll2=ll1-mm1*mm2
    cfcorr=lambda x,z,ll1,ll2,sgm1,sgm2:ncx2.pdf(x,1,((ll2/rr2**2)),loc=0.0,scale=sgm2)*ncx2.cdf((x+z),1,((ll1/rr1**2)),loc=0.0,scale=sgm1)
    fcorr=lambda x,z,ll1,ll2,sgm1,sgm2:ncx2.pdf(x,1,((ll2/rr2**2)),loc=0.0,scale=sgm2)*ncx2.pdf(x+z,1,((ll1/rr1**2)),loc=0.0,scale=sgm1)
    
    if ispdf:
        out=quad(fcorr,0,np.inf,args=(z,ll1,ll2,rr1**2,rr2**2))[0]
    else:
        out=quad(cfcorr,0,np.inf,args=(z,ll1,ll2,rr1**2,rr2**2))[0]
    return out

def NRProdInv(x0,pars,xq):
    mm1,mm2,ss1,ss2,rr=pars
    if x0 is None:
        x0=norm.ppf(xq,loc=mm2,scale=abs(ss2))*norm.ppf(xq,loc=mm1,scale=abs(ss1))+abs(ss1)*abs(ss2)*rr
    C=1.0 
    #0.9998843169275278
    #x0=1e-3  
    eps=1.
    it=0
    firsteval=Pnormprodpdfbynchi2(x0,pars,False,True)/C
    #if (firsteval>xq):
     #   return print('Fun at x=0.0 is greater than q')
    #else:        
    while eps>1e-6:
        it+=1
        fundiff=Pnormprodpdfbynchi2(x0,pars,False,True)/C-xq
        x1=x0-(fundiff)/Pnormprodpdfbynchi2(x0,pars,True,True)
        eps=abs(fundiff)
        x0=x1
        print(x1)
        print(it)
        # if x0<0.:
        #     x0=xq/10.
        if it>10:
            break;
        print("final Convergence: %.8f" %(eps))
    return x1

def NRProdInvnChi2(x0,pars,xq):
    mm1,mm2,ss1,ss2,rr=pars
    C=1.0
    #x0=1e-3  
    eps=1.
    it=0
    firsteval=nchidirect(x0,pars,False)/C
    #if (firsteval>xq):
     #   return print('Fun at x=0.0 is greater than q')
    #else:        
    while eps>1e-7:
        it+=1
        fundiff=nchidirect(x0,pars,False)/C-xq
        x1=x0-(fundiff)/nchidirect(x0,pars,True)
        eps=abs(fundiff)
        x0=x1
        print(x1)
        print(it)
        if x0<0.:
            x0=xq/10.
        if it>10:
            break;
        print("final Convergence: %.8f" %(eps))
    return x1

def normprodpdfbynchi2(z,mm1,mm2,ss1,ss2,rr,ispdf):
    rr2=np.sqrt(ss1*ss2*(1-rr)/2)
    rr1=np.sqrt(rr*ss1*ss2+rr2**2)
    ll1=((mm1**2*ss2**2+mm2**2*ss1**2-2*rr*mm1*mm2*ss1*ss2)+4*mm1*mm2*rr1**2)/(4*rr1**2+4*rr2**2)
    ll2=ll1-mm1*mm2
    nchix2=lambda x,l:(fastNpdf(np.sqrt(x)+np.sqrt(l))+fastNpdf(np.sqrt(x)-np.sqrt(l)))*0.5/np.sqrt(x)
    cnchix2=lambda x,l:(norm.cdf(np.sqrt(x)+np.sqrt(l))+norm.cdf(np.sqrt(x)-np.sqrt(l))-1)
    crmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)
    nrmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*nchix2((x+z)/sgm1,ll1)/(sgm1*sgm2)
    if ispdf:
        
        out=quad(nrmprdnchicorr, abs(np.minimum(z,0.0)),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
        #
        
    else:
        crmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)
        out=quad(crmprdnchicorr, abs(np.minimum(z,0.0)),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
    return out

def nprodMLE(pars,ispdf,z,withmu,nocorr):
    mm1,mm2,ss1,ss2,rr=pars
    if nocorr:
        rr=0.0
    else:
        rr=1.0/(1.0+np.exp(-rr))
    if withmu:
        
        return -np.sum(np.log(Pnormprodpdfbynchi2gk(z,pars,ispdf,False,5.0)))
    else:
        return -np.sum(np.log(NPCorr(z,0.0,0.0,abs(ss1),abs(ss2),rr)))
    
def NIGcdf(ub,sgmn,k,thetan,mun):
    if ub<0.0:
        I=gk(lambda x:NIGddnew(x,sgmn,k,thetan,mun),a=-7.0,b=ub)
    else:
        I=gk(lambda x:NIGddnew(x,sgmn,k,thetan,mun),a=-7.0,b=0.0)+gk(lambda x:NIGddnew(x,sgmn,k,thetan,mun),a=0.0,b=ub)
    return I

def vNIGcdf(ub,sgmn,k,thetan,mun):
    cdfv=[NIGcdf(u,sgmn,k,thetan,mun) for u in ub]
    return cdfv

 

def VGcdf(ub,sgm,nu,theta,mu):
    if ub<0.0:
        I=gk(lambda x:VGddnew(x,sgm,nu,theta,mu),a=-7.0,b=ub)
    else:
        I=gk(lambda x:VGddnew(x,sgm,nu,theta,mu),a=-7.0,b=0.0)+gk(lambda x:VGddnew(x,sgm,nu,theta,mu),a=0.0,b=ub)
    return I

 

def vVGcdf(ub,sgm,nu,theta,mu):
    cdfv=[VGcdf(u,sgm,nu,theta,mu) for u in ub]
    return cdfv
#%%
mm1,mm2,ss1,ss2,rr=1.0,0.5,2.0,0.2,0.6
bivr=mvn.rvs([mm1,mm2],cov=[[ss1**2,rr*ss1*ss2],[rr*ss1*ss2,ss2**2]],size=100000)
rvsvcorr=bivr[:,0]*bivr[:,1]
xmincorr=rvsvcorr.min()
xmaxcorr=rvsvcorr.max()
zzcorr=np.linspace(xmincorr,xmaxcorr,100)
ecdfcor=ECDF(rvsvcorr)
cdfsimcorr=ecdfcor(zzcorr)
print(rvsvcorr.max())
print(rvsvcorr.min())
zzcorr=np.linspace(xmincorr,xmaxcorr,100)
fchcorr=Pnormprodpdfbynchi2(zzcorr,[mm1,mm2,ss1,ss2,rr],True,False)
semiApdf=[getfnFsxsyN(xz,ss1,ss2,mm1,mm2,rr)[0] for xz in zzcorr]
#%%
plt.figure(figsize=(14, 8))
sns.distplot(rvsvcorr,kde=False,hist=True,norm_hist=True,color='orange')
#sns.histplot(rvsvcorr,kde=False,stat='density')
plt.plot(zzcorr,fchcorr,'--',color='blue')
plt.plot(zzcorr,semiApdf,'o',color='green')
plt.legend(['Eqn_10','Eqn_12','Simulation'])
# plt.grid(True)
plt.xlabel("z")
plt.ylabel("F_Z(z)")
plt.show()
