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
from multiprocessing import Pool
import pandas_datareader as pdrd
import quandl as qdl
import datetime as dt
import numexpr as ne
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
import math
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
from QuantLib import GaussKronrodNonAdaptive
gk=GaussKronrodNonAdaptive(1e-6,50,1e-6)
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
#os.chdir('C:\\Users\\Administrator\\Desktop\\abdl\\calibre_HCJ')

@jit(nopython=True)
def getGKL():
   #  XGK=np.array([
   #     -0.995657163025808080735527280689003,
   #     -0.973906528517171720077964012084452,
   #     -0.930157491355708226001207180059508,
   #     -0.865063366688984510732096688423493,
   #     -0.780817726586416897063717578345042,
   #     -0.679409568299024406234327365114874,
   #     -0.562757134668604683339000099272694,
   #     -0.433395394129247190799265943165784,
   #     -0.294392862701460198131126603103866,
   #     -0.148874338981631210884826001129720,
   #      0.000000000000000000000000000000000,
   #      0.148874338981631210884826001129720,
   #      0.294392862701460198131126603103866,
   #      0.433395394129247190799265943165784,
   #      0.562757134668604683339000099272694,
   #      0.679409568299024406234327365114874,
   #      0.780817726586416897063717578345042,
   #      0.865063366688984510732096688423493,
   #      0.930157491355708226001207180059508,
   #      0.973906528517171720077964012084452,
   #      0.995657163025808080735527280689003
   # ])
   #  WGK = np.array([
   #      0.011694638867371874278064396062192,
   #      0.032558162307964727478818972459390,
   #      0.054755896574351996031381300244580,
   #      0.075039674810919952767043140916190,
   #      0.093125454583697605535065465083366,
   #      0.109387158802297641899210590325805,
   #      0.123491976262065851077958109831074,
   #      0.134709217311473325928054001771707,
   #      0.142775938577060080797094273138717,
   #      0.147739104901338491374841515972068,
   #      0.149445554002916905664936468389821,
   #      0.147739104901338491374841515972068,
   #      0.142775938577060080797094273138717,
   #      0.134709217311473325928054001771707,
   #      0.123491976262065851077958109831074,
   #      0.109387158802297641899210590325805,
   #      0.093125454583697605535065465083366,
   #      0.075039674810919952767043140916190,
   #      0.054755896574351996031381300244580,
   #      0.032558162307964727478818972459390,
   #      0.011694638867371874278064396062192
   #  ])

   #  WG = np.array([
   #      0.066671344308688137593568809893332,
   #      0.149451349150580593145776339657697,
   #      0.219086362515982043995534934228163,
   #      0.269266719309996355091226921569469,
   #      0.295524224714752870173892994651338,
   #      0.295524224714752870173892994651338,
   #      0.269266719309996355091226921569469,
   #      0.219086362515982043995534934228163,
   #      0.149451349150580593145776339657697,
   #      0.066671344308688137593568809893332
   #  ])

    
    WG=np.empty(6,np.float64)
    WGK=np.empty(12,np.float64)
    XGK=np.empty(12,np.float64)
    WG[0]=0.272925086777901
    WG[1]=5.56685671161745E-02
    WG[2]=0.125580369464905
    WG[3]=0.186290210927735
    WG[4]=0.233193764591991
    WG[5]=0.262804544510248
     
    WGK[0]=0.136577794711118
    WGK[1]=9.76544104596129E-03
    WGK[2]=2.71565546821044E-02
    WGK[3]=4.58293785644267E-02
    WGK[4]=6.30974247503748E-02
    WGK[5]=7.86645719322276E-02
    WGK[6]=9.29530985969007E-02
    WGK[7]=0.105872074481389
    WGK[8]=0.116739502461047
    WGK[9]=0.125158799100319
    WGK[10]=0.131280684229806
    WGK[11]=0.135193572799885
    XGK[0]=0.0#
    XGK[1]=0.996369613889543
    XGK[2]=0.978228658146057
    XGK[3]=0.941677108578068
    XGK[4]=0.887062599768095
    XGK[5]=0.816057456656221
    XGK[6]=0.730152005574049
    XGK[7]=0.630599520161965
    XGK[8]=0.519096129206812
    XGK[9]=0.397944140952378
    XGK[10]=0.269543155952345
    XGK[11]=0.136113000799362
    return WG,WGK,XGK
@jit(nopython=True,parallel=True)
def gauss_kronrod_quadrature(f, a, b, args):
    midpoint = (a + b) / 2.0
    half_range = (b - a) / 2.0
    WG, WGK, XGK= getGKL()  
    GKI = WGK[0] * f(midpoint, *args)
    GI = WG[0] * f(midpoint, *args)

    for j in range(1, len(XGK)):
        shifted_val1 = midpoint - half_range * XGK[j]
        shifted_val2 = midpoint + half_range * XGK[j]
        val1 = f(shifted_val1, *args)
        val2 = f(shifted_val2, *args)
        GKI += WGK[j] * (val1 + val2)
        if j % 2 == 0:
            GI += WG[int(j / 2)] * (val1 + val2)

    GK = GKI * half_range
    G = GI * half_range
    return GK, G

@jit(nopython=True,parallel=True)
def gauss_kronrod_quadraturenoarg(f, a, b):
    midpoint = (a + b) / 2.0
    half_range = (b - a) / 2.0
    WG, WGK, XGK= getGKL()  
    GKI = WGK[0] * f(midpoint)
    GI = WG[0] * f(midpoint)

    for j in range(1, len(XGK)):
        shifted_val1 = midpoint - half_range * XGK[j]
        shifted_val2 = midpoint + half_range * XGK[j]
        val1 = f(shifted_val1)
        val2 = f(shifted_val2)
        GKI += WGK[j] * (val1 + val2)
        if j % 2 == 0:
            GI += WG[int(j / 2)] * (val1 + val2)

    GK = GKI * half_range
    G = GI * half_range
    return GK, G

@jit(nopython=True,parallel=True)
def adGKIntegrate(f, a, b, tol, args, max_iter=1000):
    stack = [(a, b)]
    total_result = 0.0
    total_resultg = 0.0
    total_error = 0.0
    iter_count = 0
    eps=1.0

    while stack and iter_count < max_iter:
        a, b = stack.pop()
        GK, G = gauss_kronrod_quadrature(f, a, b, args)
        error = np.abs(GK - G)

        if error < tol:
            total_result += GK
            total_resultg+=G
           
        else:
            midpoint = (a + b) / 2.0
            stack.append((a, midpoint))
            stack.append((midpoint, b))
            total_error += error
            
        eps=abs(total_result-total_resultg)    
        iter_count += 1

    return total_result, eps, iter_count


@jit(nopython=True,parallel=True)
def adGKIntegratenoarg(f, a, b, tol, max_iter=1000):
    stack = [(a, b)]
    total_result = 0.0
    total_resultg = 0.0
    total_error = 0.0
    iter_count = 0
    eps=1.0

    while stack and iter_count < max_iter:
        a, b = stack.pop()
        GK, G = gauss_kronrod_quadraturenoarg(f, a, b)
        error = np.abs(GK - G)

        if error < tol:
            total_result += GK
            total_resultg += G
        else:
            midpoint = (a + b) / 2.0
            stack.append((a, midpoint))
            stack.append((midpoint, b))
            total_error += error
        eps=abs(total_result-total_resultg)
       
        iter_count += 1

    return total_result, eps, iter_count


@jit(nopython=True)
def gauss_kronrod_nodes_weights():
    # Gauss-Kronrod 10-21 nodes and weights
    gk21_nodes = np.array([
        -0.995657163025808080735527280689003,
        -0.973906528517171720077964012084452,
        -0.930157491355708226001207180059508,
        -0.865063366688984510732096688423493,
        -0.780817726586416897063717578345042,
        -0.679409568299024406234327365114874,
        -0.562757134668604683339000099272694,
        -0.433395394129247190799265943165784,
        -0.294392862701460198131126603103866,
        -0.148874338981631210884826001129720,
         0.000000000000000000000000000000000,
         0.148874338981631210884826001129720,
         0.294392862701460198131126603103866,
         0.433395394129247190799265943165784,
         0.562757134668604683339000099272694,
         0.679409568299024406234327365114874,
         0.780817726586416897063717578345042,
         0.865063366688984510732096688423493,
         0.930157491355708226001207180059508,
         0.973906528517171720077964012084452,
         0.995657163025808080735527280689003
    ])

    gk21_weights = np.array([
        0.011694638867371874278064396062192,
        0.032558162307964727478818972459390,
        0.054755896574351996031381300244580,
        0.075039674810919952767043140916190,
        0.093125454583697605535065465083366,
        0.109387158802297641899210590325805,
        0.123491976262065851077958109831074,
        0.134709217311473325928054001771707,
        0.142775938577060080797094273138717,
        0.147739104901338491374841515972068,
        0.149445554002916905664936468389821,
        0.147739104901338491374841515972068,
        0.142775938577060080797094273138717,
        0.134709217311473325928054001771707,
        0.123491976262065851077958109831074,
        0.109387158802297641899210590325805,
        0.093125454583697605535065465083366,
        0.075039674810919952767043140916190,
        0.054755896574351996031381300244580,
        0.032558162307964727478818972459390,
        0.011694638867371874278064396062192
    ])

    g10_nodes = np.array([
        -0.973906528517171720077964012084452,
        -0.865063366688984510732096688423493,
        -0.679409568299024406234327365114874,
        -0.433395394129247190799265943165784,
        -0.148874338981631210884826001129720,
         0.148874338981631210884826001129720,
         0.433395394129247190799265943165784,
         0.679409568299024406234327365114874,
         0.865063366688984510732096688423493,
         0.973906528517171720077964012084452
    ])

    g10_weights = np.array([
        0.066671344308688137593568809893332,
        0.149451349150580593145776339657697,
        0.219086362515982043995534934228163,
        0.269266719309996355091226921569469,
        0.295524224714752870173892994651338,
        0.295524224714752870173892994651338,
        0.269266719309996355091226921569469,
        0.219086362515982043995534934228163,
        0.149451349150580593145776339657697,
        0.066671344308688137593568809893332
    ])

    return gk21_nodes, gk21_weights, g10_nodes, g10_weights

@jit(nopython=True,parallel=True)
def gauss_kronrod_integration(f, a, b,args):
    nodes, weights, g_nodes, g_weights = gauss_kronrod_nodes_weights()

    
    mid = (a + b) / 2.0
    half_length = (b - a) / 2.0
    integral = 0.0
    integralg=0.0
    error = 0.0

    for i in range(len(nodes)):
        x = mid + half_length * nodes[i]
        fx = f(x,*args)
        integral += weights[i] * fx
        xg = mid + half_length * g_nodes[i]
        fxg = f(xg,*args)
        integralg += g_weights[i] * fxg

    integral *= half_length
    integralg *= half_length
    error+= np.abs((integral - integralg) / 3.0)
    
    return integral, error
@jit(nopython=True)
def adGK_integration(f, a, b, tol,args,maxiter=1000):
    stack = [(a, b)]
    niter=0
    integral, error=gauss_kronrod_integration(f, a, b,args)
    while error>tol:
        a, b = stack.pop()
        integral, error=gauss_kronrod_integration(f, a, b,args)
        midpoint = (a + b) / 2.0
        stack.append((a, midpoint))
        stack.append((midpoint, b))
        niter+=1
        if niter>(maxiter-1):
            break
        print('GK error',error)
    return integral,error

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
@jit(nopython=True,parallel=True)
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
    rr=-1+2.0/(1.0+np.exp(-rr))
    bivr=multivariate_normal.rvs([mm1,mm2],cov=[[ss1**2,rr*ss1*ss2],[rr*ss1*ss2,ss2**2.]],size=nsim)
    return bivr[:,0]*bivr[:,1]

def Pnormprodpdfbynchi2(z,pars,ispdf,scalar):
    mm1,mm2,ss1,ss2,rr=pars
    ss1=abs(ss1)
    ss2=abs(ss2)
    rr=-1+2.0/(1.0+np.exp(-rr))
    rr2=np.sqrt(ss1*ss2*(1-rr)/2)
    rr1=np.sqrt(rr*ss1*ss2+rr2**2)
    ll1=((mm1**2*ss2**2+mm2**2*ss1**2-2*rr*mm1*mm2*ss1*ss2)+4*mm1*mm2*rr1**2)/(4*rr1**2+4*rr2**2)
    ll2=ll1-mm1*mm2
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
        else:
            out=Parallel(n_jobs=8)(delayed(lambda zz:quad(crmprdnchicorr, abs(np.minimum(zz,0.0)),np.inf,epsabs=1e-5,args=(zz,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0])(zz)  for zz in z)
    return out

@jit(nopython=True)
def nchix2(x,l):
    return (fastNpdf(np.sqrt(x)+np.sqrt(l))+fastNpdf(np.sqrt(x)-np.sqrt(l)))*0.5/np.sqrt(x)

@jit(nopython=True)
def cnchix2(x,l):
    return (appNormCDF(np.sqrt(x)+np.sqrt(l))+appNormCDF(np.sqrt(x)-np.sqrt(l)))-1.0


@jit(nopython=True)
def crmprdnchicorr(x,z,ll1,ll2,sgm1,sgm2):
    return nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)/sgm2

@jit(nopython=True)
def nrmprdnchicorr(x,z,ll1,ll2,sgm1,sgm2):
    return nchix2(x/sgm2,ll2)*nchix2((x+z)/sgm1,ll1)/(sgm1*sgm2+1e-12)

def Pnormprodpdfbynchi2gk(z,pars,ispdf,scalar,bb):
    mm1,mm2,ss1,ss2,rr=pars
    ss1=abs(ss1)
    ss2=abs(ss2)
    rr=-1.0+2.0/(1.0+np.exp(-rr))
    rr2=np.sqrt(ss1*ss2*(1-rr+1e-12)/2)
    rr1=np.sqrt(rr*ss1*ss2+rr2**2)
    ll1=((mm1**2*ss2**2+mm2**2*ss1**2-2*rr*mm1*mm2*ss1*ss2)+4*mm1*mm2*rr1**2)/(4*rr1**2+4*rr2**2)
    ll2=ll1-mm1*mm2 
    
    # nchix2=lambda x,l:(fastNpdf(np.sqrt(x)+np.sqrt(l))+fastNpdf(np.sqrt(x)-np.sqrt(l)))*0.5/np.sqrt(x)
    # cnchix2=lambda x,l:(appNormCDF(np.sqrt(x)+np.sqrt(l))+appNormCDF(np.sqrt(x)-np.sqrt(l)))-1.0
    # crmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)
    # nrmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*nchix2((x+z)/sgm1,ll1)/(sgm1*sgm2)
    if ispdf:
        if scalar:
            out=adGKIntegrate(f=nrmprdnchicorr, a=abs(np.minimum(z,0.0)),b=bb,tol=1e-9,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0]
        else:
            out=[adGKIntegrate(f=nrmprdnchicorr, a=abs(np.minimum(zz,0.0)),b=bb,tol=1e-9,args=(zz+1e-15,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0]  for zz in z]
            #Parallel(n_jobs=8)(delayed(lambda zz:adGKIntegrate(f=nrmprdnchicorr, a=abs(np.minimum(zz,0.0)),b=bb,tol=1e-9,args=(zz,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0])(zz)  for zz in z)
        #quad(nrmprdnchicorr, abs(np.minimum(z,0.0)),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
        
    else:
    
        
        if scalar:
            out=adGKIntegrate(f=crmprdnchicorr, a=abs(np.minimum(z,0.0)),b=bb,tol=1e-9,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0]
        else:
            out=[adGKIntegrate(f=crmprdnchicorr, a=abs(np.minimum(zz,0.0)),b=bb,tol=1e-9,args=(zz+1e-12,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0]  for zz in z]
            #Parallel(n_jobs=8)(delayed(lambda zz:adGKIntegrate(f=crmprdnchicorr, a=abs(np.minimum(zz,0.0)),b=bb,tol=1e-9,args=(zz,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2),max_iter=1000)[0])(zz)  for zz in z)
    return out

def nchidirect(z,pars,ispdf):
    mm1,mm2,ss1,ss2,rr=pars
    rr=-1+2.0/(1.0+np.exp(-rr))
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
        x0=ncx2.norm.ppf(xq,loc=mm2,scale=abs(ss2))*ncx2.norm.ppf(xq,loc=mm1,scale=abs(ss1))+abs(ss1)*abs(ss2)*rr
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
    cnchix2=lambda x,l:(appNormCDF(np.sqrt(x)+np.sqrt(l))+appNormCDF(np.sqrt(x)-np.sqrt(l))-1)
    crmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)
    nrmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*nchix2((x+z)/sgm1,ll1)/(sgm1*sgm2)
    if ispdf:
        
        out=quad(nrmprdnchicorr, abs(np.minimum(z,0.0)),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
        #
        
    else:
        crmprdnchicorr=lambda x,z,ll1,ll2,sgm1,sgm2:nchix2(x/sgm2,ll2)*cnchix2((x+z)/sgm1,ll1)
        out=quad(crmprdnchicorr, abs(np.minimum(z,0.0)),np.inf,epsabs=1e-5,args=(z,(ll1/rr1**2),(ll2/rr2**2),rr1**2,rr2**2))[0]
    return out

def NPCorr(x,m1,m2,s1,s2,r):
    A=np.exp(-0.5*m1**2/(s1**2*(1-r**2))-0.5*m2**2/(s2**2*(1-r**2))+m1*m2*r/(1-r**2))
    #A=1.
    f=1.0/np.sqrt(1.0-r**2)/(np.pi*s1*s2)*np.exp(x*r/((1.0-r**2)*(s1*s2)))*A*kv(0,np.abs(x)/((1.0-r**2)*(s1*s2)))
    return f

def nprodMLE(pars,ispdf,z,withmu,nocorr):
    mm1,mm2,ss1,ss2,rr=pars
    if nocorr:
        rr=0.0
    else:
        rr=-1+2.0/(1.0+np.exp(-rr))
    if withmu:
        
        return -np.sum(np.log(Pnormprodpdfbynchi2(z,pars,ispdf,False)))
    else:
        return -np.sum(np.log(NPCorr(z,0.0,0.0,abs(ss1),abs(ss2),rr)))

def rho_to_rr(rho):
    if rho==1:
        rr=np.inf
    elif rho==-1:
        rr=-np.inf
    else:
        rr=-np.log(2/(rho+1)-1)
    return rr

def NIGcdf(ub,sgmn,k,thetan,mun):
    if ub<0.0:
        I=gk(lambda x:NIGddnew(x,sgmn,k,thetan,mun),a=-7.0,b=ub)
    else:
        I=gk(lambda x:NIGddnew(x,sgmn,k,thetan,mun),a=-7.0,b=0.0)+gk(lambda x:NIGddnew(x,sgmn,k,thetan,mun),a=0.0,b=ub)
    return I

def vNIGcdf(ub,sgmn,k,thetan,mun):
    cdfv=[NIGcdf(u,sgmn,k,thetan,mun) for u in ub]
    return cdfv

# def NIGcdfgk(ub,sgmn,k,thetan,mun):
#     if ub<0.0:
#         I=adGKIntegrate(f=NIGddnew, a=-7,b=ub,tol=1e-9,args=(sgmn,k,thetan,mun),max_iter=1000)[0]
#     else:
#         I=adGKIntegrate(f=NIGddnew, a=-7,b=0,tol=1e-9,args=(sgmn,k,thetan,mun),max_iter=1000)[0]+adGKIntegrate(f=NIGddnew, a=0,b=ub,tol=1e-9,args=(sgmn,k,thetan,mun),max_iter=1000)[0]
#     return I

# def vNIGcdfgk(ub,sgmn,k,thetan,mun):
#     cdfv=[NIGcdfgk(u,sgmn,k,thetan,mun) for u in ub]
#     return cdfv

def VGcdf(ub,sgm,nu,theta,mu):
    if ub<0.0:
        I=gk(lambda x:VGddnew(x,sgm,nu,theta,mu),a=-7.0,b=ub)
        
    else:
        I=gk(lambda x:VGddnew(x,sgm,nu,theta,mu),a=-7.0,b=0.0)+gk(lambda x:VGddnew(x,sgm,nu,theta,mu),a=0.0,b=ub)
    return I

def vVGcdf(ub,sgm,nu,theta,mu):
    cdfv=[VGcdf(u,sgm,nu,theta,mu) for u in ub]
    return cdfv

# def VGcdfgk(ub,sgm,nu,theta,mu):
#     if ub<0.0:
#         I=adGKIntegrate(f=VGddnew, a=-7,b=ub,tol=1e-9,args=(sgm,nu,theta,mu),max_iter=1000)[0]
#     else:
#         I=adGKIntegrate(f=VGddnew, a=-7,b=0,tol=1e-9,args=(sgm,nu,theta,mu),max_iter=1000)[0]+adGKIntegrate(f=VGddnew, a=0,b=ub,tol=1e-9,args=(sgm,nu,theta,mu),max_iter=1000)[0]
#     return I

# def vVGcdfgk(ub,sgm,nu,theta,mu):
#     cdfv=[VGcdfgk(u,sgm,nu,theta,mu) for u in ub]
#     return cdfv

def NP_sim(mc,optpar,ssize,mm1,mm2,ss1,ss2,rho,onesample=False):
    sampsim=simNormProd(optpar,ssize)
    if onesample==False:
        G0=getnchix2rvs(mm1,mm2,ss1,ss2,rho,ssize)
        kst=ks_2samp(sampsim,G0,alternative='two-sided')
    else:
        pars=(mm1,mm2,ss1,ss2,rho_to_rr(rho))
        kst=kstest(sampsim,Pnormprodpdfbynchi2,args=(pars,False,False),alternative='two-sided')
    ks_stat=kst[0]
    ks_pval=kst[1]
    return (ks_stat,ks_pval)

def NP_sim_gk(mc,optpar,ssize,mm1,mm2,ss1,ss2,rho,onesample=False):
    sampsim=simNormProd(optpar,ssize)
    if onesample==False:
        G0=getnchix2rvs(mm1,mm2,ss1,ss2,rho,ssize)
        kst0=ks_2samp(sampsim,G0,alternative='two-sided')
        sgmVG,nuVG,thetaVG,muVG=VGmom(sampsim)      
        kst1=kstest(sampsim,vVGcdf,args=(sgmVG,nuVG,thetaVG,muVG),alternative='two-sided')
        sgmNIG,kNIG,thetaNIG,muNIG=NIGmom(sampsim)
        kst2=kstest(sampsim,vNIGcdf,args=(sgmNIG,kNIG,thetaNIG,muNIG),alternative='two-sided')
    else:
        pars=(mm1,mm2,ss1,ss2,rho_to_rr(rho))
        kst0=kstest(sampsim,Pnormprodpdfbynchi2gk,args=(pars,False,False,7),alternative='two-sided')
        # kst1=kstest(sampsim,Pnormprodpdfbynchi2,args=(pars,False,False),alternative='two-sided')
        sgmVG,nuVG,thetaVG,muVG=VGmom(sampsim)      
        kst1=kstest(sampsim,vVGcdf,args=(sgmVG,nuVG,thetaVG,muVG),alternative='two-sided')
        sgmNIG,kNIG,thetaNIG,muNIG=NIGmom(sampsim)
        kst2=kstest(sampsim,vNIGcdf,args=(sgmNIG,kNIG,thetaNIG,muNIG),alternative='two-sided')
    ks_stat0=kst0[0]
    ks_pval0=kst0[1]
    ks_stat1=kst1[0]
    ks_pval1=kst1[1]    
    ks_stat2=kst2[0]
    ks_pval2=kst2[1]
    return (ks_stat0,ks_pval0,ks_stat1,ks_pval1,ks_stat2,ks_pval2)
    # return (ks_stat0,ks_pval0,ks_stat1,ks_pval1)


    