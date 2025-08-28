# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 00:33:23 2024

@author: aheki
"""
import numpy as np
from numba import jit
from joblib import Parallel, delayed
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
        
    
