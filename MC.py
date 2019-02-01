#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:00:00 2019

@author: guysatat
"""

import numpy as np
from multiprocessing import Pool
from itertools import repeat



#Assumptions:
# 1. Detector is circular, at 0,0,0 on the x-y plane and points towards the positive z axis
# 2. Doesn't dupport reflective targets.
# 3. Support only impulse pencil beam.



def lunchBatch(batchSize = 1000,
               muS  =  1.0, g = 0.85,
               source = {'r': np.array([0.0, 0.0, 0.0]),
                          'mu': np.array([0.0, 0.0, 1.0]),
                          'method': 'pencil', 'time_profile': 'delta'},
               detR = 1.0,
               max_N = 1e5,
               max_distance_from_det = 1000.0,
               ret_cols = [1,2,7,8]):
    
    
    # Assume all photons start the same (impulse pencil)
    r0 = source['r']
    nu0 = source['mu']
    d0 = 0.0
    n0 = 0        
    arg = (muS, g, r0, nu0, d0, n0, detR, max_N, max_distance_from_det, ret_cols)
    with Pool() as tp:
        data = tp.starmap(propPhoton, repeat(arg, batchSize))
    data = np.array(data)
    
#    THIS IS A NON_PARALLEL HISTORIC VERSION
#     data = np.zeros(shape=(batchSize, len(ret_cols)), dtype=float)
#     for i in range(batchSize):
#        #r0, nu0, d0, n0 = simSource(source)
#        r0 = source['r']
#        nu0 = source['mu']
#        d0 = 0.0
#        n0 = 0        
#        data[i, :] = propPhoton(muS = muS, g = g,
#                       r0  = r0,
#                       nu0 = nu0,
#                       d0 = d0, n0 = n0,
#                       detR = detR,
#                       max_N = max_N,
#                       max_distance_from_det = max_distance_from_det,
#                       ret_cols=ret_cols )

    
    return data


#Return columns:
#    0    1  2  3   4     5     6    7  8
# Status, x ,y, z, mu_x, mu_y, mu_z, d, n
def propPhoton(muS  =  1.0, g = 0.85,
               r0  = np.array([0.0,0.0,0.0]),
               nu0 = np.array([0.0,0.0,1.0]),
               d0 = 0.0, n0 = 0,
               detR = 1.0,
               max_N = 1e5,
               max_distance_from_det = 1000.0,
               ret_cols = [1,2,7,8]):
    np.random.seed()
    r =  r0.copy()
    nu = nu0.copy()
    d = d0
    n = n0        
    
    detR2 = detR**2
    ret = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 , -1.0, -1.0])
    while True:
        #print(r, nu, d, n)
        # Calc random prop params
        cd = - np.log(np.random.random()) / muS
        
        t_r = r + cd * nu
        
        #check if we hit the detector
        if t_r[2] <= 0: #did we pass the detector?
            cd = - r[2] / nu[2]
            t_r = r + cd * nu
            if t_r[0]**2 + t_r[1]**2 < detR2:
                d+=cd
                n+=1
                r = t_r
                ret = np.array([0, r[0],r[1],r[2],nu[0],nu[1],nu[2],d,n])
                break
            else:  # if we passed the detector and didn't hit it we should restart
                ret[0] = 1
                break
            
        # prop photon
        for i in range(3):
            r[i] = t_r[i] 
        d += cd #prop distance
        n += 1 #increase scatter counter
        
        #scatter to new angle
        psi = 2*np.pi*np.random.random()
        mu = 1/(2*g) * (1+g**2 - ( (1-g*g)/(1-g+2*g*np.random.random()))**2)
        
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        sqrt_mu = np.sqrt(1-mu**2)
        sqrt_w  = np.sqrt(1-nu[2]**2)
        
        #update angels
        if sqrt_w != 0:
            prev_nu = nu.copy()            
            nu[0] = prev_nu[0]*mu + (prev_nu[0]*prev_nu[2]*cos_psi - prev_nu[1]*sin_psi)*sqrt_mu/sqrt_w
            nu[1] = prev_nu[1]*mu + (prev_nu[1]*prev_nu[2]*cos_psi + prev_nu[0]*sin_psi)*sqrt_mu/sqrt_w
            nu[2] = prev_nu[2]*mu - cos_psi*sqrt_mu*sqrt_w
        elif nu[2]==1.0:
            nu[0] = sqrt_mu*cos_psi
            nu[1] = sqrt_mu*sin_psi
            nu[2] = mu
        else: # nu[2]==-1.0
            nu[0] = sqrt_mu*cos_psi
            nu[1] = -sqrt_mu*sin_psi
            nu[2] = -mu

        #should we relaunch the photon?
        if n >= max_N:
            ret[0] = 2
            break
        if np.linalg.norm(r) > max_distance_from_det:  # assumes detector at origin
            ret[0] = 3
            break
    ret = ret[ret_cols]
    return ret





def simSource(source = {'r': np.array([0.0, 0.0, 0.0]),
                       'mu': np.array([0.0, 0.0, 1.0]),
                       'method': 'pencil', 
                       'time_profile': 'delta'}):
        
    if source['method'] == 'pencil':
        r0 = source['r']
        mu0 = source['mu']
        if source['time_profile'] == 'delta':
            t0 = 0.0
            n0 = 0
            
    return r0, nu0, d0, n0







def initPhotonAndRun(muS  =  1.0, g = 0.85,
               source = {'r': np.array([0.0, 0.0, 0.0]),
                          'mu': np.array([0.0, 0.0, 1.0]),
                          'method': 'pencil', 'time_profile': 'delta'},
               detR = 1.0,
               max_N = 1e5,
               max_distance_from_det = 1000.0, 
               ret_cols = [0,1,2,3,4,5,6,7,8]):
    
    r0, nu0, d0, n0 = simSource(source)
    
    return propPhoton(muS  =  muS, g = g,
               r0  = r0,
               nu0 = nu0,
               d0 = d0, n0 = n0,
               detR = detR,
               max_N = max_N,
               max_distance_from_det = max_distance_from_det,
               ret_cols = ret_cols)






def lunchPacketwithBatch(batchSize = 1000,
                        nPhotonsRequested = 1e6,
                        nPhotonsToRun = 1e10,
                        muS = 1.0, g = 0.85,
                        source = {'r': np.array([0.0, 0.0, 0.0]),
                                  'mu': np.array([0.0, 0.0, 1.0]),
                                  'method': 'pencil', 'time_profile': 'delta'},
                        detector = {'radius': 10.0},
                        control_param = {'max_N': 1e5,
                                         'max_distance_from_det': 1000},
                        normalize_d = None,
                        ret_cols = [1,2,7,8]
                        ):
    muS = float(muS)
    g = float(g)
    detR = float(detector['radius'])
    max_N = control_param['max_N']
    max_distance_from_det = float(control_param['max_distance_from_det'])
    
    nPhotonsRequested = int(nPhotonsRequested)
    batchSize = int(batchSize)
    data = np.ndarray(shape=(nPhotonsRequested, len(ret_cols)), dtype=float)
    
    num_detected = 0
    num_simulated = 0
    
    while num_simulated < nPhotonsToRun and num_detected < nPhotonsRequested:
        print('{:.0e}'.format(num_simulated), end="\r")
        ret = lunchBatch(batchSize = batchSize,
                         muS = muS, g = g,
                       source = source,
                       detR = detR,
                       max_N = max_N,
                       max_distance_from_det = max_distance_from_det, ret_cols=ret_cols)
        
        # Not valid photons return with n=-1 - so remove them
        ret = ret[ret[:, -1]>=0, :]
        if ret.shape[0] > nPhotonsRequested - num_detected:
            ret = ret[:nPhotonsRequested - num_detected, :]
        data[num_detected : (num_detected+ret.shape[0]), :] = ret
        
        num_detected += ret.shape[0]
        num_simulated += batchSize
            
    data = data[:num_detected, :] 
    if normalize_d is not None:
        data[:, -2] *= normalize_d
    return data, num_simulated, num_detected




def lunchPacket(nPhotonsRequested = 1e6,
                nPhotonsToRun = 1e10,
                muS = 1.0, g = 0.85,
                source = {'r': np.array([0.0, 0.0, 0.0]),
                          'mu': np.array([0.0, 0.0, 1.0]),
                          'method': 'pencil', 'time_profile': 'delta'},
                detector = {'radius': 10.0},
                control_param = {'max_N': 1e5,
                                 'max_distance_from_det': 1000},
                normalize_d = None,
                ret_cols = [1,2,7,8]
                ):
    muS = float(muS)
    g = float(g)
    detR = float(detector['radius'])
    max_N = control_param['max_N']
    max_distance_from_det = float(control_param['max_distance_from_det'])
    
     
    nPhotonsRequested = int(nPhotonsRequested)
    data = np.ndarray(shape=(nPhotonsRequested, len(ret_cols)), dtype=float)
    
    num_detected = 0
    num_simulated = 0
    
    while num_simulated < nPhotonsToRun and num_detected < nPhotonsRequested:
        ret = initPhotonAndRun(muS = muS, g = g,
               source = source,
               detR = detR,
               max_N = max_N,
               max_distance_from_det = max_distance_from_det)
        if ret[-1] >= 0:  #success
            data[num_detected, :] = ret
            num_detected += 1
        num_simulated+=1
            
    data = data[:num_detected, :] 
    if normalize_d is not None:
        data[:, 3] *= normalize_d
    return data, num_simulated, num_detected



