#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:00:00 2019

@author: guysatat
"""

import numpy as np
from multiprocessing import Pool
from itertools import repeat
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import sys
import math

#Assumptions:
# 1. Detector is circular, at 0,0,0 on the x-y plane and points towards the positive z axis
# 2. Doesn't dupport reflective targets.
# 3. Support only impulse pencil beam.

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
                        ret_cols = [0,1,2,3],
                        target = {'type':0,
                                  'mask':np.zeros(shape=(60,60)),
                                  'grid_size':np.array([1,1]),
                                  'z_target':20},
                        start_from_media_z_boundary = False
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
        ret = lunchBatchGPU(batchSize = batchSize,
                         muS = muS, g = g,
                       source = source,
                       detR = detR,
                       max_N = max_N,
                       max_distance_from_det = max_distance_from_det, ret_cols=ret_cols, target=target,
                       start_from_media_z_boundary=start_from_media_z_boundary)
        # Not valid photons return with n=-1 - so remove them
        ret = ret[ret[:, 0]>=0, :]
        if ret.shape[0] > nPhotonsRequested - num_detected:
            ret = ret[:nPhotonsRequested - num_detected, :]
        data[num_detected : (num_detected+ret.shape[0]), :] = ret
        num_detected += ret.shape[0]
        num_simulated += batchSize
            
    data = data[:num_detected, :] 
    if normalize_d is not None:
        data[:, 1] *= normalize_d
    return data, num_simulated, num_detected


def lunchBatchGPU(batchSize = 1000,
               muS  =  1.0, g = 0.85,
               source = {'r': np.array([0.0, 0.0, 0.0]),
                          'mu': np.array([0.0, 0.0, 1.0]),
                          'method': 'pencil', 'time_profile': 'delta'},
               detR = 1.0,
               max_N = 1e5,
               max_distance_from_det = 1000.0,
               ret_cols = [0,1,2,3],
               target = {'type':0,
                         'mask':np.zeros(shape=(60,60)),
                         'grid_size':np.array([1,1]),
                         'z_target':20},
               start_from_media_z_boundary = False
            ):
    
    
    muS = float(muS)
    g = float(g)
    
    source_type, source_param1, source_param2 = simSource(source = source)
        
    target_type = target['type']
    target_mask = target['mask']
    target_gridsize = target['grid_size'].astype(float)
    z_target = target['z_target']

    
    threads_per_block = 256 
    blocks = 64
    photons_per_thread = int(np.ceil(float(batchSize)/(threads_per_block * blocks)))
    device_id = 0

    device = cuda.select_device(device_id)
    data_out = cuda.device_array((threads_per_block*blocks, photons_per_thread, 11), dtype=np.float32)
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=np.random.randint(sys.maxsize))
    
    propPhotonGPU[blocks, threads_per_block](rng_states, data_out, photons_per_thread, muS, g, source_type, source_param1, source_param2, detR, max_N, max_distance_from_det,target_type,target_mask,target_gridsize,z_target,start_from_media_z_boundary)
        
    data = data_out.copy_to_host()
    data = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
    data = data[:, ret_cols]

    return data



#Return columns:
#    0  1  2  3  4    5    6      7   8, 9, 10
#    n, d, x ,y, z, mu_x, mu_y, mu_z, reserved 
@cuda.jit
def propPhotonGPU(rng_states, data_out, photons_per_thread, muS, g, source_type, source_param1, source_param2, detR, max_N, max_distance_from_det, target_type,target_mask,target_gridsize,z_target,start_from_media_z_boundary):
    
    thread_id = cuda.grid(1)
    target_x_dim = target_mask.shape[1]
    target_y_dim = target_mask.shape[0]
    x_center_index = target_x_dim / 2
    y_center_index = target_y_dim / 2
    
    if source_type == 1:
        rand_theta = xoroshiro128p_uniform_float32(rng_states, thread_id)
        rand_psi = xoroshiro128p_uniform_float32(rng_states, thread_id)

    for photon_ind in range(photons_per_thread):
        # Initiate photons based on the illumination type
        #determine x,y,z
        if source_type == 0:
            x, y, z, z_start =  source_param1[0], source_param1[1], source_param1[2], source_param1[2]
            nux, nuy, nuz = source_param1[3], source_param1[4], source_param1[5]
            d, n = source_param1[6], source_param1[7]
        elif source_type == 1: 
            x, y, z, z_start =  source_param1[0], source_param1[1], source_param1[2], source_param1[2]
            theta =source_param1[6]*(rand_theta-0.5) #theta is angle change of optical axis
            psi = 2*math.pi*rand_psi
            mu = math.cos(theta)
            sqrt_mu = math.sin(theta)
            sin_psi = math.sin(psi)
            cos_psi = math.cos(psi)
            sqrt_w  = math.sqrt(1-source_param1[5]**2)
            if source_param1[5] == 1.0:
                nux = sqrt_mu*cos_psi
                nuy = sqrt_mu*sin_psi
                nuz = mu 
            elif source_param1[5] == -1.0:
                nux = sqrt_mu*cos_psi
                nuy = -sqrt_mu*sin_psi
                nuz = -mu
            elif sqrt_w != 0:
                nux = source_param1[3]*mu +(source_param1[3]*source_param1[5]*cos_psi - source_param1[4]*sin_psi)*sqrt_mu/sqrt_w
                nuy = source_param1[4]*mu +(source_param1[4]*source_param1[5]*cos_psi + source_param1[3]*sin_psi)*sqrt_mu/sqrt_w
                nuz = source_param1[5]*mu - cos_psi*sqrt_mu*sqrt_w
            d, n = source_param1[7],source_param1[8]

        detR2 = detR**2
        while True:
            rand1 = xoroshiro128p_uniform_float32(rng_states, thread_id) 
            rand2 = xoroshiro128p_uniform_float32(rng_states, thread_id) 
            rand3 = xoroshiro128p_uniform_float32(rng_states, thread_id)             
            
            #print(r, nu, d, n)
            # Calc random prop distance
            cd = - math.log(rand1) / muS

            # Update temporary new location
            t_rx = x + cd * nux
            t_ry = y + cd * nuy
            t_rz = z + cd * nuz

            #check if we hit the detector
            if t_rz <= 0: #did we pass the detector?
                cd = - z / nuz
    
                t_rx = x + cd * nux
                t_ry = y + cd * nuy
                t_rz = z + cd * nuz
            
                if t_rx**2 + t_ry**2 < detR2:
                    d+=cd
                    n+=1
                    x,y,z = t_rx, t_ry, t_rz
                    
                    data_out[thread_id, photon_ind, 0] = n
                    data_out[thread_id, photon_ind, 1] = d
                    data_out[thread_id, photon_ind, 2] = x
                    data_out[thread_id, photon_ind, 3] = y
                    data_out[thread_id, photon_ind, 4] = z
                    data_out[thread_id, photon_ind, 5] = nux
                    data_out[thread_id, photon_ind, 6] = nuy
                    data_out[thread_id, photon_ind, 7] = nuz
                    break
                else:  # if we passed the detector and didn't hit it we should stop
                    data_out[thread_id, photon_ind, :] = -1.0
                    break
            
            if target_type == 1: #if target is simulated
                if (t_rz - z_target) * (z - z_target) <=0 : #we passed the target plane. See if we hit target
                    cd_target = (z_target - z) / nuz
                    x = x + cd_target * nux #this x is temporally, if simulation resumed, updated by tr_x immidiately
                    y = y + cd_target * nuy
                    x_index = int(math.floor(x / target_gridsize[0]) + x_center_index) #center of camera at x=0
                    y_index = int(math.floor(y / target_gridsize[1]) + y_center_index) #canter of cameta at y=0
                    if x_index < 0 or x_index >= target_x_dim or y_index < 0 or y_index >= target_y_dim:
                        data_out[thread_id, photon_ind, :] = -4.0 #photon is out of the bound of target
                        break
                    elif target_mask[x_index,y_index] == 0:
                        data_out[thread_id, photon_ind, :] = -5.0 #we are absorbed by target
                        break
            
            #check if we are out of tissue (when starting from tissue z boundary)
            if start_from_media_z_boundary:
                if t_rz > z_start:
                    data_out[thread_id, photon_ind, :] = -6.0
                    break

            # Update photon
            x, y, z = t_rx, t_ry, t_rz 
            d += cd #prop distance
            n += 1 #increase scatter counter

            # Scatter to new angle
            psi = 2 * math.pi * rand2
            mu = 1/(2*g) * (1+g**2 - ( (1-g*g)/(1-g+2*g*rand3))**2)

            # Update angels
            sin_psi = math.sin(psi)
            cos_psi = math.cos(psi)
            sqrt_mu = math.sqrt(1-mu**2)
            sqrt_w  = math.sqrt(1-nuz**2)
            if sqrt_w != 0:
                prev_nux, prev_nuy, prev_nuz = nux, nuy, nuz
                nux = prev_nux*mu + (prev_nux*prev_nuz*cos_psi - prev_nuy*sin_psi)*sqrt_mu/sqrt_w
                nuy = prev_nuy*mu + (prev_nuy*prev_nuz*cos_psi + prev_nux*sin_psi)*sqrt_mu/sqrt_w
                nuz = prev_nuz*mu - cos_psi*sqrt_mu*sqrt_w
            elif nuz==1.0:
                nux = sqrt_mu*cos_psi
                nuy = sqrt_mu*sin_psi
                nuz = mu
            else: # nu[2]==-1.0
                nux = sqrt_mu*cos_psi
                nuy = -sqrt_mu*sin_psi
                nuz = -mu

            # Should we stop
            if n >= max_N:
                data_out[thread_id, photon_ind, :] = -2.0
                break
            if math.sqrt(x*x + y*y + z*z) > max_distance_from_det:  # Assumes detector at origin
                data_out[thread_id, photon_ind, :] = -3.0
                break


def simSource(source = {'r': np.array([0.0, 0.0, 0.0]),
                       'mu': np.array([0.0, 0.0, 1.0]),
                       'method': 'pencil', 
                       'time_profile': 'delta'}):
    r0 = source['r']
    nu0 = source['mu']
    if source['method'] == 'pencil':
        source_type = 0
        #[r[0:2], nu[0:2], d, n]
        source_param1 = np.array([r0[0], r0[1], r0[2], nu0[0], nu0[1], nu0[2], 0.0, 0]).astype(float)
        source_param2 = np.array([0]).astype(float)
    elif source['method'] == 'cone':
        source_type = 1
        theta = source['theta']
        #[r[0:2], nu[0:2],theta d, n]
        source_param1 = np.array([r0[0], r0[1], r0[2], nu0[0], nu0[1], nu0[2], theta, 0.0, 0]).astype(float)
        source_param2 = np.array([0]).astype(float)
    elif source['method'] == 'point': #point source is a special case of light cone
        source_type = 1
        theta = 2*math.pi
        source_param1 = np.array([r0[0], r0[1], r0[2], 0, 0, -1, theta, 0.0, 0]).astype(float)
        source_param2 = np.array([0]).astype(float)
    else:
        sys.exit("Source type is not supported")
            
    return source_type, source_param1, source_param2


























###
# CPU versions, deprecated
###

# def lunchPacket(nPhotonsRequested = 1e6,
#                 nPhotonsToRun = 1e10,
#                 muS = 1.0, g = 0.85,
#                 source = {'r': np.array([0.0, 0.0, 0.0]),
#                           'mu': np.array([0.0, 0.0, 1.0]),
#                           'method': 'pencil', 'time_profile': 'delta'},
#                 detector = {'radius': 10.0},
#                 control_param = {'max_N': 1e5,
#                                  'max_distance_from_det': 1000},
#                 normalize_d = None,
#                 ret_cols = [1,2,7,8]
#                 ):
#     muS = float(muS)
#     g = float(g)
#     detR = float(detector['radius'])
#     max_N = control_param['max_N']
#     max_distance_from_det = float(control_param['max_distance_from_det'])
    
     
#     nPhotonsRequested = int(nPhotonsRequested)
#     data = np.ndarray(shape=(nPhotonsRequested, len(ret_cols)), dtype=float)
    
#     num_detected = 0
#     num_simulated = 0
    
#     while num_simulated < nPhotonsToRun and num_detected < nPhotonsRequested:
#         ret = initPhotonAndRun(muS = muS, g = g,
#                source = source,
#                detR = detR,
#                max_N = max_N,
#                max_distance_from_det = max_distance_from_det)
#         if ret[-1] >= 0:  #success
#             data[num_detected, :] = ret
#             num_detected += 1
#         num_simulated+=1
            
#     data = data[:num_detected, :] 
#     if normalize_d is not None:
#         data[:, 3] *= normalize_d
#     return data, num_simulated, num_detected






# def lunchBatch(batchSize = 1000,
#                muS  =  1.0, g = 0.85,
#                source = {'r': np.array([0.0, 0.0, 0.0]),
#                           'mu': np.array([0.0, 0.0, 1.0]),
#                           'method': 'pencil', 'time_profile': 'delta'},
#                detR = 1.0,
#                max_N = 1e5,
#                max_distance_from_det = 1000.0,
#                ret_cols = [1,2,7,8]):
    
    
#     # Assume all photons start the same (impulse pencil)
#     r0 = source['r']
#     nu0 = source['mu']
#     d0 = 0.0
#     n0 = 0        
#     arg = (muS, g, r0, nu0, d0, n0, detR, max_N, max_distance_from_det, ret_cols)
#     with Pool() as tp:
#         data = tp.starmap(propPhoton, repeat(arg, batchSize))
#     data = np.array(data)
    
# #    THIS IS A NON_PARALLEL HISTORIC VERSION
# #     data = np.zeros(shape=(batchSize, len(ret_cols)), dtype=float)
# #     for i in range(batchSize):
# #        #r0, nu0, d0, n0 = simSource(source)
# #        r0 = source['r']
# #        nu0 = source['mu']
# #        d0 = 0.0
# #        n0 = 0        
# #        data[i, :] = propPhoton(muS = muS, g = g,
# #                       r0  = r0,
# #                       nu0 = nu0,
# #                       d0 = d0, n0 = n0,
# #                       detR = detR,
# #                       max_N = max_N,
# #                       max_distance_from_det = max_distance_from_det,
# #                       ret_cols=ret_cols )

    
#     return data





# def initPhotonAndRun(muS  =  1.0, g = 0.85,
#                source = {'r': np.array([0.0, 0.0, 0.0]),
#                           'mu': np.array([0.0, 0.0, 1.0]),
#                           'method': 'pencil', 'time_profile': 'delta'},
#                detR = 1.0,
#                max_N = 1e5,
#                max_distance_from_det = 1000.0, 
#                ret_cols = [0,1,2,3,4,5,6,7,8]):
    
#     r0, nu0, d0, n0 = simSource(source)
    
#     return propPhoton(muS  =  muS, g = g,
#                r0  = r0,
#                nu0 = nu0,
#                d0 = d0, n0 = n0,
#                detR = detR,
#                max_N = max_N,
#                max_distance_from_det = max_distance_from_det,
#                ret_cols = ret_cols)





# #Return columns:
# #    0    1  2  3   4     5     6    7  8
# # Status, x ,y, z, mu_x, mu_y, mu_z, d, n
# def propPhoton(muS  =  1.0, g = 0.85,
#                r0  = np.array([0.0,0.0,0.0]),
#                nu0 = np.array([0.0,0.0,1.0]),
#                d0 = 0.0, n0 = 0,
#                detR = 1.0,
#                max_N = 1e5,
#                max_distance_from_det = 1000.0,
#                ret_cols = [1,2,7,8]):
#     np.random.seed()
#     r =  r0.copy()
#     nu = nu0.copy()
#     d = d0
#     n = n0        
    
#     detR2 = detR**2
#     ret = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 , -1.0, -1.0])
#     while True:
#         #print(r, nu, d, n)
#         # Calc random prop params
#         cd = - np.log(np.random.random()) / muS
        
#         t_r = r + cd * nu
        
#         #check if we hit the detector
#         if t_r[2] <= 0: #did we pass the detector?
#             cd = - r[2] / nu[2]
#             t_r = r + cd * nu
#             if t_r[0]**2 + t_r[1]**2 < detR2:
#                 d+=cd
#                 n+=1
#                 r = t_r
#                 ret = np.array([0, r[0],r[1],r[2],nu[0],nu[1],nu[2],d,n])
#                 break
#             else:  # if we passed the detector and didn't hit it we should restart
#                 ret[0] = 1
#                 break
            
#         # prop photon
#         for i in range(3):
#             r[i] = t_r[i] 
#         d += cd #prop distance
#         n += 1 #increase scatter counter
        
#         #scatter to new angle
#         psi = 2*np.pi*np.random.random()
#         mu = 1/(2*g) * (1+g**2 - ( (1-g*g)/(1-g+2*g*np.random.random()))**2)
        
#         sin_psi = np.sin(psi)
#         cos_psi = np.cos(psi)
#         sqrt_mu = np.sqrt(1-mu**2)
#         sqrt_w  = np.sqrt(1-nu[2]**2)
        
#         #update angels
#         if sqrt_w != 0:
#             prev_nu = nu.copy()            
#             nu[0] = prev_nu[0]*mu + (prev_nu[0]*prev_nu[2]*cos_psi - prev_nu[1]*sin_psi)*sqrt_mu/sqrt_w
#             nu[1] = prev_nu[1]*mu + (prev_nu[1]*prev_nu[2]*cos_psi + prev_nu[0]*sin_psi)*sqrt_mu/sqrt_w
#             nu[2] = prev_nu[2]*mu - cos_psi*sqrt_mu*sqrt_w
#         elif nu[2]==1.0:
#             nu[0] = sqrt_mu*cos_psi
#             nu[1] = sqrt_mu*sin_psi
#             nu[2] = mu
#         else: # nu[2]==-1.0
#             nu[0] = sqrt_mu*cos_psi
#             nu[1] = -sqrt_mu*sin_psi
#             nu[2] = -mu

#         #should we relaunch the photon?
#         if n >= max_N:
#             ret[0] = 2
#             break
#         if np.linalg.norm(r) > max_distance_from_det:  # assumes detector at origin
#             ret[0] = 3
#             break
#     ret = ret[ret_cols]
#     return ret