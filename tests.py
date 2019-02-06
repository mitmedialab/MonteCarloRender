#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 10:00:00 2019

@author: guysatat
"""

import numpy as np
import time
import pickle

import MC

def testBatch():
    t1 = time.perf_counter()
    ret = MC.lunchPacketwithBatch(batchSize = 1e7,
                               nPhotonsRequested = 1e7,
                                nPhotonsToRun = 1e7,
                                muS = 1.0, g = 0.85,
                                source = {'r': np.array([0.0, 0.0, 30.0]),
                                          'mu': np.array([0.0, 0.0, -1.0]),
                                          'method': 'pencil', 'time_profile': 'delta'},
                                detector = {'radius': 100.0},
                                control_param = {'max_N': 1e5,
                                                 'max_distance_from_det': 110},
                                normalize_d = None,
                                ret_cols = [0,1,2,3]
                                )
    print(ret[1], ret[2], time.perf_counter()-t1)
    ds = {'data': ret[0],
          'num_simulated': ret[1],
          'num_detected': ret[2]}
    with open('testdata.pickle', 'wb') as handle:
        pickle.dump(ds, handle)   
    np.save('testdata.npy', ret[0])
    return ret



def testPacket():
    ret = MC.lunchPacket(nPhotonsRequested = 3,
                   nPhotonsToRun = 100,
                   muS = 1.0, g = 0.85,
                   source = {'r': np.array([0.0, 0.0, 10.0]),
                             'mu': np.array([0.0, 0.0, -1.0]),
                             'method': 'pencil', 'time_profile': 'delta'},
                   detector = {'radius': 10.0},
                   control_param = {'max_N': 1e5,
                                    'max_distance_from_det': 1000},
                   normalize_d = None,
                   ret_cols = [0,1,2,3]
                   )
    print(ret)
    return ret

def testPhoton():
    ret = MC.propPhoton(muS  =  1.0, g = 0.85,
                  r0  = np.array([0.0, 0.0, 10.0]),
                  nu0 = np.array([0.0, 0.0, -1.0]),
                  d0 = 0.0, n0 = 0,
                  detR = 10.0,
                  max_N = 1e5,
                  max_distance_from_det = 1000.0,
                   ret_cols = [0,1,2,3]
                    )
    print(ret)
    return ret
   


