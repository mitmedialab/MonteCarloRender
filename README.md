# MonteCarloRender

This is a GPU implementation of a Monte Carlo Photon Transport in scattering media


## Main function call
  Use the lunchPacketwithBatch() function to run the simulator. See MC.py for documentation.
  
## Several notes on properties / limitations:
### Media:
 Assume homogeneous media. You can provide the scattering mean free path (MFP) and Henyey Greenstein anisotropy coefficient.

### Detector:
 Assume a single detector. <br/>
 Detector is circular (you can provide diameter). <br/>
 Detector is centered at the origin. <br/>
 Detector is parallel to the x-y plane and points at the positive z direction.

### Targets:
 Supports a 2d target that is positioned at a given x-y plane (user provides the z coordinate). <br/>
 The mask can be absorbing (binary value), or scattering (reflective).
 
### GPU support:
  Supports single or multiple GPUs (user defined).

## Example:
    import MC
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
                                ret_cols = [0,1,2,3,4,5,6,7]
                                )      
                                
  The runtime is approximatley 15 sec on a single Nvidia GTX 1080. <br/>
  ret[0] is the data. <br/>
  ret[1][0] are the number of simulated photons.<br/>
  ret[1][1] are the number of detected photons.<br/>
  etc.
