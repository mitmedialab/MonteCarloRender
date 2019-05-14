# MonteCarloRender

This is a GPU implementation of a Monte Carlo Photon Transport in scattering media


## Main function call
  Use the lunchPacketwithBatch() function to run the simulator. See MC.py for documentation.

## Tests
  The tests.ipynb file provide some basic tests for the code
  
  
## Several notes on properties / limitations:
### Media:
 Assume homogeneous media. You can provide the scattering mean free path (MFP) and Henyey Greenstein anisotropy coefficient.

### Detector:
 Assume a single detector.
 Detector is circular (you can provide diameter).
 Detector is centered at the origin.
 Detector is parallel to the x-y plane and points at the positive z direction.

### Targets:
 Supports a 2d target that is positioned at a given x-y plane (user provides the z coordinate).
 The mask can be absorbing (binary value), or scattering (reflective).
 
### GPU support:
  Supports single or multiple GPUs (user defined).


