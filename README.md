# MonteCarloRender

This is a GPU implementation of a Monte Carlo Photon Transport in scattering media

## Several notes on prperties / limitations:
# Media:
 Assume homogenouse media. You can provide the scattering mean free path (MFP) and Henyey Greenstein unisotropy coefficient. 
 
# Detector:
 Assume a single detector.
 Detector is circular (you can provide diameter).
 Detector is centered at the origin.
 Detector is parallel to the x-y plane and points at the positive z direction.
 
# Targets:
 Supports a 2d target that is positioned at a given x-y plane (user provides the z coordinate).
 The mask can be absorbing (binary value), or scattering (reflective).
 
 # GPU support:
  Supports single or multiple GPUs (user defined).
  
