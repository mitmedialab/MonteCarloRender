import numpy as np

def photonsTo3DMeasurement(photons_xyd = np.array([[0,0,0],[0,0,0]]),
                              camera = {'center': [0,0],'dim':[64,64],
                                        'pixel_size':1, 'time_res':50,
                                        'time_bin':100} ):
    camera_center = camera['center']
    camera_dim = camera['dim']
    camera_pixelSize = camera['pixel_size']
    camera_timeRes = camera['time_res']
    camera_timeBin = camera['time_bin']
    
    camera_distanceBin = camera_timeRes*1e-12*3e8*1e3
    
    camera_X_bound = [camera_center[0]-camera_pixelSize*camera_dim[0]/2,camera_center[0]+camera_pixelSize*camera_dim[0]/2]
    camera_Y_bound = [camera_center[1]-camera_pixelSize*camera_dim[1]/2,camera_center[1]+camera_pixelSize*camera_dim[1]/2]
    camera_D_bound = [0,camera_timeRes*1e-12*camera_timeBin*3e8*1e3] # 1e-12:ps, 3e8:speed of light (m/s), 1e3:mm
    measurement,_ = np.histogramdd(photons_xyd, bins = (camera_dim[0],camera_dim[1],camera_timeBin),
                                 range=((camera_X_bound[0],camera_X_bound[1]),(camera_Y_bound[0],camera_Y_bound[1]),
                                        (camera_D_bound[0],camera_D_bound[1]))
                                        )
    return measurement