"""
Based on 'Background noise removal in x-ray ptychography' (Wang et al., 2017)
"""

import numpy as np
import forward as forward


def PreProcessing(b, obj, window):
    
    d = obj.shape[0]
    
    par = forward.ptycho(
            object_shape = obj.shape,
            window = window, 
            circular = True,
            loc_type = 'grid',
            shift = 1, 
            fourier_dimension = (d,d),
            float_shift = False)
    
    
    zero_obj = np.ones_like(obj)
    
    f_dark = par.forward_2D_pty(zero_obj)
    b_dark = par.forward_to_meas_2D_pty(f_dark)
    
    r = d // 2
    r2 = 10
    
    b_dark = b_dark[:,:,0]    
    

    b_dark_long = np.dstack([b_dark[r - r2:r + r2,r - r2:r + r2]] * b.shape[2])
    alpha = np.sum(np.sum(b[r - r2:r + r2,r - r2:r + r2] * b_dark_long,0) , 0) / (np.sum(b_dark[r - r2:r + r2,r - r2:r + r2] ** 2))
    b_processed = b - np.transpose(np.transpose(np.ones_like(b) * alpha) * b_dark)
    
    threshold = np.max(np.max(b_processed[r - r2:r + r2,r - r2:r + r2], 0),0) - 2 * np.std( np.std(b_processed[r - r2:r + r2,r - r2:r + r2], 0),0)
    threshold = (np.ones_like(b) * threshold)
    b_processed[b_processed >= threshold] = threshold[b_processed >= threshold]
    
    
    
    return b_processed
   
    