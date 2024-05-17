import numpy as np
import sys
sys.path.insert(1, '..')
import time
import sigpy
import cmath
from scipy.ndimage import zoom
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
colors = ["black", "lightgray", "black"]
cmap = LinearSegmentedColormap.from_list("", colors)
from PIL import Image

import utility_2D as util
import forward as forward

import wigner_with_background_removal as wdd_background
import wigner_2D as wdd

import adp as adp

### Helper functions ###

def image_to_object(im,satur_parser):
    im_hsv = rgb2hsv(im)
       
    modulus = im_hsv[:,:,2] * 255 
    phase = (modulus - np.min(modulus)) / (np.max(modulus) - np.min(modulus)) * 2*np.pi - np.pi
    
    obj_mod = modulus * np.exp(1.0j * phase)
    
    return obj_mod
    
### Load cameraman and transfrom it ###

im_cam = Image.open("cameraman.tif")
im_cam = np.array(im_cam)

outd = 64 
factor = outd*1.0/im_cam.shape[0]
im = np.zeros((outd,outd,3))
im[:,:,0] = zoom(im_cam[:,:],factor)
im[:,:,1] = zoom(im_cam[:,:],factor)
im[:,:,2] = zoom(im_cam[:,:],factor)

obj = image_to_object(im,lambda x,v: x)

### Parameter Setup ###

d = im.shape[0]
delta = 8 
shift = 1
f_dim = (d,d)
dsize = f_dim

### Construct Gaussian window ###

cov_mat = np.eye(2,dtype = complex)/0.05
mu = np.array([0.5 + delta*0.5, 0.5 + delta*0.5])
gauss = lambda x: np.exp(-0.5* (( x - mu).conj().T).dot((cov_mat/delta**2).dot(x - mu)))
window = np.zeros((delta,delta), dtype = complex);
for ix in range(delta):
    for iy in range(delta):
        window[ix,iy] = gauss( np.array([ix+1, iy+1]))

### Randomize it to avoid symmetry singularities ###

np.random.seed(3); r1 = np.random.rand(delta,delta)
np.random.seed(5); r2 = np.random.rand(delta,delta)
if (np.mod(d,2) == 0):
    window = window * np.exp(2j*r1) + 0.25 * r2
else:
    window = window * np.exp(2j*r1) 

### Prepare ptycho object and generate the forward measurements ###   

par = forward.ptycho(
            object_shape = obj.shape,
            window = window, 
            circular = True,
            loc_type = 'grid',
            shift = shift, 
            fourier_dimension = f_dim,
            float_shift = False)
    
print('Computing forward model')
f = par.forward_2D_pty(obj)
    
print('Computing measurements')
b = par.forward_to_meas_2D_pty(f)
    

### Figure 5 ###

trials = 20

noiselevel = np.zeros(trials)
subtraction_vanillaWDD_relerr = np.zeros(trials)
proposedWDD_relerr = np.zeros(trials)

for snr in range(trials):
    
    ### NOISE ###
        
    ## Background ##
    
    phantom =  Image.open("phantom.tif")
    phantom = np.array(phantom)    
    phantom = np.maximum(phantom, np.zeros_like(phantom))
    
    background = np.zeros(b.shape,dtype = 'float64')
    for r in range(b.shape[2]):
                background[:,:,r] = phantom
        
    factor = 5 * 10**(-6)  * (snr+4)
    scaling = 10 / factor**2 
    
    b_n = b +  scaling*background 
    
    ## Poisson noise ##
    scaling_p = factor**2 
    b_scaled =  scaling_p * b_n
    np.random.seed(2); b_n = np.random.poisson(b_scaled, b_scaled.shape) / scaling_p
    
    print('Noise level: ', util.relative_measurement_error(b,b_n))
   
    noiselevel[snr] = util.relative_measurement_error(b,b_n)
    
    reg_thresh = 0.6 / (0.1 * snr**2 + 1)
    
    ### Vanilla WDD after background subtraction ###
        
    start_time = time.time()
    
    # measure one extra (empty) diffraction pattern
    no_obj = np.ones_like(obj)
    f_dark = par.forward_2D_pty(no_obj)
    b_dark = par.forward_to_meas_2D_pty(f_dark)
    b_dark_n = b_dark + scaling * background 
    b_scaled =  scaling_p * b_dark_n 
    np.random.seed(5); b_dark_n = np.random.poisson(b_scaled, b_scaled.shape) / scaling_p

    # compute forward model of window (no object)
    no_obj = np.ones_like(obj) 
    f_dark = par.forward_2D_pty(no_obj)
    b_dark_forwardmodel = par.forward_to_meas_2D_pty(f_dark)

    # reconstruct background: subtract forward model from measured diffraction patterns
    background_reco = b_dark_n - b_dark_forwardmodel
    background_reco = background_reco[:,:,0]
    background_reco = np.dstack([background_reco] * np.shape(b_n)[2])
    
    # subtract computed background from measured diffraction patterns    
    b_n_denoised = b_n - background_reco
    print('Noise level: ', util.relative_measurement_error(b,b_n_denoised))

    wigner = wdd.wdd(b_n_denoised,
                      ptycho = par,
                      gamma = delta,
                      reg_type = 'percent',
                      reg_threshold = reg_thresh,
                      mg_type = 'diag',
                      mg_diagonals_type = 'percent',
                      mg_diagonals_param = 1.,
                      as_wtype = 'unweighted',
                      as_threshold = 0.0,                         
                      add_dummy = False,
                      subspace_completion = False,
                      sbc_threshold = 0.0,
                      memory_saving = False)
    
    print('Reconstructing...')
    
    obj_r = wigner.run()
    end_time = time.time()
        
    obj_r = util.align_objects(obj,obj_r,par.mask)
    
    print('Time: ', end_time - start_time)
    f_r = par.forward_2D_pty(obj_r)
    b_r = par.forward_to_meas_2D_pty(f_r)
    print('Reconstruction:')
    print( 'Relative error: ', util.relative_error(obj,obj_r,par.mask) )
    print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))
    
    subtraction_vanillaWDD_relerr[snr] = util.relative_error(obj,obj_r,par.mask) 
    
    ### Proposed method ###
    
    wignerb = wdd_background.wdd_background(b_n,
                      ptycho = par,
                      gamma = delta,
                      reg_type = 'percent',
                      reg_threshold = reg_thresh,
                      mg_type = 'diag',
                      mg_diagonals_type = 'percent',
                      mg_diagonals_param = 1.,
                      as_wtype = 'unweighted',
                      as_threshold = 0.0,                                
                      add_dummy = False,
                      subspace_completion = False,
                      sbc_threshold = 0.0,
                      memory_saving = False)
    
    print('Reconstructing...')
    
    start_time = time.time()
    obj_r = wignerb.run()
    end_time = time.time()
        
    obj_r = util.align_objects(obj,obj_r,par.mask)
    
    print('Time: ', end_time - start_time)
    
    f_r = par.forward_2D_pty(obj_r)
    b_r = par.forward_to_meas_2D_pty(f_r)
    # print('Mesurements from reconstructed object:')
    # # show_measurements(b_r,locations_2d,shift)
    # # print('Absolute pixelwise error:')
    # # show_measurements(np.abs(b-b_r),locations_2d,shift)
    
    print('Reconstruction:')
    print( 'Relative error: ', util.relative_error(obj,obj_r,par.mask) )
    print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))
 
    proposedWDD_relerr[snr] = util.relative_error(obj,obj_r,par.mask)

    obj_r_init = obj_r


plt.figure(figsize=(20,10))
plt.plot(noiselevel,np.log10(subtraction_vanillaWDD_relerr), linewidth= 8.0, color="#005293",  linestyle='dashed')
plt.plot(noiselevel,np.log10(proposedWDD_relerr), linewidth= 8.0, color="#a2ad00") 
plt.tick_params(axis='both', labelsize=42)
plt.legend(['vanilla WDD after background subtraction','proposed method'], fontsize = 42, loc = 4)
plt.ylabel('$log_{10}$ (rel. rec. err.)', fontsize = 42)
plt.xlabel('noise level', fontsize = 42)
plt.show()
    


### Table 4 ###

### NOISE ###

factor = 2 * 10**(-5)

## Background ##
phantom =  Image.open("phantom.tif")
phantom = np.array(phantom)
phantom = np.maximum(phantom, np.zeros_like(phantom))

background = np.zeros(b.shape,dtype = 'float64')
for r in range(b.shape[2]):
            background[:,:,r] = phantom

scaling = 10 / factor **2

b_n = b +  scaling*background 

## Poisson noise ##
scaling_p = factor**2 
b_scaled =  scaling_p * b_n
np.random.seed(2); b_n = np.random.poisson(b_scaled, b_scaled.shape) / scaling_p

print('Noise level: ', util.relative_measurement_error(b,b_n))

### Vanilla WDD ####

wigner = wdd.wdd(b_n,
                  ptycho = par,
                  gamma = delta,
                  reg_type = 'percent',
                  reg_threshold = 0.6,
                  mg_type = 'diag',
                  mg_diagonals_type = 'percent',
                  mg_diagonals_param = 1.,
                  as_wtype = 'unweighted',
                  as_threshold = 0.0,                                  
                  add_dummy = False,
                  subspace_completion = False,
                  sbc_threshold = 0.0,
                  memory_saving = False)

print('Reconstructing...')

start_time = time.time()
obj_r = wigner.run()
end_time = time.time()

obj_r = util.align_objects(obj,obj_r,par.mask)

print('Time: ', end_time - start_time)
f_r = par.forward_2D_pty(obj_r)
b_r = par.forward_to_meas_2D_pty(f_r)
print('Reconstruction:')
print( 'Relative error: ', util.relative_error(obj,obj_r,par.mask) )
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))

obj_init_vanilla = obj_r

### Proposed method ###

wignerb = wdd_background.wdd_background(b_n,
                  ptycho = par,
                  gamma = delta,
                  reg_type = 'percent',
                  reg_threshold = 0.6,
                  mg_type = 'diag',
                  mg_diagonals_type = 'percent',
                  mg_diagonals_param = 1.,
                  as_wtype = 'unweighted',
                  as_threshold = 0.0,                          
                  add_dummy = False,
                  subspace_completion = False,
                  sbc_threshold = 0.0,
                  memory_saving = False)

print('Reconstructing...')

start_time = time.time()
obj_r = wignerb.run()
end_time = time.time()

obj_r = util.align_objects(obj,obj_r,par.mask)

print('Time: ', end_time - start_time)
f_r = par.forward_2D_pty(obj_r)
b_r = par.forward_to_meas_2D_pty(f_r)
print('Reconstruction:')
print( 'Relative error: ', util.relative_error(obj,obj_r,par.mask) )
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))

obj_r_init = obj_r    

### ADP ###

locations_2d = forward.loc_grid_circ((d,d),(shift,shift), False)
positions = locations_2d

J = d**2
alpha1 = 0 
alpha2 = 10**3
r = 10**(-1)
K = 50
J_0 = 5

background_init = np.dstack([np.zeros([d,d] , dtype = 'complex')] * J)
obj_wdd = np.ones([d,d] , dtype = 'complex')
start_time = time.time()
obj_r_adp, background_r, rel_err_adp_1_init, meas_err_adp_1_init = adp.ADP(window,b_n,d,delta,1,J,positions,K,alpha1,alpha2,r,J_0,obj_wdd,background_init,obj,b)
end_time = time.time()
print('Time: ', end_time - start_time)
obj_r_adp = obj_r_adp[0:d,0:d]
obj_r_adp = util.align_objects(obj,obj_r_adp,par.mask)
print( 'Relative error: ', util.relative_error(obj,obj_r_adp,par.mask) )
f_r = par.forward_2D_pty(obj_r_adp)
b_r = par.forward_to_meas_2D_pty(f_r)
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))  

obj_wdd = obj_init_vanilla 
start_time = time.time()
obj_r_adp, background_r, rel_err_adp_vanilla_init, meas_err_adp_vanilla_init = adp.ADP(window,b_n,d,delta,1,J,positions,K,alpha1,alpha2,r,J_0,obj_wdd,background_init,obj,b)
end_time = time.time()
print('Time: ', end_time - start_time)
obj_r_adp = obj_r_adp[0:d,0:d]
obj_r_adp = util.align_objects(obj,obj_r_adp,par.mask)
print( 'Relative error: ', util.relative_error(obj,obj_r_adp,par.mask) )
f_r = par.forward_2D_pty(obj_r_adp)
b_r = par.forward_to_meas_2D_pty(f_r)
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))    

obj_wdd = obj_r_init 
start_time = time.time()
obj_r_adp, background_r, rel_err_adp_our_init, meas_err_adp_our_init = adp.ADP(window,b_n,d,delta,1,J,positions,K,alpha1,alpha2,r,J_0,obj_wdd,background_init,obj,b)
end_time = time.time()
print('Time: ', end_time - start_time)
obj_r_adp = obj_r_adp[0:d,0:d]
obj_r_adp = util.align_objects(obj,obj_r_adp,par.mask)
print( 'Relative error: ', util.relative_error(obj,obj_r_adp,par.mask) )
f_r = par.forward_2D_pty(obj_r_adp)
b_r = par.forward_to_meas_2D_pty(f_r)
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))    



   
    
   

