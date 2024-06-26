import numpy as np
import sys
sys.path.insert(1, '..')
import time
from scipy.ndimage import zoom
from PIL import Image


import forward as forward
import utility_2D as util
import helper

import wigner_2D as wdd

import adp as adp

### Load cameraman and transform it ###

im_cam = Image.open("cameraman.tif")
im_cam = np.array(im_cam)

outd = 128
factor_boat = outd*1.0/im_boat.shape[0]
im = np.zeros((outd,outd,3))
im[:,:,0] = zoom(im_boat[:,:],factor_boat)
im[:,:,1] = zoom(im_boat[:,:],factor_boat)
im[:,:,2] = zoom(im_boat[:,:],factor_boat)

factor = outd*1.0/im_cam.shape[0]
im_phase = np.zeros((outd,outd,3))
im_phase[:,:,0] = zoom(im_cam[:,:],factor)
im_phase[:,:,1] = zoom(im_cam[:,:],factor)
im_phase[:,:,2] = zoom(im_cam[:,:],factor)

obj = helper.image_to_object(im,im_phase,lambda x,v: x)

helper.show_object(obj)

### Parameter Setup ###

d = im.shape[0]
delta = 24
shift = 2 # 4 # 8
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

### Vanilla WDD for background-free measurements ###

wigner = wdd.wdd(b,
                  ptycho = par,
                  gamma = delta//shift,
                  reg_type = 'percent',
                  reg_threshold = 0.1,
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
helper.show_object(obj_r)

print('Time: ', end_time - start_time)
f_r = par.forward_2D_pty(obj_r)
b_r = par.forward_to_meas_2D_pty(f_r)
print('Reconstruction:')
print( 'Relative error: ', util.relative_error(obj,obj_r,par.mask) )
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))

### Background ###

phantom =  Image.open("phantom.tif")
phantom = np.array(phantom)

background = np.zeros(b.shape,dtype = 'float64')
for r in range(b.shape[2]):
            background[:,:,r] = phantom

scaling = 25 * 10**8
b_n = b +  scaling*background 

print('Noise level: ', util.relative_measurement_error(b,b_n))

#### Proposed method ###

wignerb = wdd.wdd(b_n,
                  ptycho = par,
                  gamma = delta//shift,
                  reg_type = 'percent',
                  reg_threshold = 0.1,
                  mg_type = 'diag',
                  mg_diagonals_type = 'percent',
                  mg_diagonals_param = 1.0,
                  as_wtype = 'unweighted',
                  as_threshold = 0.0,                        
                  add_dummy = False,
                  subspace_completion = False,
                  sbc_threshold = 0.0,
                  memory_saving = False,
                  background = 'general')

print('Reconstructing...')

start_time = time.time()
obj_r = wignerb.run()
end_time = time.time()

obj_r = util.align_objects(obj,obj_r,par.mask)
helper.show_object(obj_r)

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

J = d**2 // shift**2
r = 10**(-1)
alpha1 = 0 
if shift == 2:
    alpha2 = 10**3
    K = 20
elif shift == 4:
    alpha2 = 10**(3.5)
    K = 50 
elif shift == 8:
    alpha2 = 10**4.5
    K = 250 
    r = 10**(-2)
J_0 = 2

background_init = np.dstack([np.zeros([d,d] , dtype = 'complex')] * J)

obj_wdd = obj_r_init 
start_time = time.time()
obj_r_adp, background_r, rel_err_adp_our_init, meas_err_adp_our_init = adp.ADP(window,b_n,d,delta,shift,J,positions,K,alpha1,alpha2,r,J_0,obj_wdd,background_init,obj,b)
end_time = time.time()
print('Time: ', end_time - start_time)
obj_r_adp = obj_r_adp[0:d,0:d]
obj_r_adp = util.align_objects(obj,obj_r_adp,par.mask)
helper.show_object(obj_r_adp)
print( 'Relative error: ', util.relative_error(obj,obj_r_adp,par.mask) )
f_r = par.forward_2D_pty(obj_r_adp)
b_r = par.forward_to_meas_2D_pty(f_r)
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))
  
obj_wdd = np.ones([d,d] , dtype = 'complex')
start_time = time.time()
obj_r_adp, background_r, rel_err_adp_1_init, meas_err_adp_1_init = adp.ADP(window,b_n,d,delta,shift,J,positions,K,alpha1,alpha2,r,J_0,obj_wdd,background_init,obj,b)
end_time = time.time()
print('Time: ', end_time - start_time)
obj_r_adp = obj_r_adp[0:d,0:d]
obj_r_adp = util.align_objects(obj,obj_r_adp,par.mask)
helper.show_object(obj_r_adp)
print( 'Relative error: ', util.relative_error(obj,obj_r_adp,par.mask) )
f_r = par.forward_2D_pty(obj_r_adp)
b_r = par.forward_to_meas_2D_pty(f_r)
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))

