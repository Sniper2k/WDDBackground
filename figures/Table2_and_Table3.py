import numpy as np
import sys
sys.path.insert(1, '..')
import time
from scipy.ndimage import zoom
from PIL import Image

import forward as forward
import utility_2D as util

import adp as adp
import helper
import wigner_2d as wdd


## for Table 2: phase_object = 0
## for Table 3: phase_object = 1  

phase_object = 1 # 0

if phase_object == 1:
    backgrount_type = 'phase'
else:
    backgrount_type = 'general'

### Load cameraman and transfrom it ###

im_cam = Image.open("cameraman.tif")
im_cam = np.array(im_cam)

outd = 64 #96 
factor = outd*1.0/im_cam.shape[0]
im = np.zeros((outd,outd,3))
im[:,:,0] = zoom(im_cam[:,:],factor)
im[:,:,1] = zoom(im_cam[:,:],factor)
im[:,:,2] = zoom(im_cam[:,:],factor)

if phase_object == 1:
    obj = helper.image_to_phase_object(im,lambda x,v: x)
else:
    obj = helper.image_to_object(im,lambda x,v: x)

### Parameter Setup ###

d = im.shape[0]
delta = 8 #16
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

np.random.seed(1); r1 = np.random.rand(delta,delta)
np.random.seed(5); r2 = np.random.rand(delta,delta)
if (np.mod(d,2) == 0):
    window = window *np.exp(2j*r1) + 0.25 * r2
else:
    window = window *np.exp(2j*r1) 

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

### Background ###

if d == 64:
    phantom =  Image.open("phantom_64.tif")
if d == 96:
    phantom =  Image.open("phantom_96.tif")
phantom = np.array(phantom)

background = np.zeros(b.shape,dtype = 'float64')
for r in range(b.shape[2]):
            background[:,:,r] = phantom

noise_level = 3.5
scaling = (noise_level * np.linalg.norm(b) /  np.linalg.norm(background) )  

b_n = b +  scaling*background 

print('Noise level: ', util.relative_measurement_error(b,b_n))


### Proposed method ###

wignerb = wdd.wdd(b_n,
                  ptycho = par,
                  gamma = delta,
                  reg_type = 'value',
                  reg_threshold = 0.0,
                  mg_type = 'diag',
                  mg_diagonals_type = 'percent',
                  mg_diagonals_param = .0,
                  as_wtype = 'unweighted',
                  as_threshold = 0.0,                                 
                  add_dummy = False,
                  subspace_completion = False,
                  sbc_threshold = 0.0,
                  memory_saving = False,
                  background = backgrount_type)


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


wignerb = wdd.wdd(b_n,
                  ptycho = par,
                  gamma = 3,
                  reg_type = 'value',
                  reg_threshold = 0.0,
                  mg_type = 'diag',
                  mg_diagonals_type = 'percent',
                  mg_diagonals_param = .0,
                  as_wtype = 'unweighted',
                  as_threshold = 0.0,                             
                  add_dummy = False,
                  subspace_completion = False,
                  sbc_threshold = 0.0,
                  memory_saving = False,
                  background = backgrount_type)

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


### ADP ###
locations_2d = forward.loc_grid_circ((d,d),(shift,shift), False)
positions = locations_2d

J = d**2
alpha1 = 0 
alpha2 = 0.5
r = 10**(-6)
K = 10 
J_0 = 5

obj_wdd = np.ones([d,d] , dtype = 'complex')
background_init = 1/J * np.dstack([np.zeros([d,d] , dtype = 'complex')] * J)

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

