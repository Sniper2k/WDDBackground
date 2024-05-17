import numpy as np
import sys
sys.path.insert(1, '..')
import time

import forward as forward
import wigner_2D as wdd
import preprocessing as preprocessing

import helper
import utility_2D as util
from scipy.ndimage import zoom

### Load cameraman and transfrom it ###

from PIL import Image
im_cam = Image.open("cameraman.tif")
im_cam = np.array(im_cam)

outd = 128
factor = outd*1.0/im_cam.shape[0]
im = np.zeros((outd,outd,3))
im[:,:,0] = zoom(im_cam[:,:],factor)
im[:,:,1] = zoom(im_cam[:,:],factor)
im[:,:,2] = zoom(im_cam[:,:],factor)

obj = helper.image_to_object(im,lambda x,v: x)

helper.show_object(obj)

### Parameter Setup ###

d = im.shape[0]
delta = 16
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
        

helper.show_object(window)
    
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

helper.show_diffpat(b[:,:,0])


### Background ###

phantom =  Image.open("phantom.tif")
phantom = np.array(phantom)

#sigpy.shepp_logan(dsize, dtype='float64')*255 

background = np.zeros(b.shape,dtype = 'float64')
for r in range(b.shape[2]):
            background[:,:,r] = phantom


scaling = 25 * 10**8 # lower noise
#scaling = 25 * 10**9 # higher noise


b_n = b +  scaling*background 

helper.show_diffpat(b_n[:,:,0])

print('Noise level: ', util.relative_measurement_error(b,b_n))


### Vanilla WDD ####

wigner = wdd.wdd(b_n,
                  ptycho = par,
                  gamma = delta,
                  reg_type = 'percent',
                  reg_threshold = 0.0,
                  mg_type = 'diag',
                  mg_diagonals_type = 'percent',
                  mg_diagonals_param = 1.0,
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


### Vanilla WDD after preprocessing ####

b_n_prep = preprocessing.PreProcessing(b_n,obj,window)


wigner = wdd.wdd(b_n_prep,
                  ptycho = par,
                  gamma = delta,
                  reg_type = 'percent',
                  reg_threshold = 0.0,
                  mg_type = 'diag',
                  mg_diagonals_type = 'percent',
                  mg_diagonals_param = 1.0,
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


### Proposed method ###


wignerb = wdd.wdd(b_n,
                  ptycho = par,
                  gamma = delta,
                  reg_type = 'percent',
                  reg_threshold = 0.0,
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


###


wignerb = wdd.wdd(b_n,
                  ptycho = par,
                  gamma = 3,
                  reg_type = 'percent',
                  reg_threshold = 0.0,
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



