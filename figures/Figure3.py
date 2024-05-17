import numpy as np
import sys
sys.path.insert(1, '..')
import time

import forward as forward

import wigner_2D as wdd

import preprocessing as preprocessing

import utility_2D as util
from scipy.ndimage import zoom

import cmath
from skimage.color import rgb2hsv

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

colors = ["black", "lightgray", "black"]
cmap = LinearSegmentedColormap.from_list("", colors)

### Helper functions ###

def image_to_object(im,satur_parser):
    im_hsv = rgb2hsv(im)
       
    modulus = im_hsv[:,:,2] * 255 
    phase = (modulus - np.min(modulus)) / (np.max(modulus) - np.min(modulus)) * 2*np.pi - np.pi
    
    obj_mod = modulus * np.exp(1.0j * phase)
    
    return obj_mod

def show_object(obj):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    
    modulus = np.abs(obj)#np.repeat(np.round(np.abs(obj)).astype(np.uint8)[:, :,np.newaxis], 3, axis=2)

    phase = np.ones((obj.shape[0],obj.shape[1],3))
    phase[:,:,0] = (np.angle(obj) + cmath.pi)/(2*cmath.pi) 
    phase_rgb = phase[:,:,0] #hsv2rgb(phase)
    
    phase_rgb =  (np.angle(obj) - np.min(np.angle(obj))) / (np.max(np.angle(obj) -  np.min(np.angle(obj)))) * 2*np.pi - np.pi# + cmath.pi)/(2*cmath.pi) 


    ax[0].imshow(modulus, cmap = 'gray', vmin = 0, vmax  = 2*65025 - 4000)
    ax[0].axis('off')
    
    ax[1].imshow(phase_rgb,cmap =  cmap, vmin = -np.pi, vmax = np.pi, interpolation="nearest") 
    ax[1].axis('off')
       
    plt.show()
  
def show_diffpat(obj):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    obj = np.log10(obj)
                
    ax.imshow(obj,'gray', vmin = np.around(np.min(obj),1), vmax = np.around(np.max(obj),1))
    ax.axis('off')
            
    plt.show()
    

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

obj = image_to_object(im,lambda x,v: x)

show_object(obj)

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
        

show_object(window)
    
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

show_diffpat(b[:,:,0])


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

show_diffpat(b_n[:,:,0])

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
show_object(obj_r)

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
show_object(obj_r)

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
show_object(obj_r)

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
show_object(obj_r)

print('Time: ', end_time - start_time)

f_r = par.forward_2D_pty(obj_r)
b_r = par.forward_to_meas_2D_pty(f_r)

print('Reconstruction:')
print( 'Relative error: ', util.relative_error(obj,obj_r,par.mask) )
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))



