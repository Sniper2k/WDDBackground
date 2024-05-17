import numpy as np
import sys
sys.path.insert(1, '../model')
sys.path.insert(1, '../algorithms')

import time
import sigpy

import forward as forward


import wigner_2D as wdd

import wigner_phase_object_background_removal as wdd_background

import preprocessing as preprocessing


import utility_2D as util
from scipy.ndimage import zoom


import cmath
from skimage.color import rgb2hsv

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

colors = ["black", "lightgray", "black"]
cmap = LinearSegmentedColormap.from_list("", colors)




def image_to_object(im,satur_parser):
    im_hsv = rgb2hsv(im)
       
    modulus = im_hsv[:,:,2] * 255 
    phase = (modulus - np.min(modulus)) / (np.max(modulus) - np.min(modulus)) * 2*np.pi - np.pi
        
    # phase object    
    obj_mod =  np.exp(1.0j * phase)
    
    return obj_mod


def show_object(obj):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    
    modulus = np.abs(obj)

    phase = np.ones((obj.shape[0],obj.shape[1],3))
    phase[:,:,0] = (np.angle(obj) + cmath.pi)/(2*cmath.pi) 
    phase_rgb = phase[:,:,0] 
    
    phase_rgb =  (np.angle(obj) - np.min(np.angle(obj))) / (np.max(np.angle(obj) -  np.min(np.angle(obj)))) * 2*np.pi - np.pi

    
    ax.imshow(phase_rgb,cmap =  cmap, vmin = -np.pi, vmax = np.pi, interpolation="nearest") 
    ax.axis('off')
       
    plt.show()
    
    
    
def show_diffpat(obj):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    obj = np.log10(obj)
                
    ax.imshow(obj,cmap = cmap)
    ax.axis('off')
            
    plt.show()
    


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


d = im.shape[0]


delta = 16

shift = 1

f_dim = (d,d)

dsize = f_dim


locations_2d = forward.loc_grid_circ((d,d),(shift,shift),False)


show_object(obj)


cov_mat = np.eye(2,dtype = complex)/0.05
mu = np.array([0.5 + delta*0.5, 0.5 + delta*0.5])
gauss = lambda x: np.exp(-0.5* (( x - mu).conj().T).dot((cov_mat/delta**2).dot(x - mu)))
window = np.zeros((delta,delta), dtype = complex);
for ix in range(delta):
    for iy in range(delta):
        window[ix,iy] = gauss( np.array([ix+1, iy+1]))


np.random.seed(3); r1 = np.random.rand(delta,delta)
np.random.seed(5); r2 = np.random.rand(delta,delta)
if (np.mod(d,2) == 0):
    window = window *np.exp(2j*r1) + 0.25 * r2
else:
    window = window *np.exp(2j*r1) 

show_object(window)
           

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

phantom = sigpy.shepp_logan(dsize, dtype='float64')*255 

background = np.zeros(b.shape, dtype = 'float64')
for r in range(b.shape[2]):
            background[:,:,r] = phantom


scaling = 17  
#scaling = 2 # low noise
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
                  mg_diagonals_param = .0,
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

obj_r = np.exp(1.0j*np.angle(obj_r))
print( 'Relative error after magnitude adjustment: ', util.relative_error(obj,obj_r,par.mask) )
f_r = par.forward_2D_pty(obj_r)
b_r = par.forward_to_meas_2D_pty(f_r)
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))




### Vanilla WDD after preprocessing ###

start_time = time.time()
b_n_prep = preprocessing.PreProcessing(b_n,obj,window)

wigner = wdd.wdd(b_n_prep,
                  ptycho = par,
                  gamma = delta,
                  reg_type = 'percent',
                  reg_threshold = 0.0,
                  mg_type = 'diag',
                  mg_diagonals_type = 'percent',
                  mg_diagonals_param = .0,
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
show_object(obj_r)

print('Time: ', end_time - start_time)

f_r = par.forward_2D_pty(obj_r)
b_r = par.forward_to_meas_2D_pty(f_r)

print('Reconstruction:')
print( 'Relative error: ', util.relative_error(obj,obj_r,par.mask) )
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))


obj_r = np.exp(1.0j*np.angle(obj_r))
print( 'Relative error after magnitude adjustment: ', util.relative_error(obj,obj_r,par.mask) )
f_r = par.forward_2D_pty(obj_r)
b_r = par.forward_to_meas_2D_pty(f_r)
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))





#### Proposed method ###

wignerb = wdd_background.wdd_background(b_n,
                  ptycho = par,
                  gamma = delta,
                  reg_type = 'percent',
                  reg_threshold = 0.0,
                  mg_type = 'diag',
                  mg_diagonals_type = 'percent',
                  mg_diagonals_param = .0,
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
show_object(obj_r)

print('Time: ', end_time - start_time)

f_r = par.forward_2D_pty(obj_r)
b_r = par.forward_to_meas_2D_pty(f_r)

print('Reconstruction:')
print( 'Relative error: ', util.relative_error(obj,obj_r,par.mask) )
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))



###

wignerb = wdd_background.wdd_background(b_n,
                  ptycho = par,
                  gamma = 3,
                  reg_type = 'percent',
                  reg_threshold = 0.0,
                  mg_type = 'diag',
                  mg_diagonals_type = 'percent',
                  mg_diagonals_param = .0,
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
show_object(obj_r)

print('Time: ', end_time - start_time)

f_r = par.forward_2D_pty(obj_r)
b_r = par.forward_to_meas_2D_pty(f_r)

print('Reconstruction:')
print( 'Relative error: ', util.relative_error(obj,obj_r,par.mask) )
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))


