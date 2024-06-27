# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:35:34 2024

@author: oleh.melnyk
"""

import numpy as np
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cmath


def image_to_object(modulus,phase,satur_parser):
    modulus_hsv = rgb2hsv(modulus)
    phase_hsv = rgb2hsv(phase)
    
    modulus = modulus_hsv[:,:,2] * 255 
    phase = phase_hsv[:,:,2] * 255
    phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase)) * 2*np.pi - np.pi
    
    obj_mod = modulus * np.exp(1.0j * phase)
    
    return obj_mod

def image_to_phase_object(im,satur_parser):
    im_hsv = rgb2hsv(im)
       
    modulus = im_hsv[:,:,2] * 255 
    phase = (modulus - np.min(modulus)) / (np.max(modulus) - np.min(modulus)) * 2*np.pi - np.pi
        
    # phase object    
    obj_mod =  np.exp(1.0j * phase)
    
    return obj_mod

def show_object(obj):
    colors = ["black", "lightgray", "black"]
    cmap = LinearSegmentedColormap.from_list("", colors)
    colors = ["black", "gray", "lightgray", "white"]
    cmap2 = LinearSegmentedColormap.from_list("", colors)
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    
    modulus = np.abs(obj)#np.repeat(np.round(np.abs(obj)).astype(np.uint8)[:, :,np.newaxis], 3, axis=2)

    phase = np.ones((obj.shape[0],obj.shape[1],3))
    phase[:,:,0] = (np.angle(obj) + cmath.pi)/(2*cmath.pi) 
    phase_rgb = phase[:,:,0] #hsv2rgb(phase)
    
    phase_rgb =  (np.angle(obj) - np.min(np.angle(obj))) / (np.max(np.angle(obj) -  np.min(np.angle(obj)))) * 2*np.pi - np.pi# + cmath.pi)/(2*cmath.pi) 

    ax[0].imshow(modulus, cmap = cmap2, vmin = 0, vmax  = 135000)
    ax[0].axis('off')
    
    ax[1].imshow(phase_rgb,cmap =  cmap, vmin = -np.pi, vmax = np.pi, interpolation="nearest") 
    ax[1].axis('off')
       
    plt.show()
    
def show_phase_object(obj):
    colors = ["black", "lightgray", "black"]
    cmap = LinearSegmentedColormap.from_list("", colors)
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    
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
                
    ax.imshow(obj,'gray', vmin = np.around(np.min(obj),1), vmax = np.around(np.max(obj),1))
    ax.axis('off')
            
    plt.show()
