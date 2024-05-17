# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:23:52 2020

@author: oleh.melnyk
"""

import numpy as np
from scipy.sparse.linalg import eigsh

def align_objects(obj_tr,obj,mask):
    alpha = np.exp( 1j *np.angle(np.sum(obj[:,:].conj()*obj_tr[:,:] * mask, axis = None)))
    obj_r = alpha * obj
    return obj_r

def eliminate_linear_ambiguity(obj,win, threshold = 10**-10):
    obj_n, win_n =  normalize_object_and_window(obj,win)
    
    idx = np.abs(win_n) > threshold
    ph = np.ones_like(win_n)
    ph[idx] = win_n[idx]
    
    fr1 = np.angle(ph[1:,:] / ph[:(ph.shape[0]-1),:])
    usable1 = np.maximum(idx[1:,:],idx[:(idx.shape[0]-1),:])
    b1 = np.mean(fr1[usable1],axis = None)
    
    if np.isnan(b1):
        b1 = 0
    
    fr2 = np.angle(ph[:,1:] / ph[:,:(ph.shape[0]-1)])
    usable2 = np.maximum(idx[:,1:],idx[:,:(idx.shape[0]-1)])
    b2 = np.mean(fr2[usable2],axis = None)
    
    if np.isnan(b2):
        b2 = 0
    
    gy,gx = np.meshgrid(range(win_n.shape[0]), range(win_n.shape[1]))
    win_mode = np.exp(-1j*(gx*b1 + gy*b2))
    win_r = win * win_mode 
    
    gy,gx = np.meshgrid(range(obj.shape[0]), range(obj.shape[1]))
    obj_mode = np.exp(1j*(gx*b1 + gy*b2))
    obj_r = obj * obj_mode
    
    return obj_r, win_r
    
def eliminate_linear_ambiguity2(obj,win, threshold = 10**-10):
    obj, win =  normalize_object_and_window(obj,win)
    
    idx = np.abs(obj) > threshold
    ph = np.ones_like(obj)
    ph[idx] = obj[idx]
    
    fr1 = np.angle(ph[1:,:] / ph[:(ph.shape[0]-1),:])
    usable1 = np.maximum(idx[1:,:],idx[:(idx.shape[0]-1),:])
    b1 = np.mean(fr1[usable1],axis = None)
    
    if np.isnan(b1):
        b1 = 0
    
    fr2 = np.angle(ph[:,1:] / ph[:,:(ph.shape[0]-1)])
    usable2 = np.maximum(idx[:,1:],idx[:,:(idx.shape[0]-1)])
    b2 = np.mean(fr2[usable2],axis = None)
    
    if np.isnan(b2):
        b2 = 0
    
    gy,gx = np.meshgrid(range(win.shape[0]), range(win.shape[1]))
    win_mode = np.exp(1j*(gx*b1 + gy*b2))
    win_r = win * win_mode 
    
    gy,gx = np.meshgrid(range(obj.shape[0]), range(obj.shape[1]))
    obj_mode = np.exp(-1j*(gx*b1 + gy*b2))
    obj_r = obj * obj_mode
    
    return obj_r, win_r

def eliminate_grid_ambiguity_l1(obj, win, s, start0=0, end0=0, start1=0, end1=0,threshold = 10**-6, maxit= 100, eps = 10**-8):
    # mtype 'ls1' corresponds to constrained minimization of TV norm
    # selects lambda as a minimizaer of the total variation
    # via IRLS
    
    if end0 == 0:
        end0=obj.shape[0]
        
    if end0 - start0 % s != 0:
        end0 = start0 + ((end0 - start0) // s)*s
        
    if end1 == 0:
        end1 = obj.shape[1]
        
    if end1 - start1 % s != 0:
        end1 = start1 + ((end1 - start1) // s)*s    
    
    obj_trunc = obj[start0:end0,start1:end1]
    
    # lambd = np.ones((s,s),dtype = complex)/s**2
    lambd = np.random.normal(size=(s,s)) + 1j * np.random.normal(size=(s,s))
    lambd /= np.linalg.norm(lambd,'fro')
    for t in range(maxit):    
        Z = np.zeros((s,s,s,s), dtype = complex)
    
        for k1 in range(obj_trunc.shape[1]):
            idx2 = 0
            z2 = obj_trunc[idx2, k1]
            s1 = k1 % s
            for k0 in range(obj_trunc.shape[0]-1):
                idx1 = idx2
                idx2 = (k0+1) % s
                z1 = z2
                z2 = obj_trunc[k0+1,k1]        
                diff = 1.0/np.maximum(np.abs(z1 * lambd[idx1,s1] - z2 * lambd[idx2,s1]),eps)
                #print(diff)
                Z[idx1, s1, idx1, s1] += np.abs(z1)**2 * diff
                Z[idx2, s1, idx2, s1] += np.abs(z2)**2 * diff
                Z[idx1, s1, idx2, s1] -= z1.conj() * z2 * diff
                Z[idx2, s1, idx1, s1] -= z2.conj() * z1 * diff
            
        for k0 in range(obj_trunc.shape[0]):
            idy2 = 0
            z2 = obj_trunc[k0, idy2]
            s0 = k0 % s
            for k1 in range(obj_trunc.shape[1]-1):
                idy1 = idy2
                idy2 = (k1+1) % s
                z1 = z2
                z2 = obj_trunc[k0,k1+1]       
                diff = 1.0/np.maximum(np.abs(z1 * lambd[s0,idy1] - z2 * lambd[s0,idy2]),eps)
                Z[s0, idy1, s0, idy1] += np.abs(z1)**2 * diff
                Z[s0, idy2, s0, idy2] += np.abs(z2)**2 * diff
                Z[s0, idy1, s0, idy2] -= z1.conj() * z2 * diff
                Z[s0, idy2, s0, idy1] -= z2.conj() * z1 * diff

        Z = np.reshape(Z, (s**2, s**2))
     
        lambd_long = np.reshape(lambd, s**2)   
     
        try:
            sig,v = eigsh(Z,1,which = 'SM',v0 = lambd_long, maxiter=100)
            lambd_new = v[:,0] 
            
        except np.linalg.LinAlgError as err:
            if 'SVD did not converge in Linear Least Squares' in str(err):
                lambd_new = np.ones(s**2,dtype = complex)/s**2
            else:
                raise
        
        dist = 2 - 2*np.abs(lambd_new.dot(lambd_long.conj()))
        lambd_new = np.reshape(lambd_new, (s,s))
        
        print(t,sig,dist)
        
        if (dist<threshold):
            lambd = lambd_new
            break
        lambd = lambd_new

    lambd_avg = np.mean(np.abs(lambd))
    lambd /= lambd_avg
    lambd[ np.abs(lambd) < threshold] = 1
    
    offset0 = start0 % s 
    lambd = np.roll(lambd, offset0, 0)
    
    offset1 = start1 % s 
    lambd = np.roll(lambd, offset1, 1)
      
    lambd_obj = np.tile(lambd, (obj.shape[0] // s + 1, obj.shape[1] // s + 1) )
    lambd_obj = lambd_obj[:obj.shape[0],:obj.shape[1]]
    lambd_win = np.tile(lambd, (win.shape[0] // s + 1, win.shape[1] // s + 1) )
    lambd_win= lambd_obj[:win.shape[0],:win.shape[1]]
    
    obj_r = obj * lambd_obj
    win_r = win / lambd_win
    
    return obj_r,win_r, lambd

def eliminate_grid_ambiguity_l2(obj, win, s, start0=0, end0=0, start1=0, end1=0, threshold = 10**-6):
    # mtype 'ls2' corresponds to constrained minimization of l2 norm of differences
    # selects lambda as a minimizaer of the squared total variation
    
    if end0 == 0:
        end0=obj.shape[0]
        
    if end0 - start0 % s != 0:
        end0 = start0 + ((end0 - start0) // s)*s
        
    if end1 == 0:
        end1 = obj.shape[1]
        
    if end1 - start1 % s != 0:
        end1 = start1 + ((end1 - start1) // s)*s    
    
    obj_trunc = obj[start0:end0,start1:end1]
    
    Z = np.zeros((s,s,s,s), dtype = complex)
    
    for k1 in range(obj_trunc.shape[1]):
        idx2 = 0
        z2 = obj_trunc[idx2, k1]
        s1 = k1 % s
        for k0 in range(obj_trunc.shape[0]-1):
            idx1 = idx2
            idx2 = (k0+1) % s
            z1 = z2
            z2 = obj_trunc[k0+1,k1]        
            Z[idx1, s1, idx1, s1] += np.abs(z1)**2
            Z[idx2, s1, idx2, s1] += np.abs(z2)**2
            Z[idx1, s1, idx2, s1] -= z1.conj() * z2
            Z[idx2, s1, idx1, s1] -= z2.conj() * z1
            
    for k0 in range(obj_trunc.shape[0]):
        idy2 = 0
        z2 = obj_trunc[k0, idy2]
        s0 = k0 % s
        for k1 in range(obj_trunc.shape[1]-1):
            idy1 = idy2
            idy2 = (k1+1) % s
            z1 = z2
            z2 = obj_trunc[k0,k1+1]        
            Z[s0, idy1, s0, idy1] += np.abs(z1)**2
            Z[s0, idy2, s0, idy2] += np.abs(z2)**2
            Z[s0, idy1, s0, idy2] -= z1.conj() * z2
            Z[s0, idy2, s0, idy1] -= z2.conj() * z1
    
    Z = np.reshape(Z, (s**2, s**2))
    
    lambd = np.ones(s**2,dtype = complex) 
    try:
        sig,v = eigsh(Z,1,which = 'SM',v0 = lambd/s)
        lambd = v[:,0] 
        
        lambd[ np.abs(lambd) < threshold] = 1
        
    except np.linalg.LinAlgError as err:
        if 'SVD did not converge in Linear Least Squares' in str(err):
            lambd = np.ones(s**2,dtype = complex)
        else:
            raise
            
    lambd = np.reshape(lambd, (s,s))
    
    offset0 = start0 % s 
    lambd = np.roll(lambd, offset0, 0)
    
    offset1 = start1 % s 
    lambd = np.roll(lambd, offset1, 1)
    
    lambd[ np.abs(lambd) < threshold] = 1
    lambd_obj = np.tile(lambd, (obj.shape[0] // s, obj.shape[1] // s) )
    lambd_win = np.tile(lambd, (win.shape[0] // s, win.shape[1] // s) )
    
    obj_r = obj * lambd_obj
    win_r = win / lambd_win
    
    return obj_r,win_r, lambd


def relative_error(obj_tr,obj,mask):
    obj_r = align_objects(obj_tr,obj,mask)
    return np.sqrt(np.sum(np.abs(obj_r - obj_tr)**2 * mask, axis = None))/np.sqrt(np.sum(np.abs(obj_tr)**2 * mask, axis = None))

def relative_measurement_error(b, meas_obj):
    b = np.reshape(b,(np.prod(b.shape)))
    meas_obj = np.reshape(meas_obj,(np.prod(meas_obj.shape)))
    #b = np.sqrt(np.maximum(b,0))
    #meas_obj = np.sqrt(meas_obj)
    return np.linalg.norm(b-meas_obj)/np.linalg.norm(b)

def relative_sq_measurement_error(b, meas_obj):
    b = np.reshape(b,(np.prod(b.shape)))
    meas_obj = np.reshape(meas_obj,(np.prod(meas_obj.shape)))
    # b = np.sqrt(np.maximum(b,0))
    # meas_obj = np.sqrt(meas_obj)
    return np.linalg.norm(b-meas_obj)/np.linalg.norm(b)

def log10_measurement_error(b, meas_obj):
    b = np.reshape(b,(np.prod(b.shape)))
    meas_obj = np.reshape(meas_obj,(np.prod(meas_obj.shape)))
    b = np.sqrt(np.maximum(b,0))
    meas_obj = np.sqrt(meas_obj)
    return np.log10(np.linalg.norm(b-meas_obj))

def relative_intensity_error(b, meas_obj):
    b = np.reshape(b,(np.prod(b.shape)))
    meas_obj = np.reshape(meas_obj,(np.prod(meas_obj.shape)))
    return np.linalg.norm(b-meas_obj)/np.linalg.norm(b)

def normalize_window(window):
    window /= np.linalg.norm(window,'fro')
        
    return window

def normalize_object_and_window(obj, window):
    norm = np.linalg.norm(window,'fro')
    window /= norm
    obj *= norm
    
    return obj, window

