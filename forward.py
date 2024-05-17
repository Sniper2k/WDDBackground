# -*- coding: utf-8 -*-
"""
Created on Sun May 29 12:18:48 2022

@author: oleh.melnyk
"""

import numpy as np
import numpy.random
import math
# import ctypes
# import multiprocessing as mp
import copy

def loc_grid_noncirc(obj_s, win_s, shift_s,float_shift):
    locations_1 = np.array(np.arange(0,obj_s[0]-win_s[0]+1, shift_s[0]))
    locations_2 = np.array(np.arange(0,obj_s[1]-win_s[1]+1, shift_s[1]))
    if float_shift:
        locations = np.zeros((len(locations_1)*len(locations_2),2),dtype=float)
    else:
        locations = np.zeros((len(locations_1)*len(locations_2),2),dtype=int)
    locations[:,0] = np.repeat(locations_1,len(locations_2))
    locations[:,1] = np.tile(locations_2,len(locations_1))
    
    return locations    

def loc_grid_circ(obj_s, shift_s,float_shift):
    locations_1 = np.array(np.arange(0,obj_s[0],shift_s[0]))
    locations_2 = np.array(np.arange(0,obj_s[1],shift_s[1]))
    # locations_1 = np.array(range(0,obj_s[0], shift_s[0]))
    # locations_2 = np.array(range(0,obj_s[1], shift_s[1]))
    if float_shift:
        locations = np.zeros((len(locations_1)*len(locations_2),2),dtype=float)
    else:
        locations = np.zeros((len(locations_1)*len(locations_2),2),dtype=int)
        
    locations[:,0] = np.repeat(locations_1,len(locations_2))
    locations[:,1] = np.tile(locations_2,len(locations_1))

    return locations

def loc_Fermat_spiral(obj_s, win_s, seed_size):
    
    locations = []
    
    s1 = obj_s[0] - win_s[0]
    s2 = obj_s[1] - win_s[1]
    
    N = np.ceil(2*(0.5*np.max([s1,s2])/ seed_size)**2).astype(np.int32)  
    phi_0 = 8 * math.pi / (1 + math.sqrt(5))**2 
    for n in range(N):
        r = seed_size * math.sqrt(n)
        phi = n * phi_0
        x = np.round(r * math.cos(phi) + 0.5*s1).astype(np.int32)  
        y = np.round(r * math.sin(phi) + 0.5*s2).astype(np.int32)
        
        if (x < 0 or x >= s1 or y < 0 or y >= s2):
            continue
        
        locations.append([x, y])
    
    locations = np.array(locations)
    
    return locations


    
class ptycho:
    def __init__(self, **kwargs):
        
        # Object Shape
        
        assert 'object_shape' in kwargs.keys(), "Object shape is not specified"
        assert isinstance(kwargs['object_shape'], tuple), "Object shape is not a tuple"
        assert len(kwargs['object_shape']) == 2, "Object shape is not 2D"
        assert kwargs['object_shape'][0] > 0 and kwargs['object_shape'][1] >0, "Object shape should be positive"
        self.object_shape = kwargs['object_shape']
        
        self.obj = np.zeros(self.object_shape,dtype = complex)
        
        # Window (Probe)
        
        if 'window' in kwargs.keys():
            self.set_window(kwargs['window'])
        else:
            assert 'window_shape' in kwargs.keys(), "Neither window nor window_shape is specified"
            assert isinstance(kwargs['window_shape'], tuple), "Window shape is not a tuple"
            assert len(kwargs['window_shape']) == 2, "Window shape is not 2D"
            assert kwargs['window_shape'][0] > 0 and kwargs['window_shape'][1] >0, "Window shape should be positive"
            # self.window_shape = kwargs['window_shape']
            dummy_window = np.zeros(kwargs['window_shape'],dtype = complex)
            self.set_window(dummy_window)
        
        # Circular shifts
        
        if 'circular' in kwargs.keys():
            if isinstance(kwargs['circular'], bool):
                self.circular = kwargs['circular']
            else:
                print('Warning: Parameter circular is not boolean. Set to False')
                self.circular = False
        else:
            print('Warning: Parameter circular is not specified. Set to False')
            self.circular = False
         
        # Float shifts
        if 'float_shift' in kwargs.keys():
            if isinstance(kwargs['float_shift'], bool):
                self.float_shift = kwargs['float_shift']
            else:
                print('Warning: Parameter float_shift is not boolean. Set to False')
                self.float_shift = False
        else:
            print('Warning: Parameter float_shift is not specified. Set to False')
            self.float_shift = False
            
        # Scan Locations
            
        if 'locations' in kwargs.keys():
            assert isinstance(kwargs['locations'], np.ndarray), "Locations is not of type numpy.ndarray"
            assert len(kwargs['locations'].shape) == 2, "Locations is not 2D"
            assert kwargs['locations'].shape[1] == 2, "Locations should be an array of 2d coordinates"
            
            self.locations = kwargs['locations']
            self.check_outbound()
            
        elif 'loc_type' in kwargs.keys():
            if kwargs['loc_type'] == 'grid':
                assert 'shift' in kwargs.keys(), "Shift size is not given for loc_type = grid"
                
                if self.float_shift:
                    assert isinstance(kwargs['shift'], tuple) or isinstance(kwargs['shift'], int) or isinstance(kwargs['shift'], float), "Shift size is not tuple or float"
                else:
                    assert isinstance(kwargs['shift'], tuple) or isinstance(kwargs['shift'], int), "Shift size is not tuple or int. Try converting the shift to int or set float_shift = True"
                
                if isinstance(kwargs['shift'], tuple):
                    assert len(kwargs['shift']) == 2, "Shift tuple is not 2D"
                    shift = kwargs['shift']
                
                if isinstance(kwargs['shift'], int):
                    shift = (kwargs['shift'],kwargs['shift'])
                    if self.float_shift:
                        self.float_shift = False
                
                if isinstance(kwargs['shift'], float) and self.float_shift:
                    shift = (kwargs['shift'],kwargs['shift'])
                
                assert shift[0] < self.object_shape[0] and shift[1] < self.object_shape[1], "Shift is larger or equal to object shape"
                
                if shift[0] > self.window_shape[0] or shift[1] > self.window_shape[1]:
                    print('Warning: Shift is larger that window shape')
                
                if self.object_shape[0] % shift[0] != 0 or self.window_shape[0] % shift[0] != 0 or self.object_shape[1] % shift[1] != 0 or self.window_shape[1] % shift[1] != 0:
                    print('Warning: Shift is not divisor of object shape or window shape')
                
                assert shift[0] > 0 and shift[1] > 0, "Shift is negative" 
                
                self.shift =shift
                
                if self.circular:
                    self.locations = loc_grid_circ(self.object_shape, shift, self.float_shift)
                else:
                    self.locations = loc_grid_noncirc(self.object_shape, self.window_shape, shift, self.float_shift)
                    
            elif kwargs['loc_type'] == 'spiral':
                assert 'fermat_seed_size' in kwargs.keys(), "Fermat spiral seed size (fermat_seed_size) is not given for loc_type = spiral"
                assert isinstance(kwargs['fermat_seed_size'], float) or isinstance(kwargs['fermat_seed_size'], int), "Fermat spiral seed size (fermat_seed_size) is not float or int"
                assert kwargs['fermat_seed_size'] > 0, "Fermat spiral seed size should be positive"
                
                self.locations = loc_Fermat_spiral(self.object_shape, self.window_shape, kwargs['fermat_seed_size'])
            else:
                assert False, "loc_type is not in [grid, fermat spiral]"
        else:
            assert False, "Neither locations nor location type generation is given"
        
        self.R = self.locations.shape[0]
        
        # Object mask is derived from the set of locations
            
        
        
        if self.float_shift:
            self.mask = np.ones(self.object_shape)
        else:
            self.mask = np.zeros(self.object_shape)
            for loc in self.locations:
                self.mask[loc[0]:(loc[0] + self.window_shape[0]),loc[1]:(loc[1] + self.window_shape[1])]=1
            
        # Fourier dimension
            
        if 'fourier_dimension' in kwargs.keys():
            assert isinstance(kwargs['fourier_dimension'], tuple), "Fourier dimension is not a tuple"
            assert len(kwargs['fourier_dimension']) == 2, "Fourier dimension is not 2D"
            assert kwargs['fourier_dimension'][0] > 0 and kwargs['fourier_dimension'][1] >0, "Fourier dimension should be positive"
            self.fourier_dimension = kwargs['fourier_dimension']
            
            if self.fourier_dimension[0] < self.window_shape[0]:
                print('Warning: Fourier dimension[0] is smaller than window shape[0]. Fourier dimension increased')
                self.fourier_dimension[0] = self.window_shape[0]
            
            if self.fourier_dimension[1] < self.window_shape[1]:
                print('Warning: Fourier dimension[1] is smaller than window shape[1]. Fourier dimension increased')
                self.fourier_dimension[1] = self.window_shape[1]
        else:
            print('Warning: Fourier dimension is not specified, Set to object_shape')
            self.fourier_dimension = self.object_shape
            
        # Detector mask
        
        if 'detector_mask' in kwargs.keys():
            assert isinstance(kwargs['detector_mask'], np.ndarray) or isinstance(kwargs['detector_mask'], int) or isinstance(kwargs['detector_mask'], tuple) , "detector_mask is not of type numpy.ndarray, tuple or int"
            
            if isinstance(kwargs['detector_mask'], np.ndarray):
                assert len(kwargs['detector_mask'].shape) == 2, "Detector mask is not 2D"
                assert kwargs['detector_mask'].shape[0] == self.fourier_dimension[0] and kwargs['detector_mask'].shape[1] == self.fourier_dimension[1], "Detector mask shape is different from Fourier dimension"
                assert np.sum(np.isin(kwargs['detector_mask'],[0,1])) == np.prod(kwargs['detector_mask'].shape), "Detector mask should be either 0 or 1"
                
                self.detector_mask = kwargs['detector_mask']
            else:
            
                if isinstance(kwargs['detector_mask'], tuple):
                    assert len(kwargs['detector_mask']) == 2, "Detector mask size is not 2D"
                    detector_mask = kwargs['detector_mask']
                else:
                    detector_mask = (kwargs['detector_mask'],kwargs['detector_mask'])
                    
                assert detector_mask[0] > 0 and detector_mask[1] > 0, "Detector mask size is negative"
                
                if detector_mask[0] > self.fourier_dimension[0]:
                    print('Warning: Detector mask size[0] is larger than Fouried dimansion[0]. Reduced')
                    detector_mask[0] = self.fourier_dimension[0]
                    
                if detector_mask[1] > self.fourier_dimension[1]:
                    print('Warning: Detector mask size[1] is larger than Fouried dimansion[1]. Reduced')
                    detector_mask[1] = self.fourier_dimension[1]
                
                self.detector_mask =  np.zeros(self.fourier_dimension,dtype=complex)
                
                dr1 = (self.fourier_dimension[0] - detector_mask[0]) //2
                dr2 = (self.fourier_dimension[1] - detector_mask[1]) //2
                self.detector_mask[dr1:(dr1 + detector_mask[0]),dr2:(dr2 + detector_mask[1])] = 1
        else:
            self.detector_mask =  np.ones(self.fourier_dimension,dtype=complex)
            
        # multithreading
        if 'num_threads' in kwargs.keys():
            if isinstance(kwargs['num_threads'], int):
                self.num_threads = kwargs['num_threads']
            else:
                self.num_threads = 1
        else:
            self.num_threads = 1
                
    def copy(self):
        pty = ptycho(object_shape = copy.deepcopy(self.object_shape),
                      window = copy.deepcopy(self.window),
                      circular = copy.deepcopy(self.circular),
                      float_shift = copy.deepcopy(self.float_shift),
                      locations = copy.deepcopy(self.locations),
                      fourier_dimension = copy.deepcopy(self.fourier_dimension),
                      detector_mask = copy.deepcopy(self.detector_mask),
                      num_threads = copy.deepcopy(self.num_threads))
        
        pty.set_object(self.obj)
        
        if hasattr(self, 'shift'):
            pty.shift = copy.deepcopy(self.shift)
            
        if hasattr(self, 'loc_type'):
            pty.loc_type = copy.deepcopy(self.loc_type)
        
        return pty
        

    def check_outbound(self):
        if not hasattr(self, 'locations'):
            return
        
        if not hasattr(self, 'circular'):
            return
        
        # if not hasattr(self, 'object_shape'):
            # return
        
        if not hasattr(self, 'window_shape'):
            return
        
        if not self.circular and not self.float_shift:
            min_loc = np.min(self.locations, axis = 0)
            max_loc = np.max(self.locations, axis = 0)
            
            # outbound = min_loc[0] < 0 or min_loc[1] < 0 or min_loc[0] < 0 or max_loc[0] + self.window_shape[0] >= self.object_shape[0] or max_loc[1] + self.window_shape[1] >= self.object_shape[1]
            outbound = min_loc[0] < 0 or min_loc[1] < 0 or min_loc[0] < 0 or max_loc[0] + self.window_shape[0] > self.object_shape[0] or max_loc[1] + self.window_shape[1] > self.object_shape[1]
            # print(outbound,min_loc, max_loc, self.window_shape, self.object_shape)
            
            assert outbound == False, "Location out of bound"

    def set_object(self, obj):
        assert isinstance(obj, np.ndarray), "Window is not of type numpy.ndarray"
        assert len(obj.shape) == 2, "Window is not 2D"
        
        assert self.window_shape[0] <= obj.shape[0] and self.window_shape[1] <= obj.shape[1], "Window is larger than object"
        
        if hasattr(self, 'obj'):
            old_obj = self.obj
            
            if old_obj.shape[0] != obj.shape[0] or old_obj.shape[1] != obj.shape[1]:
                print('Warning: Object shape has changed')
                self.object_shape = obj.shape
                self.check_outbound()
        
        self.obj = obj.astype(complex)
        self.object_shape = obj.shape

    def set_window(self, window):
        assert isinstance(window, np.ndarray), "Window is not of type numpy.ndarray"
        assert len(window.shape) == 2, "Window is not 2D"
        
        assert window.shape[0] <= self.object_shape[0] and window.shape[1] <= self.object_shape[1], "Window is larger than object"
        
        if hasattr(self, 'window'):
            old_window = self.window
            
            if old_window.shape[0] != window.shape[0] or old_window.shape[1] != window.shape[1]:
                print('Warning: Window shape has changed')
                self.window_shape = window.shape
                self.check_outbound()
        
        self.window = window.astype(complex)
        self.window_shape = window.shape

    def forward_2D_os(self, z):
        
        # half of output dimension
        h1 = self.fourier_dimension[0] // 2 
        h2 = self.fourier_dimension[1] // 2 
        
        # half of window size
        r1 = self.window_shape[0] // 2 
        r2 = self.window_shape[1] // 2 
        
        out_padded = np.zeros(self.fourier_dimension,dtype=complex)
        out_padded[:self.window_shape[0],:self.window_shape[1]] = self.window * z
        
        # place middle of the illumineted area to the 0
        out_padded = np.roll(out_padded, -r1, axis=0)
        out_padded = np.roll(out_padded, -r2, axis=1)
        
        out_fft = np.fft.fft2(out_padded)
        
        out_fft = np.roll(out_fft, h1, axis=0)
        out_fft = np.roll(out_fft, h2, axis=1)
        
        # # cut out detector shape
        
        # # first dimension
        # t =  np.zeros((par.detector_shape[0], out_fft.shape[1]),dtype=complex)
        # if par.detector_shape[0] > out_fft.shape[0]:
        #     dr1 = (par.detector_shape[0] - out_fft.shape[0]) // 2
        #     t[dr1:(dr1 + out_fft.shape[0]),:] = out_fft
        # else:
        #     dr1 = (out_fft.shape[0] - par.detector_shape[0]) //2 
        #     t = out_fft[dr1:(dr1 + par.detector_shape[0]),:]
           
            
        # # second dimentsion
        # forw = np.zeros(par.detector_shape,dtype=complex)
        # if par.detector_shape[1] > out_fft.shape[1]:
        #     dr2 = (par.detector_shape[1] - out_fft.shape[1]) // 2 
        #     forw[:,dr2:(dr2 + out_fft.shape[1])] = t
        # else:
        #     dr2 = (out_fft.shape[1] - par.detector_shape[1])//2 
        #     forw = t[:,dr2:(dr2 + par.detector_shape[1])]
        
        forw = out_fft * self.detector_mask
        
        return forw


    def forward_adj_2D_os(self,forw):
        d1 = self.window_shape[0]
        d2 = self.window_shape[1]
            
        h1 = self.fourier_dimension[0] //2
        h2 = self.fourier_dimension[1] //2
        
        # t = np.zeros((par.detector_shape[0], par.fourier_dimension[1]),dtype=complex)
        # if par.detector_shape[1] > par.fourier_dimension[1]:
        #     dr2 = (par.detector_shape[1] - par.fourier_dimension[1])//2
        #     t = forw[:,dr2:(dr2 + par.fourier_dimension[1])]  
        # else:
        #     dr2 = (par.fourier_dimension[1] - par.detector_shape[1])//2
        #     t[:,dr2:(dr2 + par.detector_shape[1])] = forw 
        
        # unzoomed_padded = np.zeros(par.fourier_dimension,dtype=complex)
        # if par.detector_shape[0] > par.fourier_dimension[0]:
        #     dr1 = (par.detector_shape[0] - par.fourier_dimension[0])//2
        #     unzoomed_padded = t[dr1:(dr1 + par.fourier_dimension[0]),:]
        # else:
        #     dr1 = (par.fourier_dimension[0] - par.detector_shape[0])//2
        #     unzoomed_padded[dr1:(dr1 + par.detector_shape[0]),:] =  t
        
        unzoomed_padded = forw * self.detector_mask
        unzoomed_padded = np.roll(unzoomed_padded, -h2, axis=1)
        unzoomed_padded = np.roll(unzoomed_padded, -h1, axis=0)
        out_fft = np.fft.ifft2(unzoomed_padded)*np.prod(self.fourier_dimension)    
        
        r1 = d1//2
        r2 = d2//2
        
        out_fft = np.roll(out_fft, r2, axis=1)
        out_fft = np.roll(out_fft, r1, axis=0)
        
        result = self.window.conj() * out_fft[:d1,:d2]
            
        return result 


    def forward_to_meas_2D(self,forw):
        return np.abs(forw)**2

    def forward_to_meas_2D_pty(self,forw):
        return self.forward_to_meas_2D(forw)
    
    
    # def forward_os_loc(self,z,r_start, r_end):
        
    #     forw = np.zeros((self.fourier_dimension[0],self.fourier_dimension[1],r_end - r_start),dtype = complex)
        
    #     for r in np.arange(r_start,r_end):
    #         print(r)
    #         loc = self.locations[r,:] 
                
    #         if self.circular:
    #             z_r = np.roll(z,-loc[0],0)
    #             z_r = np.roll(z_r,-loc[1],1)
    #             z_r = z_r[:self.window.shape[0],:self.window.shape[1]]
    #         else:
    #             z_r = z[loc[0]:(loc[0] + self.window.shape[0]),loc[1]:(loc[1] + self.window.shape[1])]
            
    #         forw[:,:, r- r_start] = self.forward_2D_os(z_r)
            
    #     return [forw,r_start,r_end]
        
    # def forward_2D_pty(self,z):
    #     # window is vector of length delta
    #     # obj is a matrix of size d x d
    #     forw = np.zeros((self.fourier_dimension[0],self.fourier_dimension[1],self.R),dtype = complex)
        
    #     # if __name__ == 'forward': #"__main__":
    #     # pool = mp.pool.ThreadPool(self.num_threads)
    #     pool = mp.Pool(self.num_threads)
    #     step = self.R // self.num_threads
        
    #     if step * self.num_threads < self.R:
    #         step +=1
        
    #     res = []
    #     for t in range(self.num_threads):
    #         r_start = t*step
    #         r_end = np.minimum((t+1)*step,self.R)
    #         res.append(pool.apply_async(self.forward_os_loc, (z,r_start,r_end)))
        
    #     for rit in res:
    #         t = rit.get()
    #         forw[:,:,t[1]:t[2]] = t[0] 
            
    #     pool.close()
    #     pool.join()
    
    #     return forw
    
    def shift_vec(self,z,r):
        if self.float_shift:
            z_r = np.fft.fft2(z)
            z_r = np.swapaxes(np.swapaxes(z_r, 0, 1) *np.exp( 2j*np.pi*-r[0]*np.arange(self.object_shape[0]) / self.object_shape[0]), 0,1) 
            z_r = z_r * np.exp( 2j*np.pi*-r[1]*np.arange(self.object_shape[1]) / self.object_shape[1])
            z_r = np.fft.ifft2(z_r)
        else:
            z_r = np.roll(z,r[0],0)
            z_r = np.roll(z_r,r[1],1)
            
        return z_r
            
    
    def forward_2D_pty(self,z):
        # window is vector of length delta
        # obj is a matrix of size d x d
        delta1 = self.window_shape[0]
        delta2 = self.window_shape[1]
        
        forw = np.zeros((self.fourier_dimension[0],self.fourier_dimension[1],self.R),dtype = complex)
        
        for r in range(self.R):
            loc = self.locations[r,:] 
            
            if self.float_shift or self.circular:
                z_r = self.shift_vec(z, -loc)
                z_r = z_r[:delta1,:delta2]
            # if self.float_shift:
            #     z_r = np.fft.fft2(z)
            #     z_r = np.swapaxes(np.swapaxes(z_r, 0, 1) * np.exp( 2.0j*np.pi*loc[0]*np.arange(self.object_shape[0]) / self.object_shape[0]) , 0, 1)
            #     z_r = z_r * np.exp( 2.0j*np.pi*loc[1]*np.arange(self.object_shape[1]) / self.object_shape[1])
            #     # z_r = np.exp( 2.0j*np.pi*loc[1]*np.arange(self.object_shape[1]) / self.object_shape[1])* z_r
            #     z_r = np.fft.ifft2(z_r)
                
            #     # if r == 10:
            #     #     loc_i= loc.astype(int)
            #     #     z_r2 = np.roll(z,-loc_i[0],0)
            #     #     z_r2 = np.roll(z_r2,-loc_i[1],1)
                    
            #     z_r = z_r[:delta1,:delta2]
            # elif self.circular:
            #     z_r = np.roll(z,-loc[0],0)
            #     z_r = np.roll(z_r,-loc[1],1)
            #     z_r = z_r[:delta1,:delta2]
            else:
                z_r = z[loc[0]:(loc[0] + delta1),loc[1]:(loc[1] + delta2)]
            
            forw[:,:,r] = self.forward_2D_os(z_r)
        
        return forw

    def forward_adj_2D_pty(self,forw):
        # window is vector of length delta
        # forw is a array of size delta1 X delta2 X R
        delta1 = self.window_shape[0]
        delta2 = self.window_shape[1]
        
        z = np.zeros((self.object_shape[0],self.object_shape[1]),dtype = complex)
        
        for r in range(self.R):
            loc = self.locations[r,:] 
            
            if self.float_shift or self.circular:
                z_r = np.zeros(self.object_shape,dtype = complex)
                z_r[:delta1,:delta2] = self.forward_adj_2D_os(forw[:,:,r])
                z_r = self.shift_vec(z_r, loc)
                z += z_r
            # if self.float_shift:
            #     z_r = np.zeros(self.object_shape,dtype = complex)
            #     z_r[:delta1,:delta2] = self.forward_adj_2D_os(forw[:,:,r])
            #     z_r = np.fft.fft2(z_r)
            #     z_r = np.swapaxes(np.swapaxes(z_r, 0, 1) * np.exp( 2j*np.pi*-loc[0]*np.arange(self.object_shape[0]) / self.object_shape[0]),0,1)
            #     z_r = z_r* np.exp( 2j*np.pi*-loc[1]*np.arange(self.object_shape[1]) / self.object_shape[1])
            #     z_r = np.fft.ifft2(z_r)
            #     z += z_r
            # if self.circular:
            #     z_r = np.zeros(self.object_shape,dtype = complex)
            #     z_r[:delta1,:delta2] = self.forward_adj_2D_os(forw[:,:,r])       
            #     z_r = np.roll(z_r,loc[0],0)
            #     z_r = np.roll(z_r,loc[1],1)
            #     z += z_r
            else:
                z[loc[0]:(loc[0] + delta1),loc[1]:(loc[1] + delta2)] += self.forward_adj_2D_os(forw[:,:,r])
        
        return z