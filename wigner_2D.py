import numpy as np
# from scipy.sparse.linalg import eigsh

import sys
sys.path.insert(1, '../model')
import forward as forward
# import utility_2D as util

# from recordtype import recordtype
import copy

from scipy.sparse.linalg import eigsh

# def wdd_params_object():
    # return recordtype("wdd_params", "ptycho_params, shift_size, as_wtype, as_threshold, reg_threshold, mg_type,mg_factor, circ_shifts, add_dummy, subspace_completion, sbc_threshold")


class wdd:
    def __init__(self, 
                 measurements, 
                 **kwargs
                 # wddpar
                 ):
        self.b = measurements
        
        assert 'ptycho' in kwargs.keys(), "Forward model of ptycho class is not given."
        assert isinstance(kwargs['ptycho'], forward.ptycho), "ptycho is not an instance of class ptycho."
        self.par = kwargs['ptycho'].copy()
        
        assert hasattr(self.par, 'shift'), "ptycho has no atribute shift"  
        
        #### GAMMA ###
        
        if 'gamma' in kwargs.keys():
            self.gamma = kwargs['gamma']    
        
        
        ###### REGULARIZATION PARAMETERS ######
        
        if 'reg_type' in kwargs.keys():
            if not isinstance(kwargs['reg_type'], str):
                print('reg_type is given, but not a string. Set to value with 10**-12.')
                self.reg_type = 'value'
                self.reg_threshold = 10**-12
                
            if kwargs['reg_type'] == 'value':
                self.reg_type = 'value'
                
                if 'reg_threshold' in kwargs.keys():
                    if not isinstance(kwargs['reg_threshold'], float):
                        print('reg_threshold is given, but not a float. Set to 10**-12.')
                        self.reg_threshold = 10**-12
                    else:
                        self.reg_threshold  = kwargs['reg_threshold']
                else:
                    print('reg_threshold is not given. Set to 10**-12.')
                    self.reg_threshold = 10**-12
            elif kwargs['reg_type'] == 'percent':
                self.reg_type = 'percent'
                
                if 'reg_threshold' in kwargs.keys():
                    if not isinstance(kwargs['reg_threshold'], float):
                        print('reg_threshold is given, but not a float. Set to 0.')
                        self.reg_threshold = 0.0
                    elif 0 > kwargs['reg_threshold'] or kwargs['reg_threshold'] > 1:
                        print('reg_threshold is given, but not in [0,1]. Set to 0.')
                        self.reg_threshold = 0.0
                    else:
                        self.reg_threshold  = kwargs['reg_threshold']
                else:
                    print('reg_threshold is not given. Set to 0.')
                    self.reg_threshold = 0.0
            else:
                print('reg_type take on of the values: value or percent. Set to value with 10**-12.')
                self.reg_type = 'value'
                self.reg_threshold = 10**-12
        else:
            self.reg_type = 'value'
            self.reg_threshold = 10**-12
        
        ###### SUBSPACE COMPLETION PARAMETERS ######
        
        if 'subspace_completion' in kwargs.keys():
            if isinstance(kwargs['subspace_completion'], bool):
                self.subspace_completion = kwargs['subspace_completion']
            else:
                self.subspace_completion = False
        else:    
            self.subspace_completion = False
        
        if self.subspace_completion:
            if 'sbc_threshold' in kwargs.keys():
                if not isinstance(kwargs['sbc_threshold'], float):
                    print('sbc_threshold is given, but not a float. Set to 10**-12.')
                    self.sbc_threshold = 10**-12
                else:
                    self.sbc_threshold  = kwargs['sbc_threshold']
            else:
                print('sbc_threshold is not given. Set to 10**-12.')
                self.sbc_threshold = 10**-12
        
        ###### MAGNITUDE ESTIMATION PARAMETERS ######
        
        if 'mg_type' in kwargs.keys():
            if not isinstance(kwargs['mg_type'], str):
                print('mg_type is given, but not a string. Set to diag.')
                self.mg_type = 'diag'
            
            if kwargs['mg_type'] in ['diag','log']:
                self.mg_type = kwargs['mg_type']
            else:
                self.mg_type = 'diag'
                
        else:
            self.mg_type = 'diag'
        
        # if self.mg_type in ['log']:
        if 'mg_diagonals_type' in kwargs.keys():
            if not isinstance(kwargs['mg_diagonals_type'], str):
                print('mg_diagonals_type is given, but not a string. Set to all.')
                self.mg_diagonals_type = 'all'
        
            if kwargs['mg_diagonals_type'] == 'all':
                self.mg_diagonals_type = 'all'
            elif kwargs['mg_diagonals_type'] == 'value':
                self.mg_diagonals_type = 'value'
                
                if 'mg_diagonals_param' in kwargs.keys():
                    if not isinstance(kwargs['mg_diagonals_param'], tuple):
                        print('mg_diagonals_param is given, but not a tuple. Set to ptycho.window_shape')
                        self.mg_diagonals_param = self.par.window_shape
                    elif len(kwargs['mg_diagonals_param']) != 2:
                        print('mg_diagonals_param is given, but not a tuple of 2 elements. Set to ptycho.window_shape.')
                        self.mg_diagonals_param = self.par.window_shape
                    else:
                        self.mg_diagonals_param  = kwargs['mg_diagonals_param']
                else:
                    print('mg_diagonals_param is not given. Set to 10**-12.')
                    self.mg_diagonals_param = 10**-12
            elif kwargs['mg_diagonals_type'] == 'percent':
                self.mg_diagonals_type = 'percent'
                
                if 'mg_diagonals_param' in kwargs.keys():
                    if not isinstance(kwargs['mg_diagonals_param'], float):
                        print('mg_diagonals_param is given, but not a float. Set to 0.')
                        self.mg_diagonals_param = 0.0
                    elif 0 > kwargs['mg_diagonals_param'] or kwargs['mg_diagonals_param'] > 1:
                        print('mg_diagonals_param is given, but not in [0,1]. Set to 0.')
                        self.mg_diagonals_param = 0.0
                    else:
                        self.mg_diagonals_param  = kwargs['mg_diagonals_param']
                else:
                    print('mg_diagonals_param is not given. Set to 0.')
                    self.mg_diagonals_param = 0.0
            else:
                print('mg_diagonals_type take on of the values: all, value or percent. Set to all.')
                self.mg_diagonals_type = 'all'
        else:
            self.mg_diagonals_type = 'all'
                    
        ###### PHASE ESTIMATION PARAMETERS ######
        
        if 'as_wtype' in kwargs.keys():
            if not isinstance(kwargs['as_wtype'], str):
                print('as_wtype is given, but not a string. Set to weighted')
                self.as_wtype = 'weighted'
                
            if kwargs['as_wtype'] in ['unweighted','weighted','weighted_sq_amp']:
                self.as_wtype = kwargs['as_wtype']
            else:
                print('as_wtype take on of the values: unweighted, weighted or weighted_sq_amp. Set to weighted.')
                self.as_wtype = 'weighted'
        
        if 'as_threshold' in kwargs.keys():
            if not isinstance(kwargs['as_threshold'], float):
                print('as_threshold specified, but not a float. Set to default value')
                self.as_threshold = 10**-10 
            else:
                self.as_threshold = kwargs['as_threshold']
        else:
            self.as_threshold = 10**-10
            
        ##### OTHER PARAMETERS ######
        
        if 'add_dummy' in kwargs.keys():
            if isinstance(kwargs['add_dummy'], bool):
                self.add_dummy = kwargs['add_dummy']
            else:
                self.add_dummy = False
        else:    
            self.add_dummy = False
            
        if 'memory_saving' in kwargs.keys():
            if isinstance(kwargs['memory_saving'], bool):
                self.memory_saving = kwargs['memory_saving']
            else:
                self.memory_saving = False
        else:    
            self.memory_saving = False    
            
            
            
        if 'xt' in kwargs.keys():
            self.xt = kwargs['xt']
            
        # self.wddpar = wddpar
        # self.par = wddpar.ptycho_params
        self.object_shape_ext = self.par.object_shape
        
        # If we don't work with circular object and measurements,
        # we need to append object with dummy variables
        # with is the size of the window
        # (potentially size of the window -1, but it breaks divisibility of dimension)
        if (self.par.circular == False and self.add_dummy == True):
            self.object_shape_ext = self.par.object_shape + self.par.window_shape
        
        # If shift>1, compute shift size dimensions
        self.dim_c = (self.par.object_shape[0]//self.par.shift[0],self.par.object_shape[1]//self.par.shift[1])
        self.dim_c_ext = (self.object_shape_ext[0]//self.par.shift[0],self.object_shape_ext[1]//self.par.shift[1])
        self.wnd_size_c = (self.par.window_shape[0]//self.par.shift[0],self.par.window_shape[1]//self.par.shift[1])

    def compute_singular_values(self):
        self.singular_values = np.zeros((self.dim_c_ext[0],self.dim_c_ext[1],self.wnd_size_c[0],self.wnd_size_c[1]),dtype = complex)
        self.singular_values_neg = np.zeros((self.dim_c_ext[0],self.dim_c_ext[1],self.wnd_size_c[0],self.wnd_size_c[1]-1),dtype = complex)
        
        gamma1 = self.gamma
            
        for k0 in range(gamma1):
                            
            if self.gamma == self.wnd_size_c[1]:
                gamma2 = self.wnd_size_c[1]
            else:
                gamma2 = gamma1 - k0    
        
        #for k0 in range(self.wnd_size_c[0]):
            # positive second index 
            for k1 in range(gamma2):#(self.wnd_size_c[1]):
                # # Compute compressed window
                w = np.zeros(self.dim_c_ext,dtype = complex)
                w1 = self.par.window[k0*self.par.shift[0]:,k1*self.par.shift[1]:]
                w1 = w1*self.par.window[:w1.shape[0],:w1.shape[1]].conj()
                w2 = np.zeros_like(self.par.window,dtype = complex)
                w2[:w1.shape[0],:w1.shape[1]] = w1
                # w[:w1.shape[0],:w1.shape[1]] = w1
                for n0 in range(self.wnd_size_c[0]):
                    for n1 in range(self.wnd_size_c[1]):
                        w[n0,n1] = np.sum(w2[n0*self.par.shift[0]:(n0+1)*self.par.shift[0],n1*self.par.shift[1]:(n1+1)*self.par.shift[1]],axis =None)
                w_fft = np.fft.fft2(w.conj())
                w_fft = w_fft.conj()    
                
                self.singular_values[:,:,k0,k1] = w_fft
                
            # negative second index
            for k1 in range(gamma2 - 1):
                # # Compute compressed window
                
                w1 = self.par.window[k0*self.par.shift[0]:,:(self.par.window_shape[1]-(k1+1)*self.par.shift[1])]
                w2 = self.par.window[:w1.shape[0],:].conj()
                w2[:,:(self.par.window_shape[1]-w1.shape[1])] = 0
                w2[:,(self.par.window_shape[1]-w1.shape[1]):] *= w1
                
               
                w3 = np.zeros_like(self.par.window,dtype = complex)
                w3[:w2.shape[0],:w2.shape[1]] = w2
                w = np.zeros(self.dim_c_ext,dtype = complex)
                # w[:w1.shape[0],:w1.shape[1]] = w1
                for n0 in range(self.wnd_size_c[0]):
                    for n1 in range(self.wnd_size_c[1]):
                        w[n0,n1] = np.sum(w3[n0*self.par.shift[0]:(n0+1)*self.par.shift[0],n1*self.par.shift[1]:(n1+1)*self.par.shift[1]],axis =None)
                w_fft = np.fft.fft2(w.conj())
                w_fft = w_fft.conj()    
                
                self.singular_values_neg[:,:,k0,k1] = w_fft
       
       
        if self.reg_type == 'value':
             self.reg_param = self.reg_threshold
             # print( np.sum(np.abs(self.singular_values) < self.reg_param )/np.prod(self.singular_values.shape))
        elif self.reg_type == 'percent':
            all_sing = np.concatenate((self.singular_values,self.singular_values_neg),axis = 3)
            self.reg_param = np.quantile(np.abs(all_sing),self.reg_threshold, axis = None)     
               
        self.above_threshold = np.abs(self.singular_values) > self.reg_param
        self.above_threshold_neg = np.abs(self.singular_values_neg) > self.reg_param
        
        
    def construct_diagonals_fft(self):
        h1 = self.par.fourier_dimension[0] // 2 
        h2 = self.par.fourier_dimension[1] // 2
        
        print('Inversion...')
        b_shifted = np.zeros_like(self.b,dtype = complex)
        b_shifted = np.roll(self.b, -h2, axis=1)    
        b_shifted = np.roll(b_shifted, -h1, axis=0)
        
        totalshifts = len(self.par.locations)
        b_iift = np.zeros_like(self.b,dtype = complex)
        for s in range(totalshifts):
            b_iift[:,:,s] = np.fft.ifft2(b_shifted[:,:,s])
            # m[:,:,s] = np.fft.fft2(m_iift[:,:,s])
        
        # For each frequency divisible by shift recover compressed diagonal
          
        self.diags_fft = np.zeros( (self.dim_c_ext[0], self.dim_c_ext[1],self.wnd_size_c[0],self.wnd_size_c[1]), dtype = complex)
        self.diags_fft_neg = np.zeros( (self.dim_c_ext[0], self.dim_c_ext[1],self.wnd_size_c[0],self.wnd_size_c[1]-1), dtype = complex) 
        
        # add dummy measurements
        x_dummy = np.ones(self.par.window_shape,dtype=complex)
        x_dummy = x_dummy / np.linalg.norm(x_dummy,'fro') * np.sqrt(np.sum(self.b)) / np.linalg.norm(self.par.window,'fro')
        forw_dummy = self.par.forward_2D_os(x_dummy)
        diff_pat_dummy = np.abs(forw_dummy)**2
        diff_pat_dummy = np.roll(diff_pat_dummy, -h2, axis=1)    
        diff_pat_dummy = np.roll(diff_pat_dummy, -h1, axis=0)
        diff_pat_ifft_dummy = np.fft.ifft2(diff_pat_dummy)
        
        diff_pat_ifft_dummy[:,:] = 0.0
        
        # number of given scan positions
        if (self.par.circular == False):
            nshifts = ( 1 + (self.par.object_shape[0]-self.par.window_shape[0])//self.par.shift[0],1+(self.par.object_shape[1]-self.par.window_shape[1])//self.par.shift[1])
        else:
            nshifts = ( (self.par.object_shape[0])//self.par.shift[0],(self.par.object_shape[1])//self.par.shift[1])
                 
        gy,gx = np.meshgrid(range(self.dim_c_ext[0]), range(self.dim_c_ext[1]))
       
       
        gamma = self.gamma
        
        print(gamma)   
        for k0 in range(gamma):
            
            if self.gamma == self.wnd_size_c[1]:
                gamma2 = self.wnd_size_c[1]
            else:
                gamma2 = gamma - k0        
    
            # positive second index
            
            for k1 in range(gamma2): 
                
                # Apply location-wise Fourier transform
                b_cur = b_iift[k0 * self.par.shift[0],k1*self.par.shift[1],:]
                b_2d = diff_pat_ifft_dummy[k0 * self.par.shift[0],k1*self.par.shift[1]]*np.ones(self.dim_c_ext,dtype= complex)
                b_2d[:nshifts[0],:nshifts[1]] = np.reshape(b_cur,nshifts)
                b_2d_fft = np.fft.fft2(b_2d)
               
                above_threshold = self.above_threshold[:,:,k0,k1]
                self.diags_fft[above_threshold,k0,k1] = b_2d_fft[above_threshold] / self.singular_values[above_threshold,k0,k1] 
               
                # diags_fft currently correspond to conxcS-kx
                # we adjust to make xcSkconx
                f_mode = np.exp(-2j*np.pi*(gx*k0*1.0/self.dim_c_ext[0] + gy*k1*1.0/self.dim_c_ext[1]))
                self.diags_fft[:,:,k0,k1] *= f_mode
                
                #print(self.diags_fft[:,:,k0,k1])
                
            # negative second index
            
            for k1 in range(gamma2-1):
                
                # Apply location-wise Fourier transform
                b_cur = b_iift[k0 * self.par.shift[0],self.par.fourier_dimension[1]- (k1+1)*self.par.shift[1],:]
                b_2d = diff_pat_ifft_dummy[k0 * self.par.shift[0],self.par.fourier_dimension[1]- (k1+1)*self.par.shift[1]]*np.ones(self.dim_c_ext,dtype= complex)
                b_2d[:nshifts[0],:nshifts[1]] = np.reshape(b_cur,nshifts)
                b_2d_fft = np.fft.fft2(b_2d)
                        
                above_threshold = self.above_threshold_neg[:,:,k0,k1]
                self.diags_fft_neg[above_threshold,k0,k1] = b_2d_fft[above_threshold] / self.singular_values_neg[above_threshold,k0,k1] 
                
                
                # diags_fft currently correspond to conxcS-kx
                # we adjust to make xcSkconx
                f_mode = np.exp(-2j*np.pi*(gx*k0*1.0/self.dim_c_ext[0] - gy*(k1+1)*1.0/self.dim_c_ext[1]))
                self.diags_fft_neg[:,:,k0,k1] *= f_mode
               
        
        
        # above_threshold_idx = np.abs(self.singular_values) > self.reg_param
        self.diags_fft[~self.above_threshold] = 0
        self.diags_fft_neg[~self.above_threshold_neg] = 0
        
    def construct_lifted_matrix(self):
        print('Lifted matrix construction...')
        self.x_lifted_comp = np.zeros( (np.prod(self.dim_c_ext),np.prod(self.dim_c_ext)), dtype = complex)
             
        
        # lpos =  []
        # lneg =  []
        # Add reconstructed diagonals to the rank-one matrix
        for k0 in range(self.wnd_size_c[0]):
            
            #positive second index
            
            for k1 in range(self.wnd_size_c[1]):
                dk = np.fft.ifft2(self.diags_fft[:,:,k0,k1])
                for n0 in range(self.dim_c_ext[0]):
                    n0mk0_circ = (n0 - k0) % self.dim_c_ext[0]
                    for n1 in range(self.dim_c_ext[1]):
                        n1mk1_circ = (n1 - k1) % self.dim_c_ext[1]
                        self.x_lifted_comp[n0mk0_circ*self.dim_c_ext[1] + n1mk1_circ,n0* self.dim_c_ext[1] + n1] = dk[n0,n1].conj()
                        self.x_lifted_comp[n0* self.dim_c_ext[1] + n1,n0mk0_circ*self.dim_c_ext[1] + n1mk1_circ] = dk[n0,n1]
                        # lpos.append( (n0, n1, n0mk0_circ, n1mk1_circ) )
                        
            # negative second index            
            for k1 in range(self.wnd_size_c[1]-1):
                dk = np.fft.ifft2(self.diags_fft_neg[:,:,k0,k1])
                
                # tr_diag = np.roll(np.roll(self.xt.conj(), k0, axis= 0), -(k1+1), axis = 1)
                # tr_diag *= self.xt
                
                # i = 0
                for n0 in range(self.dim_c_ext[0]):
                    n0mk0_circ = (n0 - k0) % self.dim_c_ext[0]
                    for n1 in range(self.dim_c_ext[1]):
                        n1mk1_circ = (n1 + k1+1) % self.dim_c_ext[1]
                        self.x_lifted_comp[n0mk0_circ*self.dim_c_ext[1] + n1mk1_circ,n0* self.dim_c_ext[1] + n1] = dk[n0,n1].conj()
                        self.x_lifted_comp[n0* self.dim_c_ext[1] + n1,n0mk0_circ*self.dim_c_ext[1] + n1mk1_circ] = dk[n0,n1]
                        # lneg.append( (n0, n1, n0mk0_circ, n1mk1_circ) )
        
        # inter = [value for value in lpos if value in lneg]
        # i = 0
        
    def construct_diags(self):
        self.diags = np.zeros( (self.dim_c_ext[0],self.dim_c_ext[1],2*self.wnd_size_c[0] - 1, 2 * self.wnd_size_c[1] - 1), dtype = complex )
        
        diags = np.fft.ifft2(self.diags_fft, axes = (0,1))
        self.diags[:,:,:self.wnd_size_c[0], :self.wnd_size_c[1]] = np.flip(copy.deepcopy(diags),axis=(2,3))
        
        
        diags = diags.conj()
        for k0 in range(self.wnd_size_c[0]):
            diags[:,:,k0,:] = np.roll(diags[:,:,k0,:],-k0,axis = 0)
            
        for k1 in range(self.wnd_size_c[1]):
            diags[:,:,:,k1] = np.roll(diags[:,:,:,k1],-k1,axis = 1)
        
        self.diags[:,:,(self.wnd_size_c[0]-1):, (self.wnd_size_c[1]-1):] =  diags
        
        
        diags_neg = np.fft.ifft2(self.diags_fft_neg, axes = (0,1))
        
        self.diags[:,:,:self.wnd_size_c[0], self.wnd_size_c[1]:] = np.flip(copy.deepcopy(diags_neg),axis=2)
    
        diags_neg = diags_neg.conj()
        for k0 in range(self.wnd_size_c[0]):
            diags_neg[:,:,k0,:] = np.roll(diags_neg[:,:,k0,:],-k0,axis = 0)
            
        for k1 in range(self.wnd_size_c[1]-1):
            diags_neg[:,:,:,k1] = np.roll(diags_neg[:,:,:,k1],k1+1,axis = 1)
            
        diags_neg = np.flip(diags_neg, axis =3)
        
        self.diags[:,:,(self.wnd_size_c[0]-1):, :(self.wnd_size_c[1]-1)] =  diags_neg
    
    def multiply_via_diag(self, matrix, vector):
        # vector = np.roll(np.roll(vector, self.wnd_size_c[0]-1, axis = 0), self.wnd_size_c[1]-1, axis = 1)
        prod = np.zeros(self.dim_c_ext, dtype = complex )
        # prod1 = np.zeros_like(prod)
        dd0m1 = 2*self.wnd_size_c[0] - 1
        dd1m1 = 2*self.wnd_size_c[1] - 1
        
        # for n0 in range(self.dim_c_ext[0]):
        #     #suboptimal
        #     vec = np.roll(vector, -n0, axis = 0)[:dd0m1,:] 
        #     for n1 in range(self.dim_c_ext[1]):
        #         vec_supp = np.roll(vec,-n1, axis =1)[:,:dd1m1]
        #         prod[n0,n1] = np.sum(matrix[n0,n1,:,:] * vec_supp, axis = None)
                
                # # if n0==0 and n1 == 0:
                # #     for k0 in range(2*self.wnd_size_c[0]-1):
                # #         for k1 in range(2*self.wnd_size_c[1] - 1):
                # #             print(matrix[n0,n1,k0,k1]*vec_supp[k0,k1] , vec_supp[k0,k1])
        
        
        for k0 in range(dd0m1):
            vector_t0 = np.roll(vector, -k0 + self.wnd_size_c[0]-1, axis = 0)
            for k1 in range(dd1m1):
                vector_t1 = np.roll(vector_t0, -k1 + self.wnd_size_c[1]-1, axis = 1)
                prod += matrix[:,:,k0,k1] * vector_t1
                
        return prod
                        
    def reconstruct_magnitudes_from_lifted_matrix(self):
        print('Magnitude estimation...')
    
        if self.mg_type == 'diag':
            self.x_mag = np.sqrt(np.abs(np.diag(self.x_lifted_comp)))
        elif self.mg_type == 'block_mag':
            # TBD
            t=0
        elif self.mg_type == 'log':
            # idx = np.abs(self.x_lifted_comp) > self.as_threshold
            idx_0 = np.abs(self.x_lifted_comp) > 0
            idx = np.abs(self.x_lifted_comp) <= self.as_threshold
            idx_0_tr = idx_0 & idx
            
            # select used diagonals 
            used_diags = np.ones(self.wnd_size_c)
            used_diags_neg = np.ones( (self.wnd_size_c[0],self.wnd_size_c[1]-1) )
            # if self.mg_diagonals_type == 'all':
                # gamma = self.wnd_size_c
            if self.mg_diagonals_type == 'value':
                gamma0 = np.maximum(self.mg_diagonals_param[0] // self.par.shift[0],1)
                gamma1 = np.maximum(self.mg_diagonals_param[1] // self.par.shift[1],1)
                # gamma = (gamma0,gamma1)
                used_diags[gamma0:,:] = 0
                used_diags[:,gamma1:] = 0
                used_diags_neg[gamma0:,:] = 0
                used_diags_neg[:,gamma1-1:] = 0
            elif self.mg_diagonals_type == 'percent':
                above_threshold_per_diag = np.sum(self.above_threshold,axis = (0,1))/np.prod(self.dim_c_ext)
                used_diags = above_threshold_per_diag >= self.mg_diagonals_param
                
                above_threshold_per_diag_neg = np.sum(self.above_threshold_neg,axis = (0,1))/np.prod(self.dim_c_ext)
                used_diags_neg = above_threshold_per_diag_neg >= self.mg_diagonals_param
                
                if (np.sum(used_diags) + np.sum(used_diags_neg) == 0):
                    used_diags[0,0]=1        
              
            # construct matrix B^*B corresponding to the index set of used diagonals    
             
            B_lam = np.zeros(self.dim_c_ext)
            B_lam[:self.wnd_size_c[0],:self.wnd_size_c[1]] = used_diags
            B_lam[(self.dim_c_ext[0]-self.wnd_size_c[0]+1):,(self.dim_c_ext[1]-self.wnd_size_c[1]+1):] = np.flip(used_diags[1:,1:], axis = (0,1))
            B_lam[(self.dim_c_ext[0]-self.wnd_size_c[0]+1):,0] = np.flip(used_diags[1:,0], axis = (0))
            B_lam[:self.wnd_size_c[0],(self.dim_c_ext[1]-self.wnd_size_c[1]+1):] = np.flip(used_diags_neg, axis= 1)
            B_lam[(self.dim_c_ext[0]-self.wnd_size_c[0]+1):,1:self.wnd_size_c[1]] = np.flip(used_diags_neg[1:,:], axis= 0)
            B_lam[0,0] = 2*np.sum(used_diags) + 2*np.sum(used_diags_neg[1:,:]) + 2 * used_diags[0,0]        
            
            if self.mg_diagonals_type != 'all':
                gx2,gx1 = np.meshgrid(range(self.dim_c_ext[0]), range(self.dim_c_ext[0]))
                dist_x = gx1-gx2
                dist_x[dist_x < 0] += self.dim_c_ext[0]
                
                gy2,gy1 = np.meshgrid(range(self.dim_c_ext[1]), range(self.dim_c_ext[1]))
                dist_y = gy1-gy2
                dist_y[dist_y < 0] += self.dim_c_ext[1]
                
                dist = np.zeros( (2, np.prod(self.dim_c_ext), np.prod(self.dim_c_ext)),dtype= int)
                dist[0,:,:] = np.repeat(np.repeat(dist_x,self.dim_c_ext[1],axis = 0),self.dim_c_ext[1],axis = 1)
                dist[1,:,:] = np.tile(dist_y,(self.dim_c_ext[0],self.dim_c_ext[0]))
                
                # dist_flat = dist[0,:,:]*self.dim_c_ext[1] + dist[1,:,:]
                
                # dist[0,:,:] = (gx - gy + np.prod(self.dim_c_ext)) // self.dim_c_ext[0]
                # dist[1,:,:] = (gx - gy + np.prod(self.dim_c_ext)) % self.dim_c_ext[0]
                # dist = np.maximum( , )
                # np.max.outer( dist_x,dist_y)
                # dist = np.swapaxes(dist,1,2)
                
                # dist = np.einsum('ij,kl->ikjl', dist_x,dist_y)
                # dist = np.reshape(dist,(np.prod(self.dim_c_ext),self.dim_c_ext[0],self.dim_c_ext[1]))
                # dist = np.reshape(dist,(np.prod(self.dim_c_ext),np.prod(self.dim_c_ext)))
                # B_lam_flat = B_lam.flatten()
                
                # idx_gamma = np.take_along_axis(B_lam,dist[0,:,:],axis = 0)
                # idx_gamma = B_lam_flat[dist_flat]
                idx_gamma = B_lam[dist[0,:,:],dist[1,:,:]] == 0
                idx_0_tr = idx_0_tr & ~idx_gamma
                idx = idx | idx_gamma
            
            Ab = np.zeros_like(self.x_lifted_comp)
            Ab[~idx] = np.log(np.abs(self.x_lifted_comp[~idx]))
            Ab[idx_0_tr] = self.as_threshold
            Btm = np.sum(Ab,axis=1) + np.diag(Ab)
            # Btb =idx + np.diag( 2 + np.sum(idx,axis = 1))
            # log_mag = np.linalg.lstsq(Btb, Btm, rcond = wddpar.as_threshold)
            # Btb = Btb.astype(float)
            
            # B_lam to be fixed when all diagonals are used
            
            # only positive
            # B_lam = np.zeros(self.dim_c_ext)
            # B_lam[:self.wnd_size_c[0],:self.wnd_size_c[1]] = 1
            # B_lam[(self.dim_c_ext[0]-self.wnd_size_c[0]+1):,(self.dim_c_ext[1]-self.wnd_size_c[1]+1):] = 1
            # B_lam[0,(self.dim_c_ext[1]-self.wnd_size_c[1]+1):] = 1
            # B_lam[(self.dim_c_ext[0]-self.wnd_size_c[0]+1):,0] = 1
            # B_lam[0,0] = 2*np.prod(self.wnd_size_c) + 2
            
            # with negative
            
            # B_lam = np.zeros(self.dim_c_ext)
            # B_lam[:self.wnd_size_c[0],:self.wnd_size_c[1]] = 1
            # B_lam[(self.dim_c_ext[0]-self.wnd_size_c[0]+1):,(self.dim_c_ext[1]-self.wnd_size_c[1]+1):] = 1
            # B_lam[:self.wnd_size_c[0],(self.dim_c_ext[1]-self.wnd_size_c[1]+1):] = 1
            # B_lam[(self.dim_c_ext[0]-self.wnd_size_c[0]+1):,:self.wnd_size_c[1]] = 1
            # B_lam[0,0] = (2*self.wnd_size_c[0]-1)*(2*self.wnd_size_c[1]-1) + 3
            
            # with flexible diagonals
            
            
            B_lam_ifft = np.fft.ifft2(B_lam)
            idx2 = np.abs(B_lam_ifft) > self.as_threshold
            
            Btm_res = np.reshape(Btm, self.dim_c_ext,'C')
            Btm_res = np.fft.fft2(Btm_res)
            Btm_res[idx2] = Btm_res[idx2] / B_lam_ifft[idx2]
            Btm_res[~idx2] = 0
            Btm_res = np.fft.ifft2(Btm_res)/np.prod(self.dim_c_ext)
            log_mag2 = np.reshape(Btm_res, np.prod(self.dim_c_ext),'C')
            
            self.x_mag = np.exp( log_mag2 )
    
    def reconstruct_magnitudes_from_diags(self):
        print('Magnitude estimation...')
    
        if self.mg_type == 'diag':
            self.x_mag = np.sqrt(np.abs(self.diags[:,:,self.wnd_size_c[0]-1,self.wnd_size_c[1]-1]))
        elif self.mg_type == 'block_mag':
            # TBD
            t=0
        elif self.mg_type == 'log':
            
            # idx = np.abs(self.diags) > self.as_threshold
            
            idx_0 = np.abs(self.diags) > 0
            idx = np.abs(self.diags) <= self.as_threshold
            idx_0_tr = idx_0 & idx
            
            used_diags = np.ones(self.wnd_size_c)
            used_diags_neg = np.ones( (self.wnd_size_c[0],self.wnd_size_c[1]-1) )
            
            if self.mg_diagonals_type == 'value':
                gamma0 = np.maximum(self.mg_diagonals_param[0] // self.par.shift[0],1)
                gamma1 = np.maximum(self.mg_diagonals_param[1] // self.par.shift[1],1)
                used_diags[gamma0:,:] = 0
                used_diags[:,gamma1:] = 0
                used_diags_neg[gamma0:,:] = 0
                used_diags_neg[:,gamma1-1:] = 0
            elif self.mg_diagonals_type == 'percent':
                above_threshold_per_diag = np.sum(self.above_threshold,axis = (0,1))/np.prod(self.dim_c_ext)
                used_diags = above_threshold_per_diag >= self.mg_diagonals_param
                
                above_threshold_per_diag_neg = np.sum(self.above_threshold_neg,axis = (0,1))/np.prod(self.dim_c_ext)
                used_diags_neg = above_threshold_per_diag_neg >= self.mg_diagonals_param
                
                if (np.sum(used_diags) + np.sum(used_diags_neg) == 0):
                    used_diags[0,0]=1 
            
            B_lam = np.zeros(self.dim_c_ext)
            B_lam[:self.wnd_size_c[0],:self.wnd_size_c[1]] = used_diags
            B_lam[(self.dim_c_ext[0]-self.wnd_size_c[0]+1):,(self.dim_c_ext[1]-self.wnd_size_c[1]+1):] = np.flip(used_diags[1:,1:], axis = (0,1))
            B_lam[(self.dim_c_ext[0]-self.wnd_size_c[0]+1):,0] = np.flip(used_diags[1:,0], axis = (0))
            B_lam[:self.wnd_size_c[0],(self.dim_c_ext[1]-self.wnd_size_c[1]+1):] = np.flip(used_diags_neg, axis= 1)
            B_lam[(self.dim_c_ext[0]-self.wnd_size_c[0]+1):,1:self.wnd_size_c[1]] = np.flip(used_diags_neg[1:,:], axis= 0)
            B_lam[0,0] = 2*np.sum(used_diags) + 2*np.sum(used_diags_neg[1:,:]) + 2 * used_diags[0,0]
            
            # filter our diagonal which are not used
            
            if self.mg_diagonals_type != 'all':
                unused_diags = np.zeros((self.diags.shape[2],self.diags.shape[3]),dtype= int)
                unused_diags[:self.wnd_size_c[0], :self.wnd_size_c[1]] = np.flip(used_diags)
                unused_diags[self.wnd_size_c[0]-1:, self.wnd_size_c[1]-1:] = used_diags
                unused_diags[self.wnd_size_c[0]-1:, :self.wnd_size_c[1]-1] = np.flip(used_diags_neg,axis = 1)
                unused_diags[:self.wnd_size_c[0], self.wnd_size_c[1]:] = np.flip(used_diags_neg,axis = 0)
                # unused_diags = 1- unused_diags
                
                idx[:,:,unused_diags==0] = True
                idx_0_tr[:,:,unused_diags==0] = False
                 
            
            Ab = np.zeros_like(self.diags)
            Ab[~idx] = np.log(np.abs(self.diags[~idx]))
            Ab[idx_0_tr] = self.as_threshold
            Btm = np.sum(Ab,axis=(2,3)) + Ab[:,:,self.wnd_size_c[0]-1,self.wnd_size_c[1]-1]
            
            # B_lam to be fixed when all diagonals are used
            
            # only positive
            # B_lam = np.zeros(self.dim_c_ext)
            # B_lam[:self.wnd_size_c[0],:self.wnd_size_c[1]] = 1
            # B_lam[(self.dim_c_ext[0]-self.wnd_size_c[0]+1):,(self.dim_c_ext[1]-self.wnd_size_c[1]+1):] = 1
            # B_lam[0,(self.dim_c_ext[1]-self.wnd_size_c[1]+1):] = 1
            # B_lam[(self.dim_c_ext[0]-self.wnd_size_c[0]+1):,0] = 1
            # B_lam[0,0] = 2*np.prod(self.wnd_size_c) + 2
            
            # with negative
            
            # B_lam = np.zeros(self.dim_c_ext)
            # B_lam[:self.wnd_size_c[0],:self.wnd_size_c[1]] = 1
            # B_lam[(self.dim_c_ext[0]-self.wnd_size_c[0]+1):,(self.dim_c_ext[1]-self.wnd_size_c[1]+1):] = 1
            # B_lam[:self.wnd_size_c[0],(self.dim_c_ext[1]-self.wnd_size_c[1]+1):] = 1
            # B_lam[(self.dim_c_ext[0]-self.wnd_size_c[0]+1):,:self.wnd_size_c[1]] = 1
            # B_lam[0,0] = (2*self.wnd_size_c[0]-1)*(2*self.wnd_size_c[1]-1) + 3
            
            
            B_lam_ifft = np.fft.ifft2(B_lam)
            idx2 = np.abs(B_lam_ifft) > self.as_threshold
            
            Btm_res = np.reshape(Btm, self.dim_c_ext,'C')
            Btm_res = np.fft.fft2(Btm_res)
            Btm_res[idx2] = Btm_res[idx2] / B_lam_ifft[idx2]
            Btm_res[~idx2] = 0
            Btm_res = np.fft.ifft2(Btm_res)/np.prod(self.dim_c_ext)
            log_mag2 = np.reshape(Btm_res, np.prod(self.dim_c_ext),'C')
            
            self.x_mag = np.exp( log_mag2 ) 
            self.x_mag = np.reshape(self.x_mag,self.dim_c_ext)
        
    def angular_sync_from_lifted_matrix(self):    
        print('Phase estimation...')
    
        if self.as_wtype == 'weighted':
            idx = np.abs(self.x_lifted_comp) > self.as_threshold
            weights = np.abs(self.x_lifted_comp) * idx 
        elif self.as_wtype == 'unweighted':
            weights = (np.abs(self.x_lifted_comp) > self.as_threshold).astype(float)
    
        idx = weights > 0
        ph_diff = np.zeros_like(self.x_lifted_comp)
        ph_diff[idx] = self.x_lifted_comp[idx]/np.abs(self.x_lifted_comp[idx])
        degree = np.sum(weights,axis = 1)
        laplacian = np.diag(degree) - ph_diff * weights
        sig,v = eigsh(laplacian,1,which = 'SM')
        
        self.phases = v[:,0]
        idx = np.abs(self.phases)>self.as_threshold
        self.phases[idx] = self.phases[idx]/np.abs(self.phases[idx]) 
        self.phases[~idx] = 1
    
    def angular_sync_from_diags(self):    
        print('Phase estimation...')
    
        if self.as_wtype == 'weighted':
            idx = np.abs(self.diags) > self.as_threshold
            weights = np.abs(self.diags) * idx 
        elif self.as_wtype == 'unweighted':
            weights = (np.abs(self.diags) > self.as_threshold).astype(float)
        
        
        used_diags = np.ones(self.wnd_size_c)
        used_diags_neg = np.ones( (self.wnd_size_c[0],self.wnd_size_c[1]-1) )
        if self.mg_diagonals_type == 'value':
            gamma0 = np.maximum(self.mg_diagonals_param[0] // self.par.shift[0],1)
            gamma1 = np.maximum(self.mg_diagonals_param[1] // self.par.shift[1],1)
            used_diags[gamma0:,:] = 0
            used_diags[:,gamma1:] = 0
            used_diags_neg[gamma0:,:] = 0
            used_diags_neg[:,gamma1-1:] = 0
        elif self.mg_diagonals_type == 'percent':
            above_threshold_per_diag = np.sum(self.above_threshold,axis = (0,1))/np.prod(self.dim_c_ext)
            used_diags = above_threshold_per_diag >= self.mg_diagonals_param
            
            above_threshold_per_diag_neg = np.sum(self.above_threshold_neg,axis = (0,1))/np.prod(self.dim_c_ext)
            used_diags_neg = above_threshold_per_diag_neg >= self.mg_diagonals_param
            
            if (np.sum(used_diags) + np.sum(used_diags_neg) == 0):
                used_diags[1,0]=1
                used_diags[0,1]=1
        
        idx = np.abs(self.diags) <= self.as_threshold    
        if self.mg_diagonals_type != 'all':
            unused_diags = np.zeros((self.diags.shape[2],self.diags.shape[3]),dtype= int)
            unused_diags[:self.wnd_size_c[0], :self.wnd_size_c[1]] = np.flip(used_diags)
            unused_diags[self.wnd_size_c[0]-1:, self.wnd_size_c[1]-1:] = used_diags
            unused_diags[self.wnd_size_c[0]-1:, :self.wnd_size_c[1]-1] = np.flip(used_diags_neg,axis = 1)
            unused_diags[:self.wnd_size_c[0], self.wnd_size_c[1]:] = np.flip(used_diags_neg,axis = 0)
            # unused_diags = 1- unused_diags
            
            idx[:,:,unused_diags==0] = True
        
        weights[idx] = 0
        
        idx = weights > 0
        ph_diff = np.zeros_like(self.diags)
        ph_diff[idx] = self.diags[idx]/np.abs(self.diags[idx])
        degree = np.sum(weights,axis = (2,3))
        max_deg = np.max(degree)
        # laplacian = - ph_diff * weights
        # laplacian[:,:,self.wnd_size_c[0]-1,self.wnd_size_c[1]-1] += degree
        
        # We will search for top eigenvalue of  maxdeg*I - L = maxdeg*I - D + W \Phi
        laplacian = ph_diff * weights
        laplacian[:,:,self.wnd_size_c[0]-1,self.wnd_size_c[1]-1] +=(max_deg * np.ones_like(degree) - degree)
        
        # sig,v = eigsh(laplacian,1,which = 'SM')
        # POWER METHOD 
        
        v = np.random.randn(self.dim_c_ext[0], self.dim_c_ext[1]) + 1.0j*np.random.randn(self.dim_c_ext[0], self.dim_c_ext[1])
        v = v / np.linalg.norm(v,'fro')
        
        Nit = 20*np.ceil(np.log(np.prod(self.dim_c_ext))).astype(int)
        
        for n in range(Nit):
            v_old = v
            v = self.multiply_via_diag(laplacian,v)
            v = v / np.linalg.norm(v,'fro')
            
            diff= np.linalg.norm(v - v_old,'fro') / np.linalg.norm(v_old,'fro')
            #print ('Power Method Iteration ', n, ' RS ',diff) #  Relative Step
            
            if diff < 10**-5:
                break
        
        self.phases = v
        idx = np.abs(self.phases)>self.as_threshold
        self.phases[idx] = self.phases[idx]/np.abs(self.phases[idx]) 
        self.phases[~idx] = 1
    
        
    def construct_object(self):
        print('Joining results...')
    
        x_vec = self.x_mag * self.phases
        
        if self.memory_saving:
            x_compressed = x_vec
        else:
            x_compressed = np.reshape(x_vec,self.dim_c_ext)
            
        x_compressed_cutted = x_compressed[:self.dim_c[0],:self.dim_c[1]]
        x = np.repeat(x_compressed_cutted, self.par.shift[0], axis = 0)
        x = np.repeat(x, self.par.shift[1], axis = 1)

        return x
    
    def run(self):
        self.compute_singular_values()
        self.construct_diagonals_fft()
        # self.truncate_diag_fft()
        
        if self.subspace_completion:
            self.subspace_completion()
            
        if self.memory_saving:
            self.construct_diags()
            self.reconstruct_magnitudes_from_diags()
            self.angular_sync_from_diags()
        else:
            self.construct_lifted_matrix()
            self.reconstruct_magnitudes_from_lifted_matrix()
            self.angular_sync_from_lifted_matrix()
                

        # u_vec = np.arange(np.prod(self.dim_c_ext))
        # u = np.reshape(u_vec,self.dim_c_ext)
        
        # u = np.random.rand(self.dim_c_ext[0],self.dim_c_ext[1])
        # u_vec = np.reshape(u,(np.prod(self.dim_c_ext)))
        
        # e1 = self.x_lifted_comp[0,:] * u_vec
        
        # res1 = self.x_lifted_comp.dot(u_vec)
        # res2 = self.multiply_via_diag(self.diags, u)
        
        # self.reconstruct_magnitudes()
        
        return self.construct_object()
        
