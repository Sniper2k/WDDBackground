"""
Based on 'Advanced denoising for X-ray ptychography' (Chang et al., 2019)
"""

import numpy as np
import forward as forward
import utility_2D as util


def ADP(window,b,d,delta,s,J,positions,K,alpha1,alpha2,r,J_0,obj_wdd,background,obj_gt,meas_gt):
    
    par = forward.ptycho(
            object_shape = obj_gt.shape,
            window = window, 
            circular = True,
            loc_type = 'grid',
            shift = s, 
            fourier_dimension = (d,d),
            float_shift = False)
    
    
    # Initialization
    u = obj_wdd
    omega =  window
    
    z = par.forward_2D_pty(u)
        
    mu = np.sqrt(background)
    
    Lambda1 = np.zeros([d,d,J],dtype = complex)
    Lambda2 = np.zeros([d,d,J],dtype = complex)
    
    rel_err = np.zeros([1,K])
    meas_err = np.zeros([1,K])
    
    S_T_omega_abs = np.zeros([2*d,2*d,J],dtype = complex)
    for j in range(J):      
        S_T_omega_abs[positions[j,0]:(positions[j,0]+delta),positions[j,1]:(positions[j,1]+delta),j]  = np.abs(omega)**2

    S_T_omega_abs = S_T_omega_abs[0:d,0:d,:] + S_T_omega_abs[0:d,d:,:]  + S_T_omega_abs[d:,0:d,:] + S_T_omega_abs[d:,d:,:]
    
    
    for k in range(K):
        
        print(k)
        
        S_T_omega_F = np.zeros([2*d,2*d,J],dtype = complex)
        for j in range(J):   
            S_T_omega_F[positions[j,0]:(positions[j,0]+delta),positions[j,1]:(positions[j,1]+delta),j]  = par.forward_adj_2D_os(z[:,:,j] + Lambda1[:,:,j]) 

        S_T_omega_F = 1/J *( S_T_omega_F[0:d,0:d,:] + S_T_omega_F[0:d,d:,:]  + S_T_omega_F[d:,0:d,:] + S_T_omega_F[d:,d:,:])
        
        u = (np.sum(S_T_omega_F,axis = 2) + alpha2 * u) / (np.sum(S_T_omega_abs,axis = 2) + alpha2 * np.ones([d,d]))

        A_omega_u = par.forward_2D_pty(u)

        del S_T_omega_F
        
        sum_mu = np.sum(mu, axis=2)
        MU_k = 1/J * np.dstack([sum_mu] * J)
        
        X_star = np.sqrt(np.abs(A_omega_u - Lambda1)**2 + np.abs(background - Lambda2)**2)
        rho = (r*(X_star) + np.sqrt((r**2)*((X_star)**2) + 4*(1+r)*b)) / (2*(1+r))

        z = rho * (A_omega_u - Lambda1) / X_star
        mu = rho * (background - Lambda2) / X_star
        
        del X_star
        del rho 
    
        Lambda1 += z - A_omega_u
        
        Lambda2 += mu - MU_k
        
        if(np.mod(k+1,J_0) == 0):
            mu_k = np.sqrt(np.maximum(np.zeros_like(np.sum(b,axis=2)), 1/J*np.sum(b - np.abs(A_omega_u)**2,axis=2)))
            mu = np.dstack([mu_k] * J)
        
        del MU_k
        
        sum_mu_lambda2 = np.sum(mu + Lambda2,axis=2)
        background = 1/J * np.dstack([sum_mu_lambda2] * J)
        

        rel_err[0,k] = util.relative_error(obj_gt,u,par.mask) 
        b_r = par.forward_to_meas_2D_pty(A_omega_u)
        meas_err[0,k] = util.relative_measurement_error(meas_gt,b_r)
        del A_omega_u
            
    background = background ** 2
    
    return u, background, rel_err, meas_err
