# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:35:00 2023

@author: Akshit
"""

import os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import itertools as it
import numpy as np
from numpy import linalg as la
from autograd import jacobian
import autograd.numpy as np_ag
import matplotlib.pyplot as plt

def to_h(h_dict):
    return np.array(list(it.chain.from_iterable(list(h_dict.values()))))

def KL_divg(vec1, vec2):
    return sum(np.multiply(vec1, np.log(np.divide(vec1,vec2))))

def c_a(z):
    y = z[0:dim_y]
    x = z[dim_y:dim_x+dim_y]
    return A_a + np_ag.multiply(B_a, np_ag.divide(y,K_a+x)**4)

def run_algo_v2(tmpA_a, tmpB_a, tmpK_a, barDelta, W, q, D, alpha, eta, x_k, k, NormalzMat, h_init=None):
    global dim_x, dim_y, A_a, B_a, K_a
    dim_x = np.shape(x_k)[0]
    dim_y = np.shape(barDelta)[1]
    
    A_a = tmpA_a;   B_a = tmpB_a;   K_a = tmpK_a
        
    if k==0:
        h_0 = {}
        for w in np.arange(W)+1:
            h_0[w] = 1/q[w]*np.ones(q[w])
        h_0 = to_h(h_0)
    else:
        h_0 = h_init
    
    Q = sum(list(q.values()))
    q_w = int(Q/W)
    
    idx_dict = {1: (0, q[1]-1)}
    for w in np.arange(W-1)+2:
        idx_dict[w] = (idx_dict[w-1][1]+1, idx_dict[w-1][1]+1+q[w]-1)
    
    h_curr = h_0
    R_curr = np.zeros((Q,dim_x))   
    
    h_t = [h_curr]
    Rmat_t = [R_curr]
    
    jacobian_cost = jacobian(c_a)
    #%
    for t in range(D):
        
        # start = time.time()   
        ''' Projected Mirror Descent '''
        Z_k = np.concatenate((barDelta.T@h_curr, x_k))
        grad_h = barDelta@c_a(Z_k) + eta*np.log(h_curr)        
        # start2 = time.time()
        z_W = np.multiply(h_curr, np.exp(-alpha*grad_h))
        h_nxt = np.divide(z_W, NormalzMat@z_W)
        # print("--> Time elapsed in h-update step in Iteration %d: %fs" %(t,time.time()-start2))
        
        ''' Jacobian calculation '''
        # start3 = time.time()
        Jacb = jacobian_cost(Z_k)          
        nabla_hh_g = barDelta@Jacb[:,0:dim_y]@barDelta.T        
        nabla_xh_g = barDelta@Jacb[:,dim_y:dim_y+dim_x]
        B_t = np.diag(h_nxt) - np.multiply(NormalzMat, np.reshape(h_nxt,(Q,1))@np.reshape(h_nxt,(1,Q)))
        R_nxt = B_t @ ((1-eta*alpha)*np.diag(1/h_curr) - alpha*nabla_hh_g) @ R_curr - alpha*(B_t @ nabla_xh_g)
        
        h_curr = h_nxt
        R_curr = R_nxt
        
        # print("--> Remaining time elapsed in Iteration %d: %fs" %(t,time.time()-start3))
        # print("Total time elapsed in Iteration %d: %fs" %(t,time.time()-start))
                        
        h_t.append(h_curr)
        Rmat_t.append(R_curr)        
        
    # D_pltIters = D    
    # h_opt = h_t[D];   Rmat_opt = Rmat_t[D]
    
    # eps_h_t = [];   eps_R_t = []
    # for t in range(D_pltIters+1):
    #     eps_h_t.append(KL_divg(h_opt, h_t[t]))
    #     eps_R_t.append(la.norm(Rmat_t[t]-Rmat_opt, ord=2)**2)
    
    # fig = plt.figure(figsize=(5, 4))                 
    # ax1 = fig.add_subplot(1, 1, 1)
    # ax1.plot(np.arange(D_pltIters+1), eps_h_t, 'b')
    # # ax1.set_xlabel('Iteration No.')
    # ax1.set_xlabel(r'$t$'+' (# iterations)')
    # ax1.legend([r'$\varepsilon^{%s,t}_h=\overline{\mathsf{D}}_\mathsf{KL}(h^{%s,D},h^{%s,t})$' %(str(k), str(k), str(k))],
    #             loc='upper right', fontsize="15")
    # ax1.grid(True)
    
    # save_folder = 'Sioux Falls Network/Plots_LowerLevelIters'+str(D)
    # if not os.path.exists(save_folder):
    #     os.mkdir(save_folder)
    # plt.savefig(save_folder+'/Fig-1_LowerLevel_for_k=%s.pdf' %str(k), bbox_inches='tight')
    # plt.close()
    
    # fig = plt.figure(figsize=(5, 4))         
    # ax2 = fig.add_subplot(1, 1, 1)
    # plt.plot(np.arange(D_pltIters+1), eps_R_t, 'r')
    # ax2.set_xlabel(r'$t$'+' (# iterations)')
    # ymin_ax2, ymax_ax2 = ax2.get_ylim()
    # ax2.plot(np.argmax(eps_R_t)*np.ones(D_pltIters+1), np.linspace(0,np.max(eps_R_t),D_pltIters+1), 'k--')
    # ax2.legend([r'$\varepsilon^{%s,t}_\mathsf{R}=||\mathsf{R}_{%s,t}-\mathsf{R}_{%s,D}||^2$' %(str(k), str(k), str(k))], 
    #             loc='upper right', fontsize="15")      
    # ax2.grid(True)    
    # ax2.text(np.argmax(eps_R_t), ymin_ax2, r'$D^0$')
    # plt.savefig(save_folder+'/Fig-2_LowerLevel_for_k=%s.pdf' %str(k), bbox_inches='tight')
    # plt.close()
    
    # plt.suptitle(r'$W = %s,\ q_w = %s,\ \alpha = %.2f,\ \eta = %s$' %(str(W), str(q_w), alpha, str(eta)))
    
    
    
    return h_curr, R_curr