# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:07:01 2023

@author: Akshit
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import itertools as it
import numpy as np
from numpy import linalg as la
from scipy.linalg import block_diag

def LLObj_g(h,x,eta):
    return 0.5*((x@h)**2) + eta*sum(np.multiply(h, np.log(h))-h)

def nabla_h_g(h,x,eta):
    return (x@h)*x + eta*np.log(h)

def nabla_hh_g(h,x,eta):
    dim = np.shape(x)[0]
    return np.reshape(x,(dim,1))@np.reshape(x,(1,dim)) + eta*np.diag(1/h)

def nabla_xh_g(h,x):
    dim = np.shape(h)[0]
    return (x@h)*np.eye(dim) + np.reshape(h,(dim,1))@np.reshape(x,(1,dim))

def to_h(h_dict):
    return np.array(list(it.chain.from_iterable(list(h_dict.values()))))

def KL_divg(vec1, vec2):
    return sum(np.multiply(vec1, np.log(np.divide(vec1,vec2))))

#%%
def run_algo_and_plot(W, q, x_k, D, h0, alpha_k, eta_k, results_folder, results_folder2,
                      opt_soln = False, h_opt = None, Rmat_opt = None, plot_figs = False, D_pltIters_vec = None):
     
    dim_x = np.shape(x_k)[0]
    Q = sum(list(q.values()))
    q_w = int(Q/W)
    
    idx_dict = {1: (0, q[1]-1)}
    for w in np.arange(W-1)+2:
        idx_dict[w] = (idx_dict[w-1][1]+1, idx_dict[w-1][1]+1+q[w]-1)
    
    R_curr = np.zeros((Q,dim_x))   

    h_curr = h0
    
    h_t = [to_h(h_curr)]
    Rmat_t = [R_curr]
    R_t = [np.reshape(R_curr, (Q*dim_x,), order='F')]
    normInterM_t = []
    normStabM_t = []
    eigvalsStabM_t = []
    
    #%
    for t in range(D):
        h_full = to_h(h_curr)
        grad_h = nabla_h_g(h_full,x_k,eta_k)

        h_nxt = {}
        for w in np.arange(W)+1:
            z_w = np.multiply(h_curr[w], np.exp(-alpha_k*grad_h[idx_dict[w][0]:idx_dict[w][1]+1]))
            h_nxt[w] = z_w/sum(z_w)   
            
            B_tw = np.diag(h_nxt[w]) - np.reshape(h_nxt[w],(q[w],1))@np.reshape(h_nxt[w],(1,q[w])) 
            if w == 1:   B_t = B_tw  
            else:   B_t = block_diag(B_t, B_tw)
        
        InterM_t = np.diag(1/h_full) - alpha_k*nabla_hh_g(h_full,x_k,eta_k)
        StabM_t = B_t @ InterM_t
        R_nxt = StabM_t @ R_curr - alpha_k*(B_t @ nabla_xh_g(h_full,x_k))
        
        normInterM_t.append(la.norm(InterM_t,ord=2))
        normStabM_t.append(la.norm(StabM_t,ord=2))
        eigvalsStabM_t.append(abs(la.eigvals(StabM_t)))
        
        h_curr = h_nxt
        R_curr = R_nxt
        
        h_t.append(to_h(h_curr))
        R_t.append(np.reshape(R_curr, (Q*dim_x,), order='F'))   
        Rmat_t.append(R_curr)
    
    print(normStabM_t[D-5:D-1])
    Max_eigvalsStabM_t = []
    for t in range(D):
        Max_eigvalsStabM_t.append(max(eigvalsStabM_t[t]))
    #%=========================================================================
    import matplotlib.pyplot as plt
    
    if opt_soln == False:
        h_opt = h_t[D];   Rmat_opt = Rmat_t[D]
    
    for D_pltIters in D_pltIters_vec:
        obj_t = []; eps_h_t = [];   eps_R_t = []
        for t in range(D_pltIters+1):
            obj_t.append(LLObj_g(h_t[t], x_k, eta_k))
            eps_h_t.append(KL_divg(h_opt, h_t[t]))
            eps_R_t.append(la.norm(Rmat_t[t]-Rmat_opt, ord=2)**2)
            
        Plt_Max_eigvalsStabM_t = []
        Plt_normStabM_t = []         
        for t in range(D_pltIters):
            Plt_Max_eigvalsStabM_t.append(max(eigvalsStabM_t[t]))
            Plt_normStabM_t.append(normStabM_t[t])
            
        if plot_figs == True:
            plt_title = 'D='+str(D_pltIters)+'_alpha=%.2f' %alpha_k+'_eta='+str(eta_k)+'_W='+str(W)+'_q_w='+str(q_w)
            #--------------------------------------------------------------------------
            fig = plt.figure(figsize=(5, 4))         
            ax0 = fig.add_subplot(1, 1, 1)
            ax0.plot(np.arange(D_pltIters+1), obj_t, 'm')
            ax0.set_xlabel(r'$t$'+' (# iterations)')
            ax0.legend([r'$g^\eta(h^t,\widehat x)$'], loc='upper right', fontsize="15")
            ax0.grid(True)    
            plt.savefig(results_folder2+'Fig-0_'+ plt_title +'.pdf', bbox_inches='tight')    
            plt.close()    
            
            #--------------------------------------------------------------------------
            fig = plt.figure(figsize=(5, 4))         
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.plot(np.arange(D_pltIters+1), eps_h_t, 'b')
            ax1.set_xlabel(r'$t$'+' (# iterations)')
            ax1.legend([r'$\varepsilon^{k,t}_h=\overline{\mathsf{D}}_\mathsf{KL}(h^{k,*},h^{k,t})$'], loc='upper right', fontsize="15")
            ax1.grid(True)    
            plt.savefig(results_folder2+'Fig-1_'+ plt_title +'.pdf', bbox_inches='tight')    
            plt.close()
            #--------------------------------------------------------------------------
            fig = plt.figure(figsize=(5, 4))         
            ax2 = fig.add_subplot(1, 1, 1)
            plt.plot(np.arange(D_pltIters+1), eps_R_t, 'r')
            ymin_ax2, ymax_ax2 = ax2.get_ylim()
            ax2.plot(np.argmax(eps_R_t)*np.ones(D_pltIters+1), np.linspace(0,np.max(eps_R_t),D_pltIters+1), 'k--')
            ax2.set_xlabel(r'$t$'+' (# iterations)')
            ax2.legend([r'$\varepsilon^{k,t}_\mathsf{R}=||\mathsf{R}_{k,t}-\mathsf{R}_{k,*}||^2$'], loc='upper right', fontsize="15")     
            ax2.grid(True)    
            # ax2.text(np.argmax(eps_R_t), ymin_ax2, r'$T^0_k$')
            T0_k = np.argmax(eps_R_t)
            ax2.text(T0_k+15, ymin_ax2*0.80, r'$T^0_k$')    
            plt.savefig(results_folder2+'Fig-2__'+ plt_title +'.pdf', bbox_inches='tight')    
            plt.close()
            print('T_0^k = %f' %T0_k)
            #--------------------------------------------------------------------------
            fig = plt.figure(figsize=(5, 4))         
            ax3 = fig.add_subplot(1, 1, 1)
            ax3.plot(np.arange(D_pltIters), Plt_Max_eigvalsStabM_t, 'g')
            # ax3.plot(np.arange(D_pltIters), Plt_normStabM_t, 'm')
            ax3.plot(np.arange(D_pltIters), np.ones(D_pltIters), '--')
            ax3.set_xlabel(r'$t$'+' (# iterations)')
            ax3.legend([r'$\max\ \left|\mathsf{eig}\left(M_{k,t}\right)\right|$'], loc='upper right', fontsize="15")
            # ax3.legend([r'$\max\ \left|\mathsf{eig}\left(M_{k,t}\right)\right|$',
            #             r'$\left||\mathsf{eig}\left(M_{k,t}\right)\right||$'], loc='upper right', fontsize="15")
            ax3.grid(True)   
            # ymin_ax3, ymax_ax3 = ax3.get_ylim()
            # ax3.plot(T0_k*np.ones(D_pltIters+1), np.linspace(np.min(Plt_Max_eigvalsStabM_t),1,D_pltIters+1), 'k--')
            # ax3.text(T0_k+3, ymin_ax3*1.1, r'$T^0_k$')    
            # ax3.set_ylim(0.9, 1.2)
            plt.savefig(results_folder2+'Fig-3__'+ plt_title +'.pdf', bbox_inches='tight')    
            plt.close()  
            #--------------------------------------------------------------------------
            fig = plt.figure(figsize=(5, 4))         
            ax4 = fig.add_subplot(1, 1, 1)
            ax4.plot(np.arange(D_pltIters), Plt_normStabM_t, 'm')
            ax4.plot(np.arange(D_pltIters), np.ones(D_pltIters), '--')
            ax4.set_xlabel(r'$t$'+' (# iterations)')
            ax4.legend([r'$\left||M_{k,t}\right||$'], loc='upper right', fontsize="15")
            ax4.grid(True) 
            # ax4.set_ylim(0.9, 1.2)
            # ymin_ax3, ymax_ax3 = ax3.get_ylim()
            # ax3.plot(T0_k*np.ones(D_pltIters+1), np.linspace(np.min(Plt_Max_eigvalsStabM_t),1,D_pltIters+1), 'k--')
            # ax3.text(T0_k+3, ymin_ax3*1.1, r'$T^0_k$')    
            plt.savefig(results_folder2+'Fig-4__'+ plt_title +'.pdf', bbox_inches='tight')    
            plt.close()  
            
            
            #--------------------------------------------------------------------------------------------------------------
            fig = plt.figure(figsize=(10, 8)) 
            
            # ax0 = fig.add_subplot(2, 2, 1)
            # ax0.plot(np.arange(D_pltIters+1), obj_t, 'm')
            # ax0.set_xlabel(r'$t$'+' (# iterations)')
            # ax0.legend([r'$g^\eta(h^t,\widehat x)$'], loc='upper right', fontsize="15")
            # ax0.grid(True)      
                
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot(np.arange(D_pltIters+1), eps_h_t, 'b')
            ax1.set_xlabel('Iteration No.')
            ax1.legend([r'$\varepsilon^{k,t}_h=\overline{\mathsf{D}}_\mathsf{KL}(h^{k,*},h^{k,t})$'], loc='upper right', fontsize="15")
            ax1.grid(True)
            
            ax2 = fig.add_subplot(2, 2, 3)
            plt.plot(np.arange(D_pltIters+1), eps_R_t, 'r')
            ymin_ax2, ymax_ax2 = ax2.get_ylim()
            ax2.plot(np.argmax(eps_R_t)*np.ones(D_pltIters+1), np.linspace(0,np.max(eps_R_t),D_pltIters+1), 'k--')
            ax2.legend([r'$\varepsilon^{k,t}_\mathsf{R}=||\mathsf{R}_{k,t}-\mathsf{R}_{k,*}||^2$'], loc='upper right', fontsize="15")      
            ax2.grid(True)    
            ax2.text(np.argmax(eps_R_t), ymin_ax2, r'$T^0_k$')
            # labels = ax2.get_xticks().tolist() + [np.argmax(eps_R_t)]
            # labels = [int(i) for i in labels]    
            # ax2.set_xticks(labels) 
            # labels[len(labels)-1] = r'$T^0_k$';
            # ax2.set_xticklabels(labels)    
            # ax2.set_xlabel('Iteration No.')    
            # ax2.set_xlim(left=-250)  # adjust the left leaving right unchanged
            # ax2.set_xlim(right=D+250)  # adjust the left leaving right unchanged
            
            ax3 = fig.add_subplot(2, 2, 2)
            ax3.plot(np.arange(D_pltIters), Plt_Max_eigvalsStabM_t, 'g')
            ax3.plot(np.arange(D_pltIters), np.ones(D_pltIters), '--')
            ax3.set_xlabel('Iteration No.')
            ax3.legend([r'$\max\ \left|\mathsf{eig}\left(M_{k,t}\right)\right|$'], loc='upper right', fontsize="15")
            ax3.grid(True)
            
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.plot(np.arange(D_pltIters), Plt_normStabM_t, 'm')
            ax4.plot(np.arange(D_pltIters), np.ones(D_pltIters), '--')
            ax4.set_xlabel(r'$t$'+' (# iterations)')
            ax4.legend([r'$\left||M_{k,t}\right||$'], loc='upper right', fontsize="15")
            ax4.grid(True) 
                
            plt.suptitle(r'$W = %s,\ q_w = %s,\ \alpha_k = %.2f,\ \eta_k = %s$' %(str(W), str(q_w), alpha_k, str(eta_k)))
            
            plt.savefig(results_folder+ plt_title +'.pdf' , bbox_inches='tight')
            
            plt.close()
    
    return h_t, Rmat_t, Max_eigvalsStabM_t