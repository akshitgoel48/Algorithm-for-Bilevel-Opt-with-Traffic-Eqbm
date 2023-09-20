# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:07:31 2023

@author: Akshit
"""

import datetime, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

datetime_obj = datetime.datetime.now()
folder_main = datetime_obj.strftime("%Y-%m-%d") + "_" + datetime_obj.strftime("%Hhr-%Mmin-%Ssec")
os.mkdir(folder_main)
folder_main = folder_main + "/"

folder_sub1 = folder_main+'Individual_figures'
os.mkdir(folder_sub1)
folder_sub1 = folder_sub1 + '/'

import numpy as np
from example_paper import run_algo_and_plot
from numpy import linalg as la
import matplotlib.pyplot as plt
import pandas as pd

#%%
W = 2

D = 10000

Max_eigvalM = {}
# alpha_vec = np.linspace(0, 4, 51)[1:47]
# alpha_vec = np.append(alpha_vec, [3.75, 3.76, 3.77, 3.78, 3.79, 3.80, 3.84])

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

for eta_k in [0.02]:    
    
    for q_w in [30]:     
            
        nu_min = np.exp(-2/eta_k)/q_w
        
        alpha0 = 1/(1+eta_k/nu_min)
        alpha_vec = [0.05] # [alpha0*1e2]
        
        q = q_w*np.ones((W,))
        q = q.astype(int)
        q = dict(zip(np.arange(W)+1, q))
        Q = sum(list(q.values()))
        
        rng = np.random.default_rng(seed=2023)
        x_k = rng.random(Q)
        x_k = x_k/la.norm(x_k)
        
        eps = np.exp(-2/(eta_k*100))/q_w # round(nu_min, 5)
        h_0 = {}
        for w in np.arange(W)+1:
            tmp = eps*np.ones(q[w])
            tmp[0] = tmp[0] + 0.95*(1-q[w]*eps);  tmp[1] = tmp[1] + 0.05*(1-q[w]*eps)
            h_0[w] = tmp
        print(h_0)
        
        for alpha_k in alpha_vec:
            print('\n')
            if alpha_k == min(alpha_vec) and alpha_k < 1/eta_k:
                h_t, R_t, Max_eigvalsStabM_t = run_algo_and_plot(W, q, x_k, D, h_0, alpha_k, eta_k, folder_main, folder_sub1, 
                                                                 plot_figs = True, D_pltIters_vec = [D]) # D_pltIters_vec = [150, 500])
                h_opt = h_t[D];    Rmat_opt = R_t[D]                     
            else:
                h_t, R_t, Max_eigvalsStabM_t = run_algo_and_plot(W, q, x_k, D, h_0, alpha_k, eta_k, folder_main, folder_sub1, 
                                                                 opt_soln = True, h_opt = h_opt, Rmat_opt = Rmat_opt, plot_figs = True,
                                                                 D_pltIters_vec = [D]) # D_pltIters_vec = [150, 500])
            # print('-'*25)
            # print('alpha_k = '+str(alpha_k))                                                
            # print(h_t[D])
            # print('\n')
            # print(pd.DataFrame(R_t[D]))
            Max_eigvalM[alpha_k] = Max_eigvalsStabM_t[D-1]    
                
# len_tmp = len(list(Max_eigvalM.keys()))
# fig = plt.figure(figsize=(5, 4))         
# ax0 = fig.add_subplot(1, 1, 1)
# ax0.plot(list(Max_eigvalM.keys()), list(Max_eigvalM.values()), 'm')
# ax0.set_xlabel(r'$\alpha_k$')
# ax0.legend([r'$\max\ \left|\mathsf{eig}\left(M_*\right)\right|$'], loc='upper left', fontsize="15")
# ax0.set_ylim(0,1.1)
# ax0.grid(True)    
# plt.suptitle(r'$W = %s,\ q_w = %s,\ \eta_k = %s,\ \nu^{\min} = %s$' %(str(W), str(q_w), str(eta_k), str(eps)))
# plt.savefig(folder_main+'Fig-Max_eigvalM_vs_alpha.pdf', bbox_inches='tight')    
# plt.close()