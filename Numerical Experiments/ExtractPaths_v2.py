# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 22:25:47 2023

@author: Akshit
"""

import os, time
import pandas as pd
import numpy as np
import networkx as nx
from numpy import linalg as la
from itertools import islice

def k_shortest_paths(G, source, target, k, weight=None):
    '''
    k-shortest paths
    '''
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val in value:
            return key
        
main_folder = 'Sioux Falls Network/'
        
xls = pd.ExcelFile(main_folder+'data.xlsx')
Nodes = list(np.arange(24)+1)
Arcs_sheet = pd.read_excel(xls, 'Sheet1')
Arcs_idx = list(Arcs_sheet['Links'].values)
Arcs_idx = [a-1 for a in Arcs_idx]

Arc_node1 = Arcs_sheet['Node 1'].values
Arc_node2 = Arcs_sheet['Node 2'].values
K_a = Arcs_sheet['K_a'].values
A_a = Arcs_sheet['A_a'].values
B_a = Arcs_sheet['B_a'].values

Arcs_list = []; 
Arcs_weight = []
for i in range(len(Arc_node1)):
    Arcs_list.append((Arc_node1[i], Arc_node2[i]))
    Arcs_weight.append((Arc_node1[i], Arc_node2[i], A_a[i]))
    
Arcs_dict = dict(zip(Arcs_list,Arcs_idx))

G = nx.DiGraph()
for i in Nodes:
    G.add_node(i)
G.add_weighted_edges_from(Arcs_weight)

Demand_sheet = pd.read_excel(xls, 'Sheet2')
ODpairs = {}
Demand_dict = {}
cnt = 0; cnt2 = 0
pathDict = {}
PathsDict = {}
Num_shortest_paths = 5
for i in range(len(Nodes)):
    for j in range(len(Nodes)):
        if Demand_sheet[i+1][j]>0:
            cnt = cnt + 1
            ODpairs[cnt] = [i+1,j+1]
            Demand_dict[cnt] = Demand_sheet[i+1][j]
            pathList = []
            for paths in k_shortest_paths(G, ODpairs[cnt][0], ODpairs[cnt][1], k=Num_shortest_paths, weight='weight'):
                # k=5 shortest paths considered.
                path_link = []
                for node_idx in range(len(paths)-1):
                    path_link.append(Arcs_dict[(paths[node_idx],paths[node_idx+1])])
                pathList.append(path_link)
                cnt2 = cnt2 + 1
                PathsDict[cnt2] = path_link
            pathDict[cnt] = pathList
       
barDelta = np.zeros([cnt2,len(Arcs_list)])

for j in PathsDict.keys():
    barDelta[j-1,PathsDict[j]] = Demand_dict[get_key(PathsDict[j], pathDict)]

q_dict = dict(zip(ODpairs.keys(),Num_shortest_paths*np.ones(len(ODpairs)).astype(int)))

idx_dict = {1: (0, q_dict[1]-1)}
Normalz_mat = np.zeros((len(PathsDict),len(PathsDict)))
Normalz_mat[idx_dict[1][0]:idx_dict[1][1]+1,idx_dict[1][0]:idx_dict[1][1]+1] = 1

for w in np.arange(len(ODpairs)-1)+2:
    idx_dict[w] = (idx_dict[w-1][1]+1, idx_dict[w-1][1]+1+q_dict[w]-1)
    Normalz_mat[idx_dict[w][0]:idx_dict[w][1]+1,idx_dict[w][0]:idx_dict[w][1]+1] = 1

#%%
from InnerLoop_v2 import run_algo_v2
from autograd import grad
import autograd.numpy as np_ag
# from numpy import linalg as la

dim_y = np.shape(barDelta)[1]
dim_x = dim_y

u_a = 25*np.ones(dim_y)
d_a_sheet = Arcs_sheet['d_a'].values

NotExpnd_arc_indices = np.argwhere(np.isnan(d_a_sheet))
NotExpnd_arc_indices.shape = (len(NotExpnd_arc_indices),)

Expnd_arc_indices = np.argwhere(np.isfinite(d_a_sheet))
Expnd_arc_indices.shape = (len(Expnd_arc_indices),)

u_a[NotExpnd_arc_indices] = 0
d_a = d_a_sheet.copy()
d_a[NotExpnd_arc_indices] = 0

def UL_f(z):
    y = z[0:dim_y]
    x = z[dim_y:dim_x+dim_y]
    return (A_a + np_ag.multiply(B_a, np_ag.divide(y,K_a+x)**4)) @ y + 0.001*sum(np_ag.multiply(d_a,x**2))

grad_UL_f = grad(UL_f)

#%%
UL_Iters = 100

alpha = 0.50   # lower-level step size
beta = 0.25  # upper-level step size
eta = 1e-2

UL_obj_dict = {}
UL_hatGrad_dict = {}
x_Iter_dict = {}
Time_InnerLoop_dict = {}
Total_Time_dict = {}

LL_Iters_list = [40, 60, 80, 100, 120]
 
#%%
for Max_LL_Iters in LL_Iters_list:
    
    ## from intialization in previous papers
    x_curr = np.zeros((dim_x))
    # x_curr = 12.5*np.ones((dim_x))
    # x_curr = 6.25*np.ones((dim_x))
    
    UL_obj_k = []
    UL_hatGrad_k = []
    x_Iter_k = [x_curr[Expnd_arc_indices].tolist()]
    Time_InnerLoop_k = []
    Total_Time_k = []

    print('-'*60)
    for k in range(UL_Iters):
        if eta*alpha >= 1:
            print('** Invalid eta, alpha. The product eta & alpha >= 1. **')
            break;
        
        start = time.time()
        ''' Inner Loop (lower-level iterations) '''
        if k==0:
            h_curr, R_curr = run_algo_v2(A_a, B_a, K_a, barDelta, W = len(ODpairs), q = q_dict, D = Max_LL_Iters, alpha = alpha, 
                                         eta = eta, x_k = x_curr, k = k, NormalzMat = Normalz_mat)
        else:
            h_curr, R_curr = run_algo_v2(A_a, B_a, K_a, barDelta, W = len(ODpairs), q = q_dict, D = Max_LL_Iters, alpha = alpha, 
                                         eta = eta, x_k = x_curr, k = k, NormalzMat = Normalz_mat, h_init = h_curr)
        Time_InnerLoop_k.append(time.time()-start)
        
        if sum(np.isnan(h_curr)) > 0 or sum(sum(np.isnan(R_curr))) > 0:
            print('** NaN value encountered in either h or R iterate. **')
            break;
        
        ''' Calculate Gradient Estimator '''
        Z_k = np.concatenate((barDelta.T@h_curr, x_curr))
        Grad = grad_UL_f(Z_k)        
        nabla_h_f = barDelta@Grad[0:dim_y]       
        nabla_x_f = Grad[dim_y:dim_y+dim_x]
        hat_nabla_F = nabla_x_f + R_curr.T@nabla_h_f    
        
        ''' Approximate Projected Gradient Descent '''
        x_curr = np.minimum(np.maximum(x_curr - beta*hat_nabla_F, 0), u_a)
        end = time.time()
        
        Total_Time_k.append(end-start)
        UL_obj_k.append(UL_f(Z_k))
        UL_hatGrad_k.append(hat_nabla_F)    
        x_Iter_k.append(x_curr[Expnd_arc_indices].tolist())        
        print('Total time spent in iteration k = %d is %fs' %(k,Total_Time_k[k]))
        
    UL_obj_dict[Max_LL_Iters] = UL_obj_k
    UL_hatGrad_dict[Max_LL_Iters] = UL_hatGrad_k
    x_Iter_dict[Max_LL_Iters] = x_Iter_k
    Time_InnerLoop_dict[Max_LL_Iters] = Time_InnerLoop_k
    Total_Time_dict[Max_LL_Iters] = Total_Time_k

#%%
import pandas as pd
UL_obj_dict = {}
for Max_LL_Iters in LL_Iters_list:
    xls = pd.ExcelFile('Sioux Falls Network/ObjVal_Results.xlsx')    
    Results_sheet = pd.read_excel(xls, 'ObjVal')
    UL_obj_dict[Max_LL_Iters] = list(Results_sheet['D='+str(Max_LL_Iters)].values)
    
#%% 
import matplotlib.pyplot as plt
import numpy as np
 
UL_Plt_Iters = int(UL_Iters/2)
# line_style_dict = {40: '--', 60: '--', 80: ':', 100: '-.', 120: '--'}
# line_style_dict = {40: ':', 60: '--', 80: '-.', 100: '--', 120: '--'}
# line_style_dict = {40: '-', 60: '-.', 80: '--', 100: ':', 120: ':'}
line_style_dict = {40: ':', 60: ':', 80: '--', 100: '-.', 120: '-'}
color_dict = {40: '#ffd700', 60: 'tab:brown', 80: 'b', 100: 'g', 120: 'r'}

# marker_dict = {40: None, 60: '*', 80: '+', 100: None, 120: '1'}
marker_dict = {40: '4', 60: None, 80: None, 100: None, 120: None}

## code for generating and saving plots.
def plot_results(start_Iter, num_Iters):
    num_Iters = min(num_Iters, UL_Iters)
    xaxis = np.arange(start_Iter,start_Iter+num_Iters)
    # fig, ax = plt.figure(figsize=(6, 4))      
    fig, ax = plt.subplots(figsize=(6, 4))
    for Max_LL_Iters in LL_Iters_list:
        # if Max_LL_Iters == 100 or Max_LL_Iters == 120:
        #     ax.plot(xaxis, UL_obj_dict[Max_LL_Iters][start_Iter:start_Iter+num_Iters], label=r'$D=%s$' %str(Max_LL_Iters),
        #              linestyle = line_style_dict[Max_LL_Iters], marker = marker_dict[Max_LL_Iters],
        #              markevery = 2, color = color_dict[Max_LL_Iters]) # list(np.arange(start_Iter,start_Iter+num_Iters,2)))
        # else:
        ax.plot(xaxis, UL_obj_dict[Max_LL_Iters][start_Iter:start_Iter+num_Iters], label=r'$D=%s$' %str(Max_LL_Iters),
                 linestyle = line_style_dict[Max_LL_Iters], marker = marker_dict[Max_LL_Iters],
                 markevery = 3, color = color_dict[Max_LL_Iters]) #,
                 # color = 'blue')
    # plt.grid(True)
    # plt.legend(mode="expand")
    ax.legend(bbox_to_anchor=(1.1, 1.12), ncol=5)
    ax.set_xlabel(r'$k$'+' (# iterations)')
    ax.set_ylabel(r'$f\ (h^{k,D},y^k)$')
    
    # axes = plt.axes([.3, .25, .675, .6])
    axes = plt.axes([.36, .32, .53, .5])
    start_Iter = 50; num_Iters = 50
    xaxis = np.arange(start_Iter,start_Iter+num_Iters)
    for Max_LL_Iters in LL_Iters_list:
        # if Max_LL_Iters == 100 or Max_LL_Iters == 120:
        #     axes.plot(xaxis, UL_obj_dict[Max_LL_Iters][start_Iter:start_Iter+num_Iters], label=r'$D=%s$' %str(Max_LL_Iters),
        #              linestyle = line_style_dict[Max_LL_Iters], marker = marker_dict[Max_LL_Iters],
        #              markevery = 2, color = color_dict[Max_LL_Iters]) # list(np.arange(start_Iter,start_Iter+num_Iters,2)))
        # else:
        axes.plot(xaxis, UL_obj_dict[Max_LL_Iters][start_Iter:start_Iter+num_Iters], label=r'$D=%s$' %str(Max_LL_Iters),
                 linestyle = line_style_dict[Max_LL_Iters], marker = marker_dict[Max_LL_Iters],
                 markevery = 3, color = color_dict[Max_LL_Iters]) #,
                 # color = 'blue')
    axes.grid(True)  
    ax.indicate_inset_zoom(axes) #, edgecolor="black")
    plt.savefig('Sioux Falls Network/ULObj_Iters-%s_to_%s-v3.pdf' %(str(start_Iter),str(start_Iter+num_Iters-1)), bbox_inches='tight')
    plt.close()
    
    # plt.figure(figsize=(6, 5))         
    # for Max_LL_Iters in LL_Iters_list:
    #     xaxis = np.arange(start_Iter,start_Iter+num_Iters-1)
    #     tmp = []
    #     for k in xaxis:
    #         tmp.append(la.norm(np.subtract(x_Iter_dict[Max_LL_Iters][k+1], x_Iter_dict[Max_LL_Iters][k])))
    #     if Max_LL_Iters == 100 or Max_LL_Iters == 120:
    #         plt.plot(xaxis, tmp, label=r'$D=%s$' %str(Max_LL_Iters), linestyle = line_style_dict[Max_LL_Iters], 
    #                  marker = marker_dict[Max_LL_Iters], markevery = 2)
    #         # list(np.arange(start_Iter,start_Iter+num_Iters,2)))
    #     else:
    #         plt.plot(xaxis, tmp, label=r'$D=%s$' %str(Max_LL_Iters),                  
    #                  linestyle = line_style_dict[Max_LL_Iters], 
    #                  marker = marker_dict[Max_LL_Iters])
    #                 #, color = 'red')
    # plt.grid(True)
    # plt.legend()
    # plt.xlabel(r'$k$'+' (# iterations)')
    # plt.ylabel(r'$||x^{k+1}-x^k||$')
    # plt.savefig('Sioux Falls Network/Diff_x_Iters-%s_to_%s.pdf' %(str(start_Iter),str(start_Iter+num_Iters-1)), bbox_inches='tight')
    # plt.close()

plot_results(0, 100)

#%%
## code for saving time results
writer = pd.ExcelWriter('Sioux Falls Network/Time_Results.xlsx', index = False)
import pandas as pd
tb_array = []
colNames = []
for k in range(UL_Iters):
    tb_arr = [k]
    if k==0:    colNames = ['k']
    for Max_LL_Iters in LL_Iters_list:
        if k==0:    colNames.append('D='+str(Max_LL_Iters))
        tb_arr.append(Total_Time_dict[Max_LL_Iters][k])
    tb_array.append(tb_arr)

table = pd.DataFrame(tb_array, columns = colNames) 
table.to_excel(writer, sheet_name = 'TotalTime', index = False)

tb_array = []
colNames = []
for k in range(UL_Iters):
    tb_arr = [k]
    if k==0:    colNames = ['k']
    for Max_LL_Iters in LL_Iters_list:
        if k==0:    colNames.append('D='+str(Max_LL_Iters))
        tb_arr.append(Total_Time_dict[Max_LL_Iters][k])
    tb_array.append(tb_arr)

table = pd.DataFrame(tb_array, columns = colNames) 
table.to_excel(writer, sheet_name = 'InnerLoopTime', index = False)
writer.save()   

## code for saving ObjVal results
writer = pd.ExcelWriter('Sioux Falls Network/ObjVal_Results.xlsx', index = False)
import pandas as pd
tb_array = []
colNames = []
for k in range(UL_Iters):
    tb_arr = [k]
    if k==0:    colNames = ['k']
    for Max_LL_Iters in LL_Iters_list:
        if k==0:    colNames.append('D='+str(Max_LL_Iters))
        tb_arr.append(UL_obj_dict[Max_LL_Iters][k])
    tb_array.append(tb_arr)

table = pd.DataFrame(tb_array, columns = colNames) 
table.to_excel(writer, sheet_name = 'ObjVal', index = False)
writer.save()   

## code for saving x_Iterates
writer = pd.ExcelWriter('Sioux Falls Network/xIter_Results.xlsx', index = False)
import pandas as pd

for Max_LL_Iters in LL_Iters_list:
    tb_array = []
    colNames = ['k']
    tb_arr = [' ']
    for idx in Expnd_arc_indices:
        colNames.append('Arc '+str(idx+1))
        tb_arr.append(str(Arcs_list[idx]))
    tb_array.append(tb_arr)
    for k in range(UL_Iters):
        tb_arr = [k]
        tb_arr.extend(x_Iter_dict[Max_LL_Iters][k])
        tb_array.append(tb_arr)
        
    table = pd.DataFrame(tb_array, columns = colNames) 
    table.to_excel(writer, sheet_name = 'D='+str(Max_LL_Iters), index = False)
writer.save()   
