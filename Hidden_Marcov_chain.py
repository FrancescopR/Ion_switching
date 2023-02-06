#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:57:56 2020

@author: francesco
"""


import numpy as np
import pomegranate as pm


import MultiGaussianModel as MGM
import os
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = (24, 20)




def initialize_the_model(mus=None, sig=None, dim=1):

    dim = int(dim * len(mus))      
    
    tr_shape  = (dim, dim)
    trans_mat = np.full(tr_shape, 0.33)
    
    starts = np.full(dim, 0.33)
    
    ends   = np.full(dim, 0.33)
    
    dists  = [pm.NormalDistribution(mu, sig) for mu in mus]
    
    model  = pm.HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)

    return model


def initialize_the_model1(mus=None, sig=None, dim=1):

    dim = int(dim * len(mus))      
    
    tr_shape  = (dim, dim)
    trans_mat = np.full(tr_shape, 0.33)
    
    starts = np.full(dim, 0.33)
    
    ends   = np.full(dim, 0.33)
    
    dists  = [pm.NormalDistribution(mu, sig) for mu in mus]
    
    model  = pm.HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)

    return model





def fit_HMM_Normal(Q=None, mus=None, sig=None, dim=1):
    
    #
    model = initialize_the_model(mus=mus, sig=sig, dim=dim)
    
    #
    model.fit([Q], algorithm='viterbi', 
                     distribution_inertia=1.0, 
                     min_iterations=0, 
                     max_iterations=120,
                     verbose=False)
    
    # to imporve the position
    model.fit([Q], algorithm='viterbi', 
                     distribution_inertia=0.997,
                     edge_inertia=0.95,
                     min_iterations=2, 
                     max_iterations=60,
                     verbose=False)

    
    return model



def prediction_HMM_Normal(X=None, real_values=None, model=None):
    
    y_pred_ = model.predict(X.reshape(-1,1), algorithm='viterbi')
         
    #d = decoder(real_values=real_values, model=model)
    delta = max(real_values) - max(y_pred_[2:-2])
           
    y_pred  = y_pred_[1:-1] + delta
    return y_pred




# %%




path = '/home/francesco/Machine_learning/PyTorch/liverpool-ion-switching'
os.chdir(path)


data = pd.read_csv('train.csv')
 
time     = data['time'].values 
signal   = data['signal'].values

try:
    channels = data['open_channels'].values
except:
    pass




i_blocks  = np.arange(0, len(signal) +1, 5e+5, dtype=np.int)


block =  []

for i0, i1 in zip(i_blocks[:-1], i_blocks[1:]):

    t       = time[i0:i1]
    s       = signal[i0:i1]
    try:
        c = channels[i0:i1]
    except:
        pass
    
    good_fit = MGM.check_if_sigma_is_a_good_fit(s=s, t=t)
    
    if good_fit:
        s = s - MGM.fit_sin(t, s)[0] 
        try:
            B = MGM.SingleBlock(signal=s, time=t, channels=c)
        except:
            B = MGM.SingleBlock(signal=s, time=t)
        
        block += [B]        
    elif not good_fit:
        indices = MGM.giulia(s=s, t=t)

        for j0, j1 in zip(indices[:-1], indices[1:]):
            s[j0:j1] = s[j0:j1] - MGM.fit_sin(t[j0:j1], s[j0:j1])[0]
            try:
                B = MGM.SingleBlock(signal=s[j0:j1], time=t[j0:j1], channels=c[j0:j1])
            except:
                B = MGM.SingleBlock(signal=s[j0:j1], time=t[j0:j1])
            block += [B]





fs = [1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 1.0, 1.0, 1.0, 1.0, 0.7]    
       


for i, (B,f1) in enumerate(zip(block, fs)):

    # # # 
    MGM.fit_single_block_with_MGM(single_block=B, plot=False, f=[0.4, f1])
    
    ###
    try:
        predictions, y_true      = MGM.prediction1([B])
        acc, f1, errors  = MGM.evaluation(y_true=y_true, y_pred=predictions)
        title = 'block: ' + str(i) + '  acc: ' + str(acc)[:5] + '  f1: ' + str(f1)[0:5]
        print(title)    
    except:
        pass





        



def discretize_signal(s=None, mus=None):
    sep = np.diff(mus).mean()
    
    subdiv     = np.arange(np.min(mus) - 0.5 * sep, np.max(mus) + 0.5 * sep, 0.33 * sep)
    subdiv[0]  = -333.
    subdiv[-1] =  333.
    
    
    s_discretized = np.full(len(s), 7777)
    for i, (sub0, sub1) in enumerate(zip(subdiv[:-1], subdiv[1:])):
        cond = (sub0 <= s ) & (s < sub1)
        s_discretized = np.where(cond, i*100, s_discretized)
    return s_discretized 







# %%

B = block[3]
mus = B.MultiNormalFit[:,1]

s_discretized = discretize_signal(s=B.s, mus=mus)


plt.scatter(B.t, s_discretized)    
    
    



# %%
     
dim =  len(mus)


tr_shape  = (dim, dim)
trans_mat = np.full(tr_shape, 0.33)

starts = np.full(dim, 0.33)

ends   = np.full(dim, 0.33)


dists = []
for _ in mus:
    dists  += [pm.DiscreteDistribution({k: 0.1 for k in np.unique(s_discretized)})]

model  = pm.HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)

# %%
print(model)


# %%
# TO DO
# train with labels
# remove noise from model8 and model4
# better ways to find the peaks


labels = ['None-start', 'a', 'b', 'b', 'a', 'None-end']

model.fit([s_discretized], 
                  #algorithm='viterbi',
                  #distribution_inertia=0.0,
                  #edge_inertia=1.0,
                  algorithm='labels',
                  labels=B.c,
                  min_iterations=0, 
                  max_iterations=50,
                  verbose=True)


#print(improvement)


# %%
y_pred = model.predict(s_discretized)

a, f, err = MGM.evaluation(y_pred=y_pred, y_true=B.c)

# %%

print(a, f)


# %%
print(y_pred)


n_sub = 2

lim  = np.arange(mus.min(), mus.max()+0.05, sep/n_sub)

lim_i = np.array([-3333])
lim_f = np.array([3333])

lim = np.concatenate((lim_i, lim, lim_f))


quntized_signal = np.zeros()

for a, b in zip(lim[:-1], lim[1:]):
    cond = (a < B.s) & (B.s < b)
    np.where(cond)



model = pm.HiddenMarkovModel()

s0    = pm.State( pm.NormalDistribution( mus[0], sig )) 

s1    = pm.State( pm.NormalDistribution( mus[1], sig )) 

s00   = pm.State( [pm.NormalDistribution( mus[0], sig ), 
                   pm.NormalDistribution( mus[0], sig )] )


s11   = pm.State( [pm.NormalDistribution( mus[1], sig ), 
                   pm.NormalDistribution( mus[1], sig )] ) 


states = [s00, s11]

model.add_states( states )




for state_i in states:
    for state_j in states + [model.end, model.start]:
        model.add_transition( state_i, state_j, 0.33 )

model.bake()
'''