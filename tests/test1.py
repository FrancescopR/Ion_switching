#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:16:08 2020

@author: francesco
"""
import Hidden_Marcov_chain as MC1
import lgbm_functions as LF
import MultiGaussianModel as MGM
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 40}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = (24, 20)



path = '/home/francesco/Machine_learning/PyTorch/liverpool-ion-switching'
os.chdir(path)

test  = pd.read_csv('train.csv')



time     = test['time'].values 
signal   = test['signal'].values



# %%


i_blocks = [int(i*1e+5) for i in range(1, 11)] + [ int(1.5e+6), int(2.0e+6) ] 


clean_signal, i0  = np.array([]), 0

for i1 in i_blocks:
    clean_signal_ = MGM.remove_drift(s=signal[i0:i1], t=time[i0:i1])
    clean_signal  = np.append(clean_signal, clean_signal_)
    i0            = i1


# %% plot test data


#fig, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(8, 8))
#ax1.plot(time, clean_signal, marker='.', lw=0, markersize=0.02, color='black')



# %%

blocks, i0  = [], 0    
for i1 in i_blocks:
    #
    blocks += [MGM.SingleBlock(signal = clean_signal[i0:i1], time = time[i0:i1] )]  
    i0 = i1    
     
    

# %%


for i, B in enumerate(blocks):
    # # # 
    MGM.fit_single_block_with_MGM(single_block=B, plot=True, f=[0.4, 0.9])
    



# %%  Predict Using Markov Chain
     
    
all_pred = np.array([])
t_out   =  np.array([])   

for i, B in enumerate(blocks):
    logQ   = B.s
    y_true = B.c
    
    
    params = {'n_components' : len(B.predicted_channels),
              'means_'       : B.MultiNormalFit[:,1],
              'covars_'      : B.MultiNormalFit[:,2].mean()**2,
              }
    
    my_model  = MC.fitHMM(Q=logQ, parameters=params)
    
    y_pred = MC.prediction_GaussianHMM(X=logQ, real_values=B.predicted_channels, 
                                    model=my_model)

    all_pred = np.append(all_pred, y_pred)    
    t_out    = np.append(t_out, B.t)    
    
    print()



# %%  Predict Using Markov Chain1
     
    
all_pred = np.array([])
t_out   =  np.array([])   

for i, B in enumerate(blocks):
    logQ   = B.s
    y_true = B.c


    mus = B.MultiNormalFit[:,1]
    sig = B.MultiNormalFit[:,2].mean()    
    
    my_model  = MC1.fit_HMM_Normal(Q=logQ, mus=mus, sig=sig)
    
    y_pred    = MC1.prediction_HMM_Normal(X=logQ, real_values=B.predicted_channels, 
                                          model=my_model)

    all_pred = np.append(all_pred, y_pred)    
    t_out    = np.append(t_out, B.t)    

    


# %%
print(all_pred.shape)    
print(t_out.shape)

# %%


for i, B in enumerate(blocks): 
    MGM.infer_block_type(B=B)    
    print(i, B.block_type)    





# %%
y_pred  =  np.array([])
t_out   =  np.array([])   
for B in blocks:
    y_pred_, _  = LF.prediction_multi_models([B])
    y_pred      = np.append(y_pred, y_pred_)
    t_out       = np.append(t_out, B.t)    
    
    
# %%    
    
sub = pd.DataFrame({'time': t_out, 'open_channels': all_pred.astype(np.int)})

sub.to_csv('sub1.csv', float_format='%0.4f', index=False)    
    
    


    