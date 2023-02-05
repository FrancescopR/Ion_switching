#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:07:02 2020

@author: francesco
"""

import MultiGaussianModel as MGM
#import Marcov_chain as MC
import Marcov_chain1 as MC1
import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = (24, 20)




path = '/home/francesco/Machine_learning/PyTorch/liverpool-ion-switching'
os.chdir(path)


data = pd.read_csv('train.csv')
 
time     = data['time'].values 
sign   = data['signal'].values

try:
    channels = data['open_channels'].values
except:
    pass


# %%
fig, ax1 = plt.subplots(1,1)

ax1.plot(time, sign, lw=0.0, marker='.', markersize=0.03)    


# %% - - - - - - - - - -  CLEAN SIGNAL - - - - - - - - 


i_blocks  = np.arange(0, len(sign) +1, 5e+5, dtype=np.int)


block =  []

for i0, i1 in zip(i_blocks[:-1], i_blocks[1:]):

    t       = time[i0:i1]
    s       = sign[i0:i1]
    try:
        c = channels[i0:i1]
    except:
        pass
    
    good_fit = MGM.check_if_sigma_is_a_good_fit(s=s, t=t)
    
    if good_fit:
        # remove dirft
        s_in = s - MGM.fit_sin(t, s)[0]
        # remove blur
        s_in = MGM.remove_blur(s=s_in)
        try:
            B = MGM.SingleBlock(signal=s_in, time=t, channels=c)
        except:
            B = MGM.SingleBlock(signal=s_in, time=t)
        
        block += [B]        
    elif not good_fit:
        indices = MGM.subsplit_a_block(s=s, t=t)

        for j0, j1 in zip(indices[:-1], indices[1:]):
            # remove dirft
            s_in = s[j0:j1] - MGM.fit_sin(t[j0:j1], s[j0:j1])[0]
            # remove blur
            s_in = MGM.remove_blur(s=s_in)
            try:
                B = MGM.SingleBlock(signal=s_in, time=t[j0:j1], channels=c[j0:j1])
            except:
                B = MGM.SingleBlock(signal=s_in, time=t[j0:j1])
            block += [B]


            
for B in block:
    plt.plot(B.t, B.s, lw=0.0, marker='.', markersize=0.2)            


   



# %%

#fs = np.full(len(block), 0.8)    
       

for i, B in enumerate(block):

    ### 
    MGM.fit_single_block_with_MGM(single_block=B, plot=True)
    
    ###
    try:
        predictions, y_true = MGM.prediction1([B])
        acc, f1, errors     = MGM.evaluation(y_true=y_true, y_pred=predictions)
        title = 'block: ' + str(i) + '  acc: ' + str(acc)[:5] + '  f1: ' + str(f1)[0:5]
        print(title)    
    except:
        pass



# %%

Myblok = block
predictions, y_true = MGM.prediction1(Myblok)


acc, f1, errors  = MGM.evaluation(y_true=y_true, y_pred=predictions)
title = 'all blocks:  acc: ' + str(acc)[:6] + '  f1: ' + str(f1)[0:6]


print(title)  



# %% Predict using MC

all_pred = np.array([])
t_out    = np.array([])

for i, B in enumerate(block[:]):

    print('HMM block', i)
    
    mus = B.MultiNormalFit[:,1]
    
    sig = B.MultiNormalFit[:,2].mean()
    
    my_model = MC1.fit_HMM_Normal(Q=B.s.reshape(-1,1), mus=mus, sig=sig)
    
    y_pred = MC1.prediction_HMM_Normal(X=B.s, real_values=B.predicted_channels, model=my_model)
    
    all_pred = np.append(all_pred, y_pred)    
    t_out    = np.append(t_out, B.t)    
    
    try:
        acc, f1, errors  = MGM.evaluation(y_true=B.c, y_pred=y_pred)
        title = 'block: ' + str(i) + '  acc: ' + str(acc)[:6] + '  f1: ' + str(f1)[0:6]
        print(title)
    except:
        pass



# %%
acc, f1, errors  = MGM.evaluation(y_true=channels, y_pred=all_pred)
title            = ' acc: ' + str(acc)[:6] + ' f1: ' + str(f1)[0:6]    
    
print(title)





# %%

sub = pd.DataFrame({'time': t_out, 'open_channels': all_pred.astype(np.int)})

sub.to_csv('sub1.csv', float_format='%0.4f', index=False)


# %%

print(np.unique(all_pred))











# %%



'''
#########################################################################################
############################ FIND AND CREATE BLOCK TYPES ################################
#########################################################################################



all_pred = np.array([])

for i, B in enumerate(block):
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
    
    acc, f1, errors  = MGM.evaluation(y_true=y_true, y_pred=y_pred)
    title = 'block:' + str(i) + ' acc: ' + str(acc)[:6] + ' f1: ' + str(f1)[0:6]
    print(title)
    print('\n\n')





block_indx, new_blocks = group_blocks(block)


for i, (B,f1) in enumerate(zip(new_blocks, fs)):

    # # # 
    fit_single_block_with_MGM(single_block=B, plot=False, f=[0.4, f1])
    
    
    predictions, y_true      = MGM.prediction1([B])
    acc, f1, errors  = MGM.evaluation(y_true=y_true, y_pred=predictions)
    title = 'block: ' + str(i) + '  acc: ' + str(acc)[:6] + '  f1: ' + str(f1)[0:6]
    print(title) 


    

d = {}
 
for i, B in enumerate(new_blocks):
    mu  = B.MultiNormalFit[:,1]
    pad = np.full(11-len(mu), 333)
    mu  = np.append(mu, pad)
    #
    key    = 'type' + str(i+1) 
    d[key] = mu

df = pd.DataFrame(d)    

df.to_hdf(path + '/block_type.h5', key='block_type')
     

for i, B in enumerate(new_blocks[4:5]):        
    #plt.plot(G.t, G.s + shifts[i], lw=0, marker='.', markersize=0.1)
    plt.plot(B.t, B.s, lw=0, marker='.', markersize=0.1)


##################################################################################################
'''
    
