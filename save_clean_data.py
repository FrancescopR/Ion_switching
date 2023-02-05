#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:54:27 2020

@author: francesco
"""

import Hidden_Marcov_chain as MC1
import MultiGaussianModel as MGM
import pandas as pd
import h5py
import os

import matplotlib.pyplot as plt
import matplotlib


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = (24, 20)

# %% ##############################################################################################
#                       TRAINING DATA
###################################################################################################

path = '/home/francesco/Machine_learning/PyTorch/liverpool-ion-switching'
os.chdir(path)


data = pd.read_csv('train.csv')
 
time     = data['time'].values 
sign     = data['signal'].values
channels = data['open_channels'].values


# %% BLOCK 1:

f = h5py.File('liverpool_ion_switching.hdf5','w')

grp_train = f.create_group("train")
grp_test  = f.create_group("test")

# TRAINING

data = pd.read_csv('train.csv')
 
time     = data['time'].values 
sign     = data['signal'].values
channels = data['open_channels'].values


i_blocks = [0, int(5e+5), int(6e+5)] + [int(i*5e+5) for i in range(2, 11)]

for n, (i0, i1) in enumerate(zip(i_blocks[:-1], i_blocks[1:])):
    
    t       = time[i0:i1]
    s       = sign[i0:i1]
    c       = channels[i0:i1]
    
    s_in    = s - MGM.fit_sin(t, s)[0]
    
    
    B = MGM.SingleBlock(signal=s_in, time=t, channels=c)
    
    MGM.fit_single_block_with_MGM(single_block=B, plot=True)
        
    ###
    predictions, y_true = MGM.prediction1([B])
    acc_MGM, f1_MGM, _    = MGM.evaluation(y_true=y_true, y_pred=predictions)
        
    
    
    Type     = MGM.infer_block_type(B=B)
    A_MGM    = B.MultiNormalFit[:,0]
    mus_MGM  = B.MultiNormalFit[:,1]
    sig_MGM  = B.MultiNormalFit[:,2]
    sig      = sig_MGM.mean()
    
    
    
    my_model = MC1.fit_HMM_Normal(Q=B.s.reshape(-1,1), mus=mus_MGM, sig=sig)
    y_pred   = MC1.prediction_HMM_Normal(X=B.s, real_values=B.predicted_channels, model=my_model)
    
    
    
    mus_HMM, sig_HMM = [], []
    for state in  my_model.states:
        try:
            mus_HMM += [state.distribution.parameters[0]]
            sig_HMM += [state.distribution.parameters[1]]
        except:
            pass
    
    
    
    acc_HMM, f1_HMM, _ = MGM.evaluation(y_true=B.c, y_pred=y_pred)
    text1 = 'HMM:  acc: ' + str(acc_HMM)[:6] + '  f1: ' + str(f1_HMM)[0:6]
    text2 = 'MGM:  acc: ' + str(acc_MGM)[:6] + '  f1: ' + str(f1_MGM)[0:6]
    
    print(text1)
    print(text2)
    
    print(mus_HMM)
    print(mus_MGM)
    
    
    
    grp = grp_train.create_group("block" + str(n) )
    grp.create_dataset("signal", data=B.s)
    grp.create_dataset("time", data=B.t)
    grp.create_dataset("channels", data=B.c)
    grp.create_dataset(Type, data=[0])
    
    #
    sgrp_hmm    = grp.create_group("hmm")
    sgrp_hmm.create_dataset("mu",  data=mus_HMM)
    sgrp_hmm.create_dataset("sig", data=sig_HMM)
    sgrp_hmm.create_dataset("acc",  data=[acc_HMM])
    sgrp_hmm.create_dataset("f1",  data=[f1_HMM])
    
    # 
    sgrp_mgm    = grp.create_group("mgm" )
    sgrp_mgm.create_dataset("A",  data=A_MGM)
    sgrp_mgm.create_dataset("mu",  data=mus_MGM)
    sgrp_mgm.create_dataset("sig", data=sig_MGM)
    sgrp_mgm.create_dataset("acc",  data=[acc_MGM])
    sgrp_mgm.create_dataset("f1",  data=[f1_MGM])

#  TESTING

data = pd.read_csv('test.csv')

 
time     = data['time'].values 
sign     = data['signal'].values


i_blocks = [int(i*1e+5) for i in range(0, 10)]     + [int(i*5e+5) for i in range(2, 5)]    
    
for n, (i0, i1) in enumerate(zip(i_blocks[:-1], i_blocks[1:])):
    
    t       = time[i0:i1]
    s       = sign[i0:i1]
    
    s_in    = s - MGM.fit_sin(t, s)[0]
    
    
    B = MGM.SingleBlock(signal=s_in, time=t, channels=c)
    
    MGM.fit_single_block_with_MGM(single_block=B, plot=True)
        
     
    
    Type     = MGM.infer_block_type(B=B)
    A_MGM    = B.MultiNormalFit[:,0]
    mus_MGM  = B.MultiNormalFit[:,1]
    sig_MGM  = B.MultiNormalFit[:,2]
    sig      = sig_MGM.mean()
    
    
    
    my_model = MC1.fit_HMM_Normal(Q=B.s.reshape(-1,1), mus=mus_MGM, sig=sig)
    
    
    
    mus_HMM, sig_HMM = [], []
    for state in  my_model.states:
        try:
            mus_HMM += [state.distribution.parameters[0]]
            sig_HMM += [state.distribution.parameters[1]]
        except:
            pass
    
    print(mus_HMM)
    print(mus_MGM)
    print()
    
    
    grp = grp_test.create_group("block" + str(n) )
    grp.create_dataset("signal", data=B.s)
    grp.create_dataset("time", data=B.t)
    #grp.create_dataset("channels", data=B.s)
    grp.create_dataset(Type, data=[0])
    
    #
    sgrp_hmm    = grp.create_group("hmm")
    sgrp_hmm.create_dataset("mu",  data=mus_HMM)
    sgrp_hmm.create_dataset("sig", data=sig_HMM)
    #sgrp_hmm.create_dataset("acc",  data=[acc_HMM])
    #sgrp_hmm.create_dataset("f1",  data=[f1_HMM])
    
    # 
    sgrp_mgm    = grp.create_group("mgm" )
    sgrp_mgm.create_dataset("A",  data=A_MGM)
    sgrp_mgm.create_dataset("mu",  data=mus_MGM)
    sgrp_mgm.create_dataset("sig", data=sig_MGM)
    #sgrp_mgm.create_dataset("acc",  data=[acc_MGM])
    #sgrp_mgm.create_dataset("f1",  data=[f1_MGM])


f.close()

# %%


# %%





# %%





# %%
bs1 = pd.DataFrame({'block1':b1, 
                    'block2':b2,
                    'block3':b3, 
                    'block4':b4
                    
                   
                   })
# %%
bs2 = pd.DataFrame({'block1':[0], 
                   'block2':[0]
                   })

# %%
df = pd.DataFrame({'train':[bs1], 
                   'test':[bs2]
                   })

# %%
print(df.train)