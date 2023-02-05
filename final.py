#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:56:12 2020

@author: francesco
"""
import pandas as pd
import pomegranate as pm
from lightgbm import LGBMClassifier
import h5py
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import joblib
from pathlib import Path
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

import matplotlib.pyplot as plt
import matplotlib


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = (24, 20)


   
# README ---> Before Start Coding Go to cell 2 and read carefully the instructions
# README ---> Please read the line above
# README ---> Please read the line above




def same_type(mu1=None, mu2=None):
    #
    if (len(mu1) > 9) and (len(mu2) > 9):
        return np.allclose(mu1[-2:], mu2[-2:], atol=1.2e-1)
    
    elif len(mu1)!=len(mu2):
        return False
    
    else:
        return np.allclose(mu1[:2], mu2[:2], atol=1.2e-1)



def infer_block_type(mus=None):
    file_path = Path(__file__).resolve().with_name('block_type.h5')
    df        = pd.read_hdf(file_path, key='block_type')
        
    block_type = 'unclassified'

    for Type in df:
        s   = df[Type]
        mu_ = s.values
        mu_ = mu_[mu_<300]
                   
        if same_type(mu1=mu_, mu2=mus):
            block_type = Type
    return block_type



def find_open_channels(mus=None):
    
    if len(mus) > 8:
        c_min = 11 - len(mus)
        c_max = 10
        open_channels = np.arange(c_min, c_max + 1, 1)
    else:
        c_min = 0
        c_max = len(mus)
        open_channels = np.arange(c_min, c_max , 1)        

    return open_channels



class Block(object):
    def __init__(self, b=None):
        self.s = b['signal'][()]
        self.t = b['time'][()]
        #
        self.mu_HMM  = b['hmm/mu'][()]
        self.sig_HMM = b['hmm/sig'][()]
        #
        self.mu_MGM  = b['mgm/mu'][()]
        self.sig_MGM = b['mgm/sig'][()]
        self.A_MGM   = b['mgm/A'][()]
        #
        self.open_channels = find_open_channels(mus=self.mu_HMM)
        self.block_type    = infer_block_type(mus=self.mu_HMM)
        #
        self.best_inertia()
        
        try:
            self.c = b['channels'][()]
            #
            self.f1_HMM = b['hmm/f1'][()]
            self.acc_HMM = b['hmm/acc'][()]
            #
            self.f1_MGM = b['mgm/f1'][()]
            self.acc_MGM = b['mgm/acc'][()]
        except:
            pass
    
    def best_inertia(self):
        
        if self.block_type=='type4':
            self.distribution_inertia = 0.8 
            self.edge_inertia         = 0.8
        elif self.block_type=='type5':    
            self.distribution_inertia = 0.0 
            self.edge_inertia         = 0.0
        else:    
            self.distribution_inertia = 1.0 
            self.edge_inertia         = 1.0    




class LoadData(object):
    def __init__(self, key=None):
        file_path = Path(__file__).resolve().with_name('liverpool_ion_switching.hdf5')
        f = h5py.File(file_path, 'r')

        self.block = {}
        blocks = list(f[key])

        for block in blocks:
            i = int(block.replace('block', ''))
            self.block[i] = Block(b=f[key][block])

        f.close()


def evaluation(y_pred=None, y_true=None):
    #
    f1_1   = f1_score(y_true, y_pred, average='macro')    
    errors = np.where(y_pred!=y_true)[0]      
    acc1   = accuracy_score(y_true, y_pred)     
    return acc1, f1_1, errors


def save_model(model=None, path='.', model_name=None):
    joblib.dump(model, path + '/lgbm_' + model_name + '.pkl')


def padding(kernel_size=None, signal_in=None):
    
    # kernel must be odd
    if kernel_size % 2 == 0.0:
        kernel_size += 1
    
    pad = np.full(kernel_size//2, 10000)
    X = np.concatenate((pad, signal_in, pad), axis=0)
    return np.array([X[i:i + kernel_size] for i in range(len(X)-kernel_size+1)])



def make_prediction(signal_in=None, model=None):
    # get kernel size i.e. number of features
    kernel_size = model.n_features_
    # reshape input
    model_input = padding(kernel_size=kernel_size, signal_in=signal_in)
    #
    return model.predict(model_input )


def initialize_model(params=None):
    
    global choose_model
    
    num_leaves    =  params['num_leaves']
    learning_rate =  params['learning_rate']
    n_estimators  =  params['n_estimators']    
    #min_child_weight =  params['min_child_weight']
    
    if choose_model=='LGBM':
        model = LGBMClassifier(boosting_type='gbdt', num_leaves=num_leaves, max_depth=-1,
                           learning_rate=learning_rate,  n_estimators=n_estimators, n_jobs=-1,
                           subsample_for_bin=200000, objective=None,  class_weight=None,
                           min_split_gain=0.0, min_child_weight=0.01, min_child_samples=20,
                           subsample=1.0, subsample_freq=0, colsample_bytree=1.0,
                           reg_alpha=0.0, reg_lambda=0.0, random_state=None,
                           silent=True, Aprimportance_type='split')

    elif choose_model=='GBR':
        model = GradientBoostingClassifier()    
    elif choose_model=='RF':
        model = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', max_depth=None, 
                                 min_samples_split=2, min_samples_leaf=1, bootstrap=True,
                                 min_weight_fraction_leaf=0.0, max_features='auto',
                                 max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                 min_impurity_split=None, oob_score=False, n_jobs=None, 
                                 random_state=None, verbose=0, warm_start=False, 
                                 class_weight=None, ccp_alpha=0.0, max_samples=None)
    else:
        raise Exception('Model Selected does not exist:', choose_model)
        
    return model


def fit_model(model_input=None, labels_in=None, params=None):
    model = initialize_model(params=params)
    model.fit(model_input, labels_in)
    return model


def function_to_optimize(kernel_size=3, learning_rate=0.4): #, min_child_weight=0.01):    
    kernel_size = np.int(kernel_size)
    
    parameters = {'kernel_size'   : kernel_size, 
                  'num_leaves'    : 21, 
                  'learning_rate' : learning_rate, 
                  'n_estimators'  : 110}
                   #'min_child_weight' :  min_child_weight}
    
    global signal_input, signal_eval 
    global labels_input, labels_eval 
    
    model_input  = padding(kernel_size=kernel_size, signal_in=signal_input)
    model        = fit_model(model_input=model_input, labels_in=labels_input, params=parameters)
    
    y_pred = make_prediction(signal_in=signal_eval, model=model)
    acc, f1, _ = evaluation(y_pred=y_pred, y_true=labels_eval)    
    return f1

TR = LoadData(key='train')



# %%


# README ---> Please Read carefully all the info contained in this cell!!!
    

# 1) Select the block you want to use for fitting 
fitting_block = TR.block[5]



# 2) select the block you want to use for evaluation 
#     - make sure that evaluat_block is of the same type of fitting_block.
#     - for example 5 and 10 or 6 and 9 are of the same type as well as (0,1,2) and (3, 7) and (4, 8) 
#     - to ckeck if two blocks are of the same type check the positions of the peaks, if they
#       have the peaks in the same positions they are of the same type.
#     - print(TR.block[i].mu_HMM) to check the peaks position of one block.
evaluat_block = TR.block[10]


# 3) select N the Size of the signal to fit the model
#     - N=-1  --> for the entire signal
#     - N < 5000 --> reccomanded for GRB method (too slow for higher N)
N = 100      


# 4) select the model 
#   - OPTIONS: 'LGBM', 'RF', 'GBR'
choose_model='RF' 


# 5) Select  Bounded region of parameter space and number of iterations
#  - IMPORTANT: learning rate cannot  be zero    
pbounds     = {'kernel_size': (1, 20), 'learning_rate': (0.01, 1)}#, 'min_child_weight': (0.0001, 1)}
n_iter      = 5
init_points = 2

# IMPORTANT ---> If you want to add a new parameter for the ottimization follow the istruction 
#                below:
#          a) in the function 'initialize_model' add the parameter as follow
#                min_child_weight =  params['min_child_weight']  (this is just an example)
#          b) add the same parameter in the args of function 'function_to_optimize': 
#                function_to_optimize(kernel_size=3, learning_rate=0.4, min_child_weigh=0.01) 
#          c) in the same function add 'min_child_weight' : min_child_weight in the dictionary 
#             parameters.
#          d) finally add the new parameter in pbounds. 

signal_input =  fitting_block.s[:N]
labels_input =  fitting_block.c[:N]

signal_eval  = evaluat_block.s
labels_eval   = evaluat_block.c


optimizer = BayesianOptimization(f=function_to_optimize, pbounds=pbounds, 
                                  random_state=1, verbose=2)


out_path  = Path(__file__).resolve().with_name('logs.json')
logger    = JSONLogger(path=str(out_path))
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(n_iter=n_iter, init_points=init_points, )
print(optimizer.max)



# %%

def plot_check(ax=None, s=None, t=None, open_channels=None, mus=None, sigma=None):
    
    t1 = np.mean( [min(t), t.mean()] )
    t0 = np.mean( [min(t), t1] )
    
    ax.plot(s, t, marker='o', lw=0, markersize=0.3, color='black')
    
    for prdct, mu in zip(open_channels, mus):
        txt   = str(prdct)
        sigmas = np.linspace(mu - sigma, mu + sigma, 30)
        #
        ax.plot(sigmas, [t0] * 30, lw=15, color='blue', alpha=0.5)
        ax.text(mu, t1, txt, size=65, ha="center", color='blue', fontweight='bold')
    
    return ax




def initialize_the_model(mus=None, sigma=None):
 
    dim = len(mus)    
    tr_shape  = (dim, dim)
    trans_mat = np.full(tr_shape, 0.33)
    
    starts = np.full(dim, 0.33)
    ends   = np.full(dim, 0.33)

    dists  = [pm.NormalDistribution(mu, sigma) for mu in mus]
    model  = pm.HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)

    return model


def fit_HMM_Normal(distribution_inertia=0.9, edge_inertia=0.9):
    
    
    global signal_input
    global mus
    global sigma
    
    #
    model = initialize_the_model(mus=mus, sigma=sigma)
    #
    model.fit([signal_input], algorithm='viterbi', 
                              distribution_inertia=1.0, 
                              min_iterations=0, 
                              max_iterations=120, 
                              verbose=False)
    #
    model.fit([signal_input], algorithm='viterbi', 
                             distribution_inertia = distribution_inertia, 
                             edge_inertia         = edge_inertia,  
                             min_iterations       = 0,  
                             max_iterations       = 340, 
                             verbose=False)
    
    return model





def prediction_HMM_Normal(model=None):
    
    global signal_input
    global mus
    global sigma
    global open_channels
    
    
    y_pred_       = model.predict(signal_input.reshape(-1,1), algorithm='viterbi')
     
    #
    delta = max(open_channels) - max(y_pred_[2:-2])
           
    y_pred  = y_pred_[1:-1] + delta
    return y_pred





def smooth(x, fraction=0.01, window='hanning'):
    """smooth the data using a window with requested size.
    
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    
    window_len = int(fraction * len(x))   
    
    if fraction >=1.0:
        window_len = int(fraction)    

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]





# %%


TR = LoadData(key='train')

                       
# %%


all_pred = np.array([])
all_c    = np.array([]) 
t_out    = np.array([]) 


for k, B in TR.block.items():
    
    #if k==5:
    #  B.mu_HMM[0] = -7.1
    #  B.mu_HMM[1] = -6.2
      
    
    Mult = 1.0
      
    signal_input  = Mult * B.s #smooth(B.s, fraction=3)
    mus           = Mult * B.mu_HMM
    sigma         = Mult * B.sig_HMM.mean()
    open_channels = B.open_channels
    
    fig, ax = plt.subplots(1,1)

    #    
    my_model      = fit_HMM_Normal(
                                   distribution_inertia = B.edge_inertia, 
                                   edge_inertia         = B.distribution_inertia
                                   )
    
    
    y_pred       = prediction_HMM_Normal(model=my_model)
    all_pred     = np.append(all_pred, y_pred)    
    #t_out        = np.append(t_out, B.t)      

    #try:
    acc_, f1, _  = evaluation(y_pred=y_pred, y_true=B.c)
    all_c    = np.append(all_c, B.c)
    print(B.block_type, f1, B.f1_HMM, B.f1_MGM)
    #except:
    #    print(B.block_type)
    
    plot_check(ax=ax, s=signal_input, t=B.t, open_channels=open_channels, mus=mus, sigma=sigma)
    plt.title(B.block_type)





# %%
        
sub = pd.DataFrame({'time': t_out, 'open_channels': all_pred.astype(np.int)})

sub.to_csv('sub1.csv', float_format='%0.4f', index=False)        



# %%

'''

step1 = 0.9
b1    = [0.0, 1.0]                   
b2    = [0.0, 1.0]

p1_max, p2_max, score = brute_force_grid(step=step1, b1=b1, b2=b2)


step2 = 0.5 * step1
b1    = [p1_max - step1, p1_max + step2]                   
b2    = [p2_max - step1, p2_max + step2]


p1_final, p2_final, score_final = brute_force_grid(step=step2, b1=b1, b2=b2)


print(score_final, score)

# %%

# for block 5   0.8 0.8 0.796036001137852

pbounds     = {'distribution_inertia': (0.7, 0.9), 'edge_inertia': (0.7, 0.9)}
n_iter      = 10
init_points = 1


optimizer = BayesianOptimization(f=function_to_optimize_HMM, pbounds=pbounds, 
                                  random_state=1, verbose=2)

#out_path  = Path(__file__).resolve().with_name('block_5_HMM_logs.json')
#logger    = JSONLogger(path=str(out_path))
#optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)


optimizer.maximize(n_iter=n_iter, init_points=init_points)
print(optimizer.max)



# %%



def function_to_optimize_HMM(distribution_inertia=0.9, edge_inertia=0.9):
    
    model        = fit_HMM_Normal(distribution_inertia = distribution_inertia, 
                                  edge_inertia         = edge_inertia)
    
    y_pred       = prediction_HMM_Normal(model=model)
    acc_, f1, _  = evaluation(y_pred=y_pred, y_true=B.c)

    
    return f1



def brute_force_grid(step=0.5, b1=[0.0, 1.0], b2=[0.0, 1.0]):
    
    range_param1 =  np.arange(b1[0], b1[1] + 1e-5, step)
    range_param2 =  np.arange(b2[0], b2[1] + 1e-5, step)                       

    
    par1, par2, score =  [], [], [] 
    
    for p1 in range_param1:
        for p2 in range_param2:
            print(p1, p2)
            f1    = function_to_optimize_HMM(distribution_inertia=p1, edge_inertia=p2)
            par1  += [p1]
            par2  += [p2] 
            score += [score] 
            print(p1, p2, f1)
    
    par1  = np.array(par1)
    par2  = np.array(par2)
    score = np.array(score)    

    i_max = np.argmax(score)

    return par1[i_max], par2[i_max], score[i_max]




'''

