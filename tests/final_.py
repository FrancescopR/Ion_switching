from bayes_opt import BayesianOptimization
import os
import time
import sys
import numpy as np
import pomegranate as pm
#import MultiGaussianModel as MGM
import pandas as pd
import sklearn

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib

from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import scipy
from scipy import integrate
from scipy.signal import find_peaks, gaussian, wiener, symiirorder1, symiirorder2
from scipy.fftpack import fftshift


from lightgbm import LGBMClassifier
'''
from filterpy.kalman import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import EnsembleKalmanFilter
from filterpy.kalman import FixedLagSmoother
from filterpy.kalman import SquareRootKalmanFilter
from filterpy.kalman import InformationFilter
from filterpy.kalman import FadingKalmanFilter
from pykalman import KalmanFilter, UnscentedKalmanFilter
'''
from tqdm import tqdm
import time
import h5py


# %%


class Block(object):
    def __init__(self, b=None):
        self.s = b['signal'][()]
        self.t = b['time'][()]
        #
        self.mu_HMM = b['hmm/mu'][()]
        self.sig_HMM = b['hmm/sig'][()]
        #
        self.mu_MGM = b['mgm/mu'][()]
        self.sig_MGM = b['mgm/sig'][()]
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


class LoadData(object):
    def __init__(self, key=None):
        path = '/home/francesco/Machine_learning/PyTorch/liverpool-ion-switching'
        f = h5py.File(path + '/liverpool_ion_switching.hdf5', 'r')

        self.block = {}
        blocks = list(f[key])

        for block in blocks:
            i = int(block.replace('block', ''))
            self.block[i] = Block(b=f[key][block])

        f.close()


# %%
TR = LoadData(key='train')

# %%

B5 = TR.block[5]
B10 = TR.block[5]

# %%
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Prediction with Machine Learning
# Load channel 5
signali = B5.s  # block_signals[5]
# len(signali)
channeli = B5.c  # block_channel[5]
# len(channeli)

######################################### LGBM MY  VERSION ##############################





def evaluation(y_pred=None, y_true=None):
    #
    f1_1 = f1_score(y_true, y_pred, average='macro')
    errors = np.where(y_pred != y_true)[0]

    #
    acc1 = accuracy_score(y_true, y_pred)
    return acc1, f1_1, errors


def lgb_make_prediction(s=None, model_name=None):
    # load model
    lgbmodel = joblib.load(path + '/lgbm_' + model_name + '.pkl')

    # get kernel size i.e. number of features
    n_features = lgbmodel.n_features_

    # reshape model
    pad = np.full(n_features//2, 10000)
    X = np.concatenate((pad, s, pad), axis=0)
    lgtm_input = np.array([X[i:i+n_features] for i in range(len(X)-n_features+1)])

    #
    return lgbmodel.predict(lgtm_input)


def lgb_make_prediction1(s=None, lgbmodel=None):
    # get kernel size i.e. number of features
    n_features = lgbmodel.n_features_

    # reshape model
    pad = np.full(n_features//2, 10000)
    X = np.concatenate((pad, s, pad), axis=0)
    lgtm_input = np.array([X[i:i+n_features] for i in range(len(X)-n_features+1)])

    #
    return lgbmodel.predict(lgtm_input)


# %%
parameters = {'kernel_size': 15,
              'num_leaves': 21,
              'learning_rate': 0.01,
              'n_estimators': 110}

lgbms = fit_and_save_lgbm_model(signal_block=B5.s, labels_block=B5.c,
                                params=parameters, model_name=False)

# %%

y_pred = lgb_make_prediction1(s=B10.s, lgbmodel=lgbms)

# %%
print(y_pred)

acc, f1, _ = evaluation(y_pred=y_pred, y_true=B10.c)

print(acc, f1)


# %%


##############################################################################################

# ************************************************************* LGBM
#+++++++++++++++++++++++++++ Windows-wise
kernel_size = 11
learning_rate = 0.1
num_leaves = 100
n_estimators = 100


def Cecilia_LGBM(kernel_size, learning_rate, num_leaves, n_estimators):
    num_leaves = int(num_leaves)
    n_estimators = int(n_estimators)
    kernel_size = int(kernel_size)

    global signali
    global channeli

    lgbms_local = LGBMClassifier(boosting_type='gbdt', num_leaves=num_leaves,
                                 max_depth=-1, learning_rate=learning_rate,
                                 n_estimators=n_estimators,
                                 subsample_for_bin=200000, objective=None,
                                 class_weight=None, min_split_gain=0.0,
                                 min_child_weight=0.001, min_child_samples=20,
                                 subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, importance_type='split')

    # Start Padding
    signal_block = signali
    labels_block = channeli

    X = signal_block
    X.shape
    Y = labels_block
    Y.shape
    for _ in range(kernel_size//2):
        X = np.insert(X, 0, 0, axis=0)
        X = np.append(X, [0], axis=0)
    # Stop of Padding

    # Kernel Snapshots of the data
    input_data = []
    for item in range(len(X)-(kernel_size-1)):
        new_world = X[item:item+kernel_size]
        input_data.append([new_world])
        new_world.shape
    # Stop Kernel Snapshots of the data

    lgtm_label = np.array(labels_block)
    lgtm_label.shape
    lgtm_input = np.array(input_data)
    lgtm_input.shape
    a, _, _ = lgtm_input.shape

    lgtm_input.shape = (a, kernel_size)
    #print("lgtm_input ",lgtm_input.shape)
    #print("lgtm_label ",lgtm_label.shape)
    lgbms_local.fit(lgtm_input, lgtm_label)
    prediction = lgbms_local.predict(lgtm_input)

    all_predictions = prediction
    all_labels = lgtm_label

    # prediction
    # lgtm_label
    f1_result = f1_score(prediction, lgtm_label, average='macro')
    #print("f1_result of block ",block," is ",f1_result)

    assert len(all_predictions) == len(all_labels)
    f1_final = f1_score(all_predictions, all_labels, average='macro')
    return f1_final


f1_final = Cecilia_LGBM(kernel_size, learning_rate, num_leaves, n_estimators)
print("f1_final part LGBM: ", f1_final)

#+++++++++++++++++++++++++++ Complete
lgbms_local = LGBMClassifier(boosting_type='gbdt', num_leaves=num_leaves, max_depth=-1,
                             learning_rate=learning_rate, n_estimators=n_estimators,
                             subsample_for_bin=200000, objective=None,
                             class_weight=None, min_split_gain=0.0,
                             min_child_weight=0.001, min_child_samples=20,
                             subsample=1.0, subsample_freq=0, colsample_bytree=1.0,
                             reg_alpha=0.0, reg_lambda=0.0, random_state=None,
                             n_jobs=-1, silent=True,
                             importance_type='split')
lgbms_local.fit(signali, np.array(channeli))
prediction = lgbms_local.predict(signali)
f1_result = f1_score(prediction, np.array(channeli), average='macro')
print("The f1_result LGBM:", f1_result)


# ************************************************************* GBR
#+++++++++++++++++++++++++++ Windows-wise
# %%

kernel_size = 11
learning_rate = 0.1


signal_block = signali
labels_block = channeli


gbr = GradientBoostingClassifier()

Y = labels_block
X = signal_block
pad = np.full(kernel_size//2, 10000)
X = np.concatenate((pad, X, pad), axis=0)

#
input_data = np.array([X[i:i+kernel_size] for i in range(len(X)-kernel_size+1)])


# %%


def Cecilia_GBR(kernel_size, learning_rate):
    kernel_size = int(kernel_size)

    # Call data gobali
    global signali
    global channeli
    signal_block = signali
    labels_block = channeli

    #num_leaves   = int(num_leaves)
    #n_estimators = int(n_estimators)
    kernel_size = int(kernel_size)

    gbr = GradientBoostingClassifier()

    Y = labels_block

    # Add padding
    X = signal_block
    pad = np.full(kernel_size//2, 10000)
    X = np.concatenate((pad, X, pad), axis=0)

    #
    input_data = np.array([X[i:i+kernel_size] for i in range(len(X)-kernel_size+1)])

    gbr.fit(input_data, )
    prediction = gbr.predict(input_data)

    all_predictions = prediction
    all_labels = Y

    # prediction
    # lgtm_label
    f1_result = f1_score(prediction, lgtm_label, average='macro')
    #print("f1_result of block ",block," is ",f1_result)

    assert len(all_predictions) == len(all_labels)
    f1_final = f1_score(all_predictions, all_labels, average='macro')
    return f1_final

# %%


Cecilia_GBR(kernel_size, learning_rate)

# %%

# Bounded region of parameter space
pbounds = {'kernel_size': (1, 20), 'learning_rate': (0, 1)}

optimizer = BayesianOptimization(
    f=Cecilia_GBR,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=3,
)
print(optimizer.max)

# ~~~~~~~~~
# %%

f1_final = Cecilia_GBR(kernel_size, learning_rate)
print("f1_final part GBR: ", f1_final)
# %%
#+++++++++++++++++++++++++++ Complete


# ************************************************************* RF
#+++++++++++++++++++++++++++ Windows-wise
kernel_size = 11
learning_rate = 0.1


def Cecilia_rf(kernel_size, learning_rate):

    global signal_block
    global labels_block

    #num_leaves   = int(num_leaves)
    #n_estimators = int(n_estimators)
    kernel_size = int(kernel_size)

    gbr = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                                 min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

    # Start Padding
    X = signal_block
    X.shape
    Y = labels_block
    Y.shape
    for _ in range(kernel_size//2):
        X = np.insert(X, 0, 0, axis=0)
        X = np.append(X, [0], axis=0)
    # Stop of Padding

    # Kernel Snapshots of the data
    input_data = []
    for item in range(len(X)-(kernel_size-1)):
        new_world = X[item:item+kernel_size]
        input_data.append([new_world])
        new_world.shape
    # Stop Kernel Snapshots of the data

    lgtm_label = np.array(labels_block)
    lgtm_label.shape
    lgtm_input = np.array(input_data)
    lgtm_input.shape
    a, _, _ = lgtm_input.shape

    lgtm_input.shape = (a, kernel_size)
    print("lgtm_input ", lgtm_input.shape)
    print("lgtm_label ", lgtm_label.shape)
    gbr.fit(lgtm_input, lgtm_label)
    prediction = gbr.predict(lgtm_input)

    all_predictions = prediction
    all_labels = lgtm_label

    # prediction
    # lgtm_label
    f1_result = f1_score(prediction, lgtm_label, average='macro')
    #print("f1_result of block ",block," is ",f1_result)

    assert len(all_predictions) == len(all_labels)
    f1_final = f1_score(all_predictions, all_labels, average='macro')
    return f1_final


f1_final = Cecilia_rf(kernel_size, learning_rate)
print("f1_final part LGBM: ", f1_final)
