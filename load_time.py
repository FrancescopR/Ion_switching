#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:10:03 2020

@author: francesco
"""

import h5py


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
        f = h5py.File('/liverpool_ion_switching.hdf5', 'r')

        self.block = {}
        blocks = list(f[key])

        for block in blocks:
            i = int(block.replace('block', ''))
            self.block[i] = Block(b=f[key][block])

        f.close()


# %%
TR = LoadData(key='train')

# %%

print(TR.block[10].f1_HMM)

# %%
