#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:06:24 2019

@author: francesco
"""
import pandas as pd
import numpy as np

from scipy.signal import find_peaks   #, peak_widths
from scipy.optimize import curve_fit
#from numpy.polynomial import polynomial as P

import matplotlib.pyplot as plt
import matplotlib

from sklearn.metrics import f1_score, accuracy_score

from math import pi


plotting = False

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = (24, 20)


#################################################################################
#                 VARIUS
#################################################################################


def smooth(x,fraction=0.01,window='hamming'):
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
    
    if fraction >1.0:
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



def probability_density(data=None, density=True):
    
    #n_bins = int(np.sqrt(len(data)))
    n, b     = np.histogram(data, bins='auto', density=density)
    n        = n.astype(np.float)
    b_center = 0.5*(b[:-1] + b[1:])
    #n_sigma  = (n + 1.0)**0.999
    return n, b_center


def find_1d_array_minima(y):
    min_mask      = np.r_[False, y[1:] < y[:-1]] & np.r_[y[:-1] < y[1:], False]
    indices       = np.arange(0, len(y), 1, dtype=np.int)
    return indices[min_mask]


def find_array_element_closest_to_value(array, value):
    array = np.asarray(array)
    idx   = (np.abs(array - value)).argmin()
    return idx

  


################################################################################################
#                          FUNCTIONS   
################################################################################################


def gaus(x,a,x0,sigma):
    #print(a,x0,sigma)
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def constant(x, c):
    return np.full(len(x), c)


def line(x, b, c):
    return c + b*x 


def parabola(x, a, b, c):
    return c + b*x +a*x**2


def sin(x, A, C):
    x = x - x[0]
    W = pi/50.0
    return A * np.sin(W*x) + C


def MultiNormal(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amp = params[i]
        mu  = params[i+1]
        sig = params[i+2]
        y   = y + gaus(x,amp,mu,sig)
    return y


################################################################################################
#                          FITTING   
################################################################################################


def fit_sin(x, y):
    
    #
    x = x - x[0]
    
    #
    A1, A2  = 0.0, 6.0
    C1, C2  = -5.0, 12.0              
    #
    bounds = ([A1, C1], [A2, C2])
    (A,C),pcov = curve_fit(sin, x, y, bounds=bounds, maxfev=10000)
    
    # To avoid problem when drift removal is not needed
    
    if np.abs(A)<1.0:
        bounds = ([C1], [C2])        
        (cost), pcov = curve_fit(constant, x, y, bounds=bounds, maxfev=10000)    
        (A,C) = (0.0, cost[0])
    
    popt = np.array([A,C])
    
    y_fit = sin(x, *popt)
    return y_fit, popt


def fit_parabola(x, y):
    
    x = x - x[0]
    
    #
    a1, a2  = -20.0, 0.0
    b1, b2  = 0.0, 1.0
    c1, c2  = -np.infty, np.infty
    
    # try to fit a parabola
    bounds = ([a1, b1, c1], [a2, b2, c2])
    (a,b,c),pcov = curve_fit(parabola, x, y, bounds=bounds, maxfev=10000)

    # try with a line
    if np.abs(a)<4e-3:
        bounds = ([b1, c1], [b2, c2])        
        (b,c),pcov = curve_fit(line, x, y, bounds=bounds, maxfev=10000)
        (a,b,c) = (0.0,b,c) 
    
    # try with a constant
    if np.abs(b)<1e-1:
        bounds = ([c1], [c2])        
        (cost), pcov = curve_fit(constant, x, y, bounds=bounds, maxfev=10000)    
        (a, b, c) = (0.0, 0.0, cost[0])
        
    popt = np.array([a,b,c])   
        
    y_fit = parabola(x, *popt)
    return y_fit, popt



def fit_multinormal(x=None, y=None,  mus=None, sig1=None, sig2=None, Dmu=7e-2):
    bound1, bound2   = [], []

    for mu in mus:
        # find local maximum
        idx = find_array_element_closest_to_value(x, mu)
        A   = np.max([1e-12, 1.2 * y[idx]])

        bound1 += [0.0]
        bound1 += [mu - Dmu*np.abs(mu)]
        bound1 += [sig1]
    
        #
        bound2 += [A]
        bound2 += [mu + Dmu*np.abs(mu)]
        bound2 += [sig2]
        
    bounds = (bound1, bound2)
    #print(bounds)
    
    popt, pcov = curve_fit(MultiNormal, x, y, p0=bound1,  bounds=bounds, maxfev=10000)
    return popt


################################################################################################
#                          MAKE PREDICTION   
################################################################################################
    
  



def prediction_MGM(s=None, fit_params=None, open_channels=None):
    #
    prob        = np.array([gaus(s,*par) for par in fit_params])
    #
    return np.array([open_channels[np.argmax(p)] for p in prob.T])





def evaluation(y_pred=None, y_true=None):
    #
    f1_1   = f1_score(y_true, y_pred, average='macro')    
    errors = np.where(y_pred!=y_true)[0]
    
    #        
    acc1   = accuracy_score(y_true, y_pred)     
    return acc1, f1_1, errors   
  


def prediction1(groups):
    predictions, times, channels = np.array([]), np.array([]), np.array([])
    for G in groups:
        time        = G.t
        signal      = G.s
        params      = G.MultiNormalFit
        pred_chan   = G.predicted_channels
        
        #
        # prob        = np.array([gaus(signal,*par) for par in params])
        # predi       = np.array([pred_chan[np.argmax(p)] for p in prob.T])
        predi       = prediction_MGM(s=signal, fit_params=params, open_channels=pred_chan)
        predictions = np.append(predictions, predi)
        times       = np.append(times, time)
        
        #
        try:
            channels = np.append(channels, G.c)
        except:
            pass
        
    p = np.argsort(times)
    
    #
    try:
        channels = channels[p]
    except:
        pass
            
    return predictions[p], channels   


    

def full_evaluation1(groups):
    y_pred = np.array([])
    y_true = np.array([])
    
    #
    for G in groups:
        y_pred = np.append(y_pred, prediction1(G))
        y_true = np.append(y_true, G.channel)    
    #
    f1_1   = f1_score(y_true, y_pred, average='macro')    
    acc1   = accuracy_score(y_true, y_pred)
    return f1_1, acc1



##################################################################################################
#                 FIND PEAKS LOCATION
##################################################################################################





def find_max_peak(y=None, x=None):
    peaks, _   = find_peaks(y, height=0.0, prominence=3.0*y.mean())
    i_max      = np.argmax(y[peaks])
    i_mu0, mu0 = peaks[i_max], x[peaks[i_max]]
    return i_mu0, mu0 



def find_peak_start_end(y, i_peak):
    i_min    = find_1d_array_minima(y)
    idx = find_array_element_closest_to_value(i_min, i_peak)
    
    if i_min[idx] < i_peak:
        i1 = i_min[idx]
        i2 = int(i1 + 2 * (i_peak - i1) )    
    elif i_min[idx] > i_peak:    
        i2 = i_min[idx]
        i1 = int(i2 - 2 * (i2 - i_peak) )

    return i1, i2



def evaluate_chi(s=None, mu=None, sig=None):

    n, b = probability_density(s)
    
    (A, mu, sig) = fit_multinormal(x=b, y=n,  mus=[mu], sig1=0.5*sig, sig2=1.7*sig, Dmu=2.0e-2)


    n_fit = gaus(b, *(A, mu, sig))

    #if len(n) < 12:    
    #    chi = np.abs(n - n_fit).sum()/15
    #else:    
    chi = np.abs(n - n_fit).sum()/len(n)
          
    return chi, (A, mu, sig) 




def remove_blur(s=None):

    mus = find_all_paeks_positions(s=s)
    
    sep = np.diff(mus).mean()
    
    cond1 = s < np.min(mus) - 0.7 * sep
    cond2 = s > np.max(mus) + 0.7 * sep
    
    
    down = np.where(cond1)[0]
    up   = np.where(cond2)[0]
    
    
    if len(down) > 3e-3 * s.size:
        for i in down:
            try:
                s[i] = np.mean(s[i-15:i+15])
            except:
                pass

    if len(up) > 3e-3 * s.size:
        for i in up:
            try:
                s[i] = np.mean(s[i-15:i+15])
            except:
                pass

    ###    
    return s




def check_signal_is_gaussian(s=None, mu=None, sig=None, chi_min=0.33):
    
    if len(s)<10:
        return False, None

    #    
    chi, (A, mu, sig) = evaluate_chi(s=s, mu=mu, sig=sig)
    
    chi_min = np.max([chi_min, 0.15])
    
    #print(chi, chi_min, A, mu, sig)
    
    if A < 1e-3:
        return  False, None        
    elif chi > chi_min:
        return  False, None
    elif chi < chi_min:
        return  True, (A, mu, sig)



def find_all_paeks_positions_(s=None):
    n, b_       = probability_density(s)
    x           = b_
    y           = smooth(n, fraction=0.07)
    
    # get boundaries
    boundaries = find_boundary(s)
    sep        = find_mean_sep(s)
    
    
    # find peak positions
    i_mu0, mu0 = find_max_peak(y=y, x=x)
    mus1  = np.arange(mu0, boundaries[0], -sep)
    mus2  = np.arange(mu0+sep, boundaries[1] - 0.35 * sep,  sep)
    mus   = np.append(mus1, mus2)
    mus.sort()
    
    # add missing peaks      
    return mus


def add_manually_extra_peak(mus=None, s=None):
    sep_new = np.diff(mus).mean()

    mus.sort()

    if len(mus) > 7:
        for i in range(1, 11 - len(mus)):
            mus += [mus[0]-i*sep_new]
    elif (len(mus) == 4) :
        mu_new = np.max(mus) + sep_new
        print(s[ s > mu_new  ].size)
        if (s[ s > mu_new  ].size > 10) and s[ s > mu_new  ].size < 100:
            mus += [mu_new]
    # peaks must be eaven        
    elif (len(mus) == 5) or (len(mus) == 3):
        
        mu_min = np.min(mus) - sep_new
        s1     = s[(s < np.min(mus) - 0.6 * sep_new)]
        ch1, (A, mu1, sig) = evaluate_chi(s=s1, mu=mu_min, sig=0.2 * sep_new)            
        
        #
        mu_max = np.max(mus) + sep_new
        s2 = s[(s > np.max(mus) + 0.6 * sep_new)]
        ch2, (A, mu2, sig) = evaluate_chi(s=s2, mu=mu_max, sig=0.2 * sep_new)

        if ch1 < ch2:
            mus += [mu1]
        elif ch1 > ch2:    
            mus += [mu2]
    
    return mus
        
            

def find_all_paeks_positions(s=None):
    n, b       = probability_density(s)
    x          = b
    y          = smooth(n, fraction=0.07)            
                
                
    i_mu0, mu0 = find_max_peak(y=y, x=x)
    sep        = find_mean_sep(s)
    
    cond = (mu0 - 0.4 * sep < s) & (s < mu0 + 0.4 * sep)
    chi_min, _ = evaluate_chi(s=s[cond], mu=mu0, sig=0.2*sep)
    
    mus = []
    
    for mu in np.arange(mu0 - 12*sep, mu0 + 12*sep, sep):
    
        cond = (mu - 0.4 * sep < s) & (s < mu + 0.4 * sep) 
        s_   = s[cond]
        is_gauss, par = check_signal_is_gaussian(s=s_, mu=mu, sig=0.2*sep,chi_min=4*chi_min)
    
        if is_gauss:
            mus  += [par[1]]
            
        # make sure that all peaks are equally distant
        try:
            if mus[-1] - mus[-2] > 1.15*sep:
                print('remove peak because not euqually spaced with the others')
                mus = mus[:-1]
                break
        except:
            pass
    
    mus = add_manually_extra_peak(mus=mus, s=s)

    mus.sort()

    return np.array(mus)


##################################################################################################
#                 FIT USING A MULTINORMAL DISTRIBUTION
##################################################################################################



def fit_single_block_with_MGM(single_block=None, plot=False):
    s = single_block.s
    t = single_block.t
        
    # find peaks's positions and open channels
    mus           = find_all_paeks_positions(s)
    open_channels = find_which_channels_are_opened(s)
    sig_mean      = find_mean_sig(s)
    
    # Multi Gaussian fit 
    
    if sig_mean > 0.38:
       sig1 = 0.4 * sig_mean
       sig2 = 0.75 * sig_mean
    else:
       sig1 = 0.5 * sig_mean
       sig2 = 1.1 * sig_mean
    
    
    y, x             = probability_density(data=s, density=False)
    popt             = fit_multinormal(x=x, y=y, mus=mus, sig1=sig1, sig2=sig2)
    
    #
    single_block.predicted_channels = open_channels
    single_block.MultiNormalFit     = np.array([ popt[i:i+3] for i in range(0, len(popt), 3) ]) 
    
    #
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 12))
        plot_check1(ax=ax1, s=s, t=t, real_channels=single_block.c)
        plot_check2(ax=ax2, s=s, parameters=single_block.MultiNormalFit, 
                    Title='', density=False)
    
    return None





##################################################################################################
#                    Data extraction
##################################################################################################
        



class SingleBlock(object):
    def __init__(self, signal=None, time=None, channels=None):
        self.s        = signal
        self.t        = time 
        try:
            self.c = channels
        except:
            print('channels not defined')
            
    def __add__(self, other):
        s_tot  = np.append(self.s, other.s)
        t_tot  = np.append(self.t, other.t)
        try:
            c_tot = np.append(self.c, other.c)  
        except:
            c_tot = None
        #    
        return SingleBlock(signal=s_tot, time=t_tot, channels=c_tot)       
        



##################################################################################################    
#                        Useful to understand which channels are opened   
##################################################################################################


def find_mean_sep(s=None):
    n, b    = probability_density(data=s)
    x = b
    y = smooth(n, fraction=0.07)
    
    # Find and fit main peaks
    i_peaks, _  = find_peaks(y, height=0.0, prominence=0.05 * y.mean())
    if len(i_peaks)<2:
        i_peaks, _  = find_peaks(y, height=0.0, prominence=0.03 * y.mean())
    
    if len(i_peaks)<2:
        i_peaks, _  = find_peaks(y, height=0.0, prominence=0.02 * y.mean())
        
    if len(i_peaks)<2:
        i_peaks, _  = find_peaks(y, height=0.0, prominence=0.01 * y.mean())    
        
    if len(i_peaks)<2:
        plt.plot(x, y)
        plt.show()
        raise Exception('Only on peak found, cannot compute separation!')
        
    seps        = np.diff(x[i_peaks])
    sep         = np.median(seps)       
    return sep



def find_mean_sig(s=None):
    n, b    = probability_density(data=s)
    x = b
    y = smooth(n, fraction=0.07)
    
    # Find and fit main peaks
    i_peaks, _    = find_peaks(y, height=0.0, prominence=0.05 * y.mean())
    sigmas        = []
           
    for i_peak in i_peaks:
        
        i1, i2  = find_peak_start_end(y, i_peak) 
        w       = x[i2] - x[i1]
        popt    = fit_multinormal(x=x[i1:i2], y=y[i1:i2], mus=[x[i_peak]], sig1=0.1*w, sig2=0.7*w)
        #   
        sigmas += [popt[2]]
    #
    sigmas      = np.array(sigmas)    
    sig_mean    = sigmas.mean()

    return sig_mean



def find_boundary(s, fract=1.0):
    N          = int(fract*len(s))
    s_sort     = np.array(sorted(s[:N]))
    boundaries = s_sort[:20].mean(), s_sort[-20:].mean()
    if boundaries[1]>5.05:
        print('Warning: boundary overlimit')
        fract = fract - 0.05
        print('Change fraction from', fract)
        boundaries = find_boundary(s, fract=fract)
    return boundaries
  


def find_which_channels_are_opened(s=None):

    #boundaries = find_boundary(s)
    mus        = find_all_paeks_positions(s=s)
    
    # find open channels
    #if (boundaries[1]>3.5) & (len(mus)>8):
    if  (len(mus)>8):
        chn_min = 11 - len(mus)
        predicted_channels = np.arange(chn_min, 11)
    
    else:
        predicted_channels = np.arange(0, len(mus))
        
    return predicted_channels    



##################################################################################################
#                 PLOTTING
##################################################################################################



def plot_check_drift_removal(t=None, s=None, s_no_draft=None, draft_fit=None):  
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    #
    fit        = draft_fit 
    s_         = s_no_draft 
    
    #
    ax1.plot(s,    t, marker='.', lw=0, markersize=0.2, label='original')
    ax1.plot(fit,  t, marker='o', lw=1, markersize=4.1, label='fit')
    ax2.plot(s_,   t, marker='.', lw=0, markersize=0.2, label='no drift')
    
    ax1.legend()
    ax2.legend()
    
    plt.subplots_adjust(wspace=0, hspace=0)  




def plot_check2(ax=None, s=None, parameters=None, Title=False, density=True):
    #
    y, x       = probability_density(data=s, density=density)
    
    if Title:
        ax.title.set_text(Title)
    
    for par in parameters:
        fit = MultiNormal(x, *par)
        ax.plot(x, y, lw=4.0, alpha=0.73)
        ax.plot(x, fit, marker='X', lw=0.0, markersize=7) 



    
    


##################################################################################################
#                        INFER GROUP TYPE
##################################################################################################

   

def same_type(mu1=None, mu2=None):
    #
    if (len(mu1) > 9) and (len(mu2) > 9):
        return np.allclose(mu1[-2:], mu2[-2:], atol=1.2e-1)
    
    elif len(mu1)!=len(mu2):
        return False
    
    else:
        return np.allclose(mu1[:2], mu2[:2], atol=1.2e-1)
   

    
def add_element_to_model(model=None, i1=None, i2=None):    
    #
    new_el            = np.array([i1, i2])
    element_not_found = True
    
    # # 
    for n, v in model.items():
        if (i1 in v) | (i2 in v):
            model[n] = np.append( model[n], new_el)
            model[n] = np.unique(model[n])
            element_not_found = False
    # #
    if element_not_found:
        try:
  
            model[n+1] = new_el
        except:
            model[0] = new_el

    return None



def group_blocks(all_blocks):
    #
    block_indx = {}
    #
    for i, B1 in enumerate(all_blocks):
        for j, B2 in enumerate(all_blocks[i+1:]):    
            #
            mu1 = B1.MultiNormalFit[:,1]
            mu2 = B2.MultiNormalFit[:,1]
            if same_type(mu1=mu1, mu2=mu2):
                add_element_to_model(model=block_indx, i1=i, i2=j+1+i)
    #
    
    new_blocks = []
    
    for k, indices in block_indx.items():
        
        new_blocks += [ all_blocks[indices[0]] ]  
        for i in indices[1:]:
            new_blocks[k] += all_blocks[i]
            
                
    return block_indx, new_blocks 



def infer_block_type(B=None):
    path = './'
    df = pd.read_hdf(path + '/block_type.h5', key='block_type')
        
    B.block_type = 'unclassified'

    for Type in df:
        s   = df[Type]
        mu_ = s.values
        mu_ = mu_[mu_<300]
        #
        mu2 = B.MultiNormalFit[:,1]
        
        #print(mu_)
        #print(mu2)            
        if same_type(mu1=mu_, mu2=mu2):
            B.block_type = Type
    return None        


def compute_shift(all_mus):
    shifts  = []    
    pos0    = 0 
    
    for mu in all_mus:    
        ##############
        if len(mu)==10:
            sep_mean = np.diff(mu).mean()
            shifts  += [pos0 - mu[1] + sep_mean]
        elif len(mu)==9:
            sep_mean = np.diff(mu).mean()
            shifts  += [pos0 - mu[2] + 2*sep_mean]
        else:    
            shifts  += [pos0 - mu[0]] 
            print(mu[0])
    return shifts        



def shift_the_signal(blocks=None):
    all_mus = np.array([B.MultiNormalFit[:,1] for B in blocks])
    
    shifts  = compute_shift(all_mus)
    
    shifted_signal = np.array([])
    
    for B, shift in zip(blocks, shifts):
        print(shift)
        shifted_signal_ = B.s + shift
        shifted_signal  = np.append(shifted_signal, shifted_signal_)
    return shifted_signal 



##################################################################################################
####            BLOCK SEPARATION    
##################################################################################################



 
    
def check_if_sigma_is_a_good_fit(s=None, t=None, n=5, fraction=0.1):  
    
    t = t - t[0]
    
    
    s_smooth    = smooth(s[::n], fraction=fraction)
    y_fit, popt = fit_sin(t[::n], s_smooth)  #fit_parabola(x=B.t, y=B.s)

    
    chi = np.sum( np.abs(s_smooth - y_fit) ) / len(s_smooth) 

    global plotting

    if False:
        plt.plot(t, s, label='signal')
        plt.plot(t[::n], y_fit, lw=20, label='sin')
        plt.plot(t[::n], s_smooth, lw=10, label='smooth')
        plt.title(str(chi), fontsize=40)
        plt.legend(fontsize=40)
        plt.show()

    
    if chi > 0.1:
        return False
    else:
        return True


        
def find_partial_drift(s=None, t=None):

    step_min = int(5e+4)     
    steps    = np.arange(0, len(s)+1, step_min, dtype=np.int) 
        
    
    fit_paras, delimiters =  [], []
    
    for i0, i1 in zip(steps[:-1], steps[1:]):
        yfit, par  = fit_parabola(t[i0:i1], s[i0:i1])
        sizes      = find_size_of_block(s=s[i0:i1]-yfit, t=t[i0:i1])
        #
        fit_paras  += [[ par[0], par[1] ]]
        delimiters += [sizes]
        #
        t_fit= t[i0:i1] - t[i0:i1][0]
        y_ = parabola(t_fit, a=par[0], b=par[1], c=par[2])
        #y_ = sin(t_fit, A=par[0], C=par[1])
        
        if plotting:
            plt.plot(t[i0:i1], y_, lw=15, color='black')
    
    if plotting:
        plt.plot(t, s, marker='.', lw=0.0, markersize=0.1, label='signal', color='red')
        plt.legend(fontsize=40)
        plt.show()

    return np.array(fit_paras), np.array(delimiters)        
        
        
        
def find_size_of_block(s=None, t=None):
    # prevent some oscilation
    s_sort = sorted(s)   
    s_min  = np.array([np.mean(s_sort[:int(n) ]) for n in np.arange(1000, 4000.1, 200)]).mean()
    s_max  = np.array([np.mean(s_sort[-int(n):]) for n in np.arange(1000, 4000.1, 200)]).mean()
    
    #
    max_    = s_max.mean()
    min_    = s_min.mean()
    
    #
    return np.array([max_, min_ ])       



def subsplit_a_block(s=None, t=None):
    ''' split in micro bins
    '''
    
    pars, delim =  find_partial_drift(s=s, t=t)
    #pars    = np.where(pars==0.0, 1000, pars)
    
    indices, i = [0], 0 
    
    for par0, par1, (a0, b0), (a1, b1) in zip(pars[:-1,:], pars[1:,:], delim[:-1,:], delim[1:,:]):
        #
        i += int(5e+4)
        #
        chi_par = np.abs(par0 - par1).sum()
        chi_sp1 = np.abs(a0 - a1)
        chi_sp2 = np.abs(b0 - b1)
        
        
        if (chi_par > 0.2) or min(chi_sp1, chi_sp2)>0.1:
            indices += [i] 
    
    indices += [int(5e+5)]
    return indices



