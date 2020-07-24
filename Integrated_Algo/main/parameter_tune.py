#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats as stats
import scipy.signal as ss
from TQWT_std import tqwt as tqwt
from ITQWT_std import itqwt as itqwt
import pdb
import scipy.io
import matplotlib.pyplot as plt
import fft_process
import wavelet_threshold_denoise as wtd
import math


def get_slice(dat, index):
    '''
    retrieve the i-th block in original files.
    '''
    return dat[index * 20480: (index + 1) * 20480]


def getEntropy(y):
    '''
    Return the envelop Entrop of input signal used for IK calculation.
    '''
    fft_model = fft_process.FFTProcess({"fs": 20480}) # choose fft_model
    para=dict()
    para['level']=2
    para['wavelet']='db1'
    para['mode']='less'# soft hard greater less garrot
    para['substitute']=0
    para['method']='minimaxi'# sqtwolog rigsure heursure minimaxi
    wtdm = wtd.WaveletThresholdDenoise(para) # choose wavelet_threshold denosie model

    b, a = ss.butter(4, 0.1, 'highpass')
    denoised = wtdm.wavelet_denoise([y])[0]
    filtered = ss.filtfilt(b, a, denoised)
    hil = ss.hilbert(filtered)
    envelope = np.sqrt(np.power(filtered, 2) + np.power(hil, 2))
    fft_out = fft_model.fft_analysis(envelope)
    prob = np.array(fft_out)*1.0/(sum(fft_out))
    entropy = 0
    for p in prob:
        entropy = -p*math.log(p) 
    return entropy


def find_kurtosis(data, Q, red,level_num):
    '''
    Find a K or IK for a given signal
    '''
    w = tqwt(data, Q, red, level_num)
    comb_signals = []

    #########################################
    for j in range (level_num + 1):
        comb_signal = [w[i] for i in range(j+1)]
        comb_signals.append(comb_signal)
    for j in range (level_num):
        comb_signals.append([np.array(w[j])]) # append the decomposed signals w[i] on the end of the []

    #######################################
    max_kurtosis = 0
    chosen_level = 0
    i=0
    re_signals = []
    for comb_signal in comb_signals:
        i = i + 1
        y = itqwt(comb_signal, Q, red, 20480)
        ########################################
        # Comment out to introduce the IK value
        # E = getEntropy(y)
        # kurtosis = kurtosis*1.0 / E # Now the kurtosis is IK
        #########################################
        re_signals.append(y)
        kurtosis = stats.kurtosis(y).real
        if kurtosis > max_kurtosis:
            max_kurtosis = kurtosis.real
            chosen_level = i # index starts from 1

    return max_kurtosis, chosen_level, comb_signals[chosen_level-1]


def Tuning_q(data, num, r, l):
    '''
    Tune Q value given fixed r, l.
    '''
    gap = 0.1
    Qs = [1.0+i*gap for i in range(num)]
    reds = [r for i in range(num)]
    levels = [l for i in range(num)]

    signal_candi = []
    k_vals = []
    levels_candi = []
    for i in range(num):
        q = Qs[i]
        red = reds[i]
        level_num = levels[i]
        try:
            max_k, level, sub_band = find_kurtosis(data, q, red, level_num)
        except:
            continue

        signal_candi.append(sub_band)
        k_vals.append(max_k)
        levels_candi.append(level)

    maxx_k = max(k_vals)
    index = k_vals.index(maxx_k)

    return Qs[index]


def Tuning_r(data, num, q, l):
    '''
    Tune r value given fixed q, l.
    '''
    gap = 0.1
    Qs = [q for i in range(num)]
    reds = [3.0 + 0.1*gap for i in range(num)]
    levels = [l for i in range(num)]

    signal_candi = []
    k_vals = []
    levels_candi = []
    for i in range(num):
        q = Qs[i]
        red = reds[i]
        level_num = levels[i]
        try:
            max_k, level, sub_band = find_kurtosis(data, q, red, level_num)
        except:
            continue

        signal_candi.append(sub_band)
        k_vals.append(max_k)
        levels_candi.append(level)

    maxx_k = max(k_vals)
    index = k_vals.index(maxx_k)

    return reds[index]


def Tuning_l(data, num, q, r):
    '''
    Tune l value given fixed q, r.
    '''
    gap = 1
    Qs = [q for i in range(num)]
    reds = [r for i in range(num)]
    levels = [1 + gap*i for i in range(num)]

    signal_candi = []
    k_vals = []
    levels_candi = []
    for i in range(num):
        q = Qs[i]
        red = reds[i]
        level_num = levels[i]
        try:
            max_k, level, sub_band = find_kurtosis(data, q, red, level_num)
        except:
            continue

        signal_candi.append(sub_band)
        k_vals.append(max_k)
        levels_candi.append(level)

    maxx_k = max(k_vals)
    index = k_vals.index(maxx_k)

    return levels[index]


def Tune_lite(data):
    '''
    Lite algorithm of finding the optimal parameteric settings.
    '''
    q_num = 15
    r_num = 10
    l_num = 8
    q_init = 1.0
    r_init = 3.0

    l_optimal = Tuning_l(data, l_num, q_init, r_init)
    q_optimal = Tuning_q(data, q_num, r_init, l_optimal)
    r_optimal = Tuning_r(data, r_num, q_optimal, l_optimal)
    print(1)
    return q_optimal, r_optimal, l_optimal


def execute_out(data_raw):
    '''
    Input: signal of a certain channel
    Output: optimal parameters (Q, r, L) for the channel signals
    '''
    data_list = []
    datalen = data_raw.shape[0] // 20480
    for i in range(datalen - 200, datalen - 150):#data20.shape[0] // 20480):
        slice_n = list(get_slice(data_raw, i))
        data_list.append(slice_n)
    
    Qs = []
    Rs = []
    Ls = []
    ##################################
    for data in data_list:
        data = np.array(data)
        q, r, l = Tune_lite(data)
        Ls.append(l)
        Qs.append(q)
        Rs.append(r)
    
    #####################################
    # Analysis

    # K_filtered = np.array([value for i, value in enumerate(Ks) if value > 3])
    # indexs = [i for i, value in enumerate(Ks) if value > 3]
    # Q_filtered = np.array([Qs[i] for i in indexs])

    R_opt = np.mean(Rs)
    Q_opt = np.mean(Qs)
    L_opt = int(np.mean(Ls))
    print(Ls)

    print(Q_opt, R_opt, L_opt)
    return Q_opt, R_opt, L_opt



if __name__ == "__main__":
    pdb.set_trace()
    data = scipy.io.loadmat('./preprocess_data/preprocessed/bearing.mat')
    # use '1', '2', '3' to get the three data sets
    # the next index is the bearing index

    data20 = data['2'][0] # gets the first bearing in dataset 2
    # data12 = data['1'][2]
    # data13 = data['1'][3]
    # data32 = data['3'][2]

    # execute(data20, 'data20') # dataset 2, bearing 1
    # execute(data12, 'data12') # dataset 1, bearing 3
    # execute(data13, 'data13') # dataset 1, bearing 4
    # execute(data32, 'data32') # dataset 3, bearing 3

    execute_out(data20)

