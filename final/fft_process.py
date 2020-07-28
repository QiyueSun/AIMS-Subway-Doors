# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:33:52 2019

@author: xingwang
"""

from scipy.fftpack import fft
import numpy as np

class FFTProcess: #快速傅里叶变换分析类
    
    def __init__(self, params=None):
        self.params = params
        
    def fft_analysis(self, dat=None):
        '''快速傅里叶变换
        Args:
            dat: 需要进行傅里叶变换的数据 
        Returns:

        '''
        if "fs" in self.params.keys():
            sampling_freqs = self.params["fs"]

        sampling_freqs = np.array(sampling_freqs)
        if np.ndarray == type(dat):
            dat = dat.tolist()
        length_dat = len(dat) #计算待处理数据长度
#        t = np.linspace(0, length_dat/sampling_freqs, num = length_dat) #计算时间分辨率
        fft_dat = np.abs(fft(dat))*2/length_dat # 计算数据的FFT变换
        half_y = fft_dat[range(int(length_dat/2))] #由于频谱的对称性，取频谱的一半进行分析
        f = np.linspace(0, sampling_freqs, num = length_dat) #生成与之对应的频率坐标点
        half_f = f[range(int(length_dat/2))] #取一半的频率范围
        
#        new_list = {}
#        new_list["data"] = half_y.tolist()
#        new_list["freq"] = half_f.tolist()
        new_list = []
        new_list.append(half_y.tolist())
        new_list.append(half_f.tolist())
        
        
        return new_list