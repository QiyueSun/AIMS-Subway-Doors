# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:59:31 2019

@author: xingwang
"""

import pywt
#conda install PyWavelets
import copy
import numpy as np

#小波(包)分解
class WaveletProcess():
     
    def __init__(self, params = None):
        '''小波变换初始化参数
        Args:
            params: 所有配置信息，包括业务标识、算法配置、环节配置等 ，json格式
        '''
        self.params = params
        
    def wavelet_analysis(self, dat=None):
        '''小波分解与重构
        Args:
            dat:原始振动数据
        Return:
            dat_list:小波分解后重构的数据，里面包括各个频率
                     分量处重构的数据，返回的数据组数与分解层数一致
        '''

        if "level" in self.params.keys():#判断key值是否为空
            level = int(self.params["level"]) #信号的采样频率
        else:
            level=3
        if "wavelet" in self.params.keys():#判断key值是否为空
            my_wave = self.params["wavelet"] #信号的采样频率
        else:
            my_wave="db5"


        #小波分解
#        db1 = pywt.Wavelet('db1') #选择小波分解函数
        dat_coeffs = pywt.wavedec(dat, wavelet=my_wave, mode='cpd', level=level) #小波分解层数设置为4，对信号进行分解
        #小波重构
        #因为小波分解将原始信号分解为多个频率分量，各个分量长度不一致，因此需要对各个
        #频率分量信号进行重构，在重构某一频率段信号时，另其他无关频率分量处数值为O
        #最后得到重构信号dat_list中包含的数据组数与分解层数一致
        dat_list = [] #用于存储小波重构后的组数据
        for i in range(level+1): #小波重构环节
            dat_new = copy.deepcopy(dat_coeffs)
            if i==0:
                for j in range(1,level+1):
                    dat_new[j][:] = 0
            elif i==level:
                for j in range(0,level):
                    dat_new[j][:] = 0
            else:
                for j in range(0, i):
                    dat_new[j][:] = 0
                for k in range(i+1, level+1):
                    dat_new[k][:] = 0
            dat_rec = pywt.waverec(dat_new, my_wave)
            dat_list.append(dat_rec.tolist())
            
        if len(dat_list) != level+1:
             raise ValueError("Refactoring data dimensions are inconsistent.")


        return dat_list #返回小波变换重构后的组数据
    
    def wp_analysis(self, dat=None):
        '''小波包分解与重构
        Args:
            dat:原始振动数据
        Return:
            dat_list:小波分解后重构的数据，里面包括各个频率
                     分量处重构的数据，返回的数据组数与分解层数一致
        '''
        #小波包分解
        if "level" in self.params.keys():#判断key值是否为空
            level = int(self.params["level"]) #信号的采样频率
        else:
            level=3
        if "wavelet" in self.params.keys():#判断key值是否为空
            my_wave = self.params["wavelet"] #信号的采样频率
        else:
            my_wave="db5"
        wp = pywt.WaveletPacket(data=dat, wavelet=my_wave, maxlevel=level)
        node_list = [node.path for node in wp.get_level(level, 'freq')]
        #小波包重构
        dat_list = []
        length=len(dat)
        for i in node_list:
            new_wp = pywt.WaveletPacket(data=None, wavelet=my_wave, maxlevel=level)
            new_wp[i] = wp[i].data
            dat_rec = new_wp.reconstruct()
            dat_list.append(dat_rec[0:length].tolist())
            


        return dat_list #返回小波变换重构后的组数据