# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:53:55 2020

@author: xingwang
"""

import pywt
import numpy as np
#import sys
#import os

#pywt.threshold(data, value, mode='soft', substitute=0)

class WaveletThresholdFunction:
    '''
    小波降噪阈值函数
    '''
    def __init__(self, params=None):
        self.params = params
    
    def sqtwolog(self,dat=None):
        '''
        固定阈值
        Use the universal threshold sqrt(2ln(length(x))).
        '''
        value=np.sqrt(2*np.log(len(dat)))
        return value

    def rigsure(self,dat=None):
        '''
        无偏风险估计阈值
        Use the principle of Stein's Unbiased Risk.
        '''
        N=len(dat)
        dat2=np.sort(np.abs(dat))**2
        risk=[(N-2*i+np.sum(dat2[0:i])+(N-1-i)*dat2[N-1-i])/N for i in range(0,N)]
        value=np.sqrt(dat2[risk.index(min(risk))])
        return value
    
    def heursure(self,dat=None):
        '''
        启发式阈值
        Use a heuristic variant of Stein's Unbiased Risk.
        '''
        N=len(dat)
        crit=np.sqrt(((np.log(N)/np.log(2))**3)/N)
        eta=(np.sum(dat**2)-N)/N
        if eta<crit:
            value=self.sqtwolog(dat)
        else:
            value=min(self.sqtwolog(dat),self.rigsure(dat))
        return value
    
    def minimaxi(self,dat=None):
        '''
        极大极小阈值
        Use minimax thresholding
        '''
        N=len(dat)
        if N>32:
            value=0.3936+0.1829*(np.log(N)/np.log(2))
        else:
            value=0
        return value
    
    def threshold_function(self,dat=None):
        '''
        降噪阈值计算主函数
        '''
        if "method" in self.params.keys():#判断key值是否为空
            method = self.params["method"] #阈值函数
        else:
            method="sqtwolog"
        
        if method == "sqtwolog":
            value=self.sqtwolog(dat)
        elif method == "rigsure":
            value=self.rigsure(dat)
        elif method == "heursure":
            value=self.heursure(dat)
        elif method == "minimaxi":
            value=self.minimaxi(dat)
        else:
            print(method, "does not exist.\n")
        value=value*np.median(np.abs(dat))/0.6745
        return value
        
    
    
class WaveletThresholdDenoise:
    '''
    小波阈值降噪
    '''
    
    def __init__(self, params=None):
        
        self.params = params
        
    def wavelet_decompose(self, dat=None):
        '''
        小波分解
        '''
        if "level" in self.params.keys():#判断key值是否为空
            level = int(self.params["level"]) #信号的分解层数
        else:
            level=3
        if "wavelet" in self.params.keys():#判断key值是否为空
            my_wave = self.params["wavelet"] #小波基
        else:
            my_wave="db5"
        dat_coeffs = pywt.wavedec(dat, wavelet=my_wave, mode='cpd', level=level) #对信号进行分解
        return dat_coeffs
    
    def wavelet_reconstruct(self, dat_coeffs=None):
        '''
        小波重构
        '''
        if "wavelet" in self.params.keys():#判断key值是否为空
            my_wave = self.params["wavelet"] #小波基
        else:
            my_wave="db5"
        dat_rec = pywt.waverec(dat_coeffs, my_wave)
        return dat_rec
    
    def wavelet_threshold(self,dat=None,value=None):
        '''
        小波分量阈值降噪
        '''
        if "mode" in self.params.keys():#判断key值是否为空
            mode = self.params["mode"] #阈值模式：soft,hard,greater,less,garrot
        else:
            mode="hard"
        if "substitute" in self.params.keys():#判断key值是否为空
            substitute = self.params["substitute"] #替换值
        else:
            substitute=0
        res=pywt.threshold(data=dat, value=value, mode=mode, substitute=substitute)
        return res
    
    def wavelet_denoise(self, dat=None):
        '''
        小波降噪主函数
        '''
        denoise_dat = []
        
        for i in range(len(dat)):
            x = dat[i]
            para={}
            if "method" in self.params.keys():#判断key值是否为空
                para["method"] = self.params["method"] #小波阈值计算函数
            else:
                para["method"] = "sqtwolog"
            ins=WaveletThresholdFunction(para)
            dat_coeffs=self.wavelet_decompose(x)
            new_coeffs=list()
            new_coeffs.append(dat_coeffs[0])
            for dec_data in dat_coeffs[1:]:
                value=ins.threshold_function(np.array(dec_data))
                new_coeffs.append(self.wavelet_threshold(dec_data,value))
            denoise_data=self.wavelet_reconstruct(new_coeffs)
            denoise_data = np.array(denoise_data)
        
            denoise_dat.append(denoise_data)
            # print(type(denoise_dat))
        
        #返回汇总后的降噪数据
        return denoise_dat
    
#############################################
#以下代码仅用于测试
#############################################

def main(dat=None):

    para=dict()
    para['level']=3
    para['wavelet']='db3'
    para['mode']='greater'# soft hard greater less garrot
    para['substitute']=0
    para['method']='sqtwolog'# sqtwolog rigsure heursure minimaxi
    
    ins=WaveletThresholdDenoise(para)
    denoise_data=ins.wavelet_denoise(dat)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(211)
    plt.plot(dat)
    plt.subplot(212)
    plt.plot(denoise_data)
    plt.show()
    
    return denoise_data
    


if __name__ == "__main__":
    import scipy.io as sci
    data = sci.loadmat("C://Users//zhidemao//Desktop//denoising_analysis//data//4712data.mat")
    x = data["x4712"]
    denoise_data=main(x)
    