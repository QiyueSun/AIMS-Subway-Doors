# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:24:33 2020

@author: zhidemao
"""
#导入包
import numpy as np
from scipy.signal import filtfilt
from scipy.linalg import toeplitz

class med_process: #最小熵解卷积处理类
    
    def __init__(self, params=None):
        '''参数初始化
        '''
        self.params = params
        
    def kurt(self, x):
        '''计算信号峭度值:
            Args:
                x :计算峭度值的数据
            Return:
                result :返回峭度指标
        '''
        RMS = np.sqrt(sum([i ** 2 for i in x]) / len(x))
        Kurtosis = sum([(abs(i) - np.mean(x))**4 for i in x])/len(x)
        result = Kurtosis/RMS**4
        
        return result
        
    def med_2d(self, dat=None):
        '''MED算法降噪:
            Args:
                dat:需要降噪的数据
            Rrturn:
                res:降噪后的数据
            这里的降噪方法实现代码具体是参考matlab编写的，因此有些地方注释不太完善
        '''
        if "filterSize" in self.params.keys(): #获取最小熵解卷积配置参数滤波窗口大小
            filterSize = self.params["filterSize"]
        else:
            filterSize = 30
            
        if "termIter" in self.params.keys(): #获取最小熵解卷积配置参数迭代长度
            termIter = self.params["termIter"]
        else:
            termIter = 30
            
        if "termDelta" in self.params.keys(): #获取最小熵解卷积配置参数衰减率
            termDelta = self.params["termDelta"]
        else:
            termDelta = 0.01
        

        denoise_dat = [] #用于存储降噪后的数据
        for i in range(len(dat)):
            x = dat[i]
            # 计算数据中大于0的数的均值
            new_x = abs(x)
            mean_x = np.mean(new_x)
                
            #计算加权的自相关矩阵作为行的平均自相关矩阵
            L = filterSize
            autoCorr = np.zeros(L)
            for k in range(L):
                x2 = np.zeros(len(x))
                x2[k:] = x[0:(len(x)-k)]
                autoCorr[k] = autoCorr[k] + sum(x[:]*x2)
            autoCorr = autoCorr/1000
            A = toeplitz(autoCorr)
            A_inv = np.linalg.inv(A)
            
            #初始化矩阵
            f = np.zeros(L)
            y = np.zeros((np.size(x),1))
            kurtIter = []
            
            f[1] = 1
            n = 1
            while n == 1 or (n<termIter and 
                             ((self.kurt(filtfilt(f,1,x))-kurtIter[n-2])>termDelta)):
                y = filtfilt(f,1,x)
                kurtIter.append(self.kurt(y))
                weightedCrossCorr = np.zeros(L)
                for k in range(L):
                    x2 = np.zeros(np.size(x))
                    x2[k:] = x[0:(len(x)-k)]
                    weightedCrossCorr[k] = weightedCrossCorr[k] + sum((y[:]**3)*x2)
                f = A_inv@weightedCrossCorr
                f = f/np.sqrt(sum(f**2))
                n = n + 1
            f_final = f
            y_final = filtfilt(f_final,1,x)
            kurtIter.append(self.kurt(y_final)) 
            
            #计算降噪后数据大于0部分的平均值
            new_y = abs(y_final)
            mean_y = np.mean(new_y)
            
            y_final = y_final*(mean_x/mean_y)
            y_final = np.array(y_final)
#            y_final = np.reshape(y_final, (len(y_final), 1))
            
            #将降噪后的数据进行汇总整合
            denoise_dat.append(y_final)

        return denoise_dat
    
##############################################
#'''
#以下代码仅用于测试
#'''
#
#def main():
#    pass
#
#######轴承仿真信号####################
#    import scipy.io as sci
#    data = sci.loadmat('../data/test_data.mat')
#    x = data['x']
#    x = np.resize(x, (5000))
#    freqs = 30
######################################
#          
##############辛辛那提大学轴承疲劳寿命试验数据##########
##    data = sci.loadmat('../data/firstData1.mat')
##    data = data['firstData']
##    x = data[600,:]
##    freqs = 230
######################################################
#    
###############中车永济测试数据##########################
##    data = sci.loadmat('../data/4200testdata.mat')
##    x = data['x']
##    x = np.resize(x, (len(x)))
##    freqs = 30
########################################################
##
##    filterSize = freqs
##    termIter=30
##    termDelta=0.01
##    plotMode=1
##    
##    ins = MED_denoising()
##    y_final, kurtosis1, kurtosis2 = ins.med_2d(x, filterSize, termIter, termDelta, plotMode)
##    print('原始信号峭度：%f , 滤波后信号峭度：%f' %(kurtosis1, kurtosis2))
##    
##    import matplotlib.pyplot as plt
##    from matplotlib.pylab import mpl
##    mpl.rcParams['font.sans-serif'] = ['SimHei'] #显示中文
##    mpl.rcParams['axes.unicode_minus']=False #显示负号
##    plt.figure()
##    plt.subplot(211)
##    plt.plot(x, linewidth=0.5)
##    plt.title('降噪前信号时域图')
##    plt.subplot(212)
##    plt.plot(y_final, linewidth=0.5)
##    plt.title('降噪后信号时域图')
##    plt.show()
#
#
#if __name__ == "__main__":
#    main()
    

        