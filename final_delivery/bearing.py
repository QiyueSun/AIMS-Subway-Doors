import scipy.io
import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt
import fft_process
import wavelet_threshold_denoise as wtd
import sys
import os
from parameter_tune import execute_out as Tune
from parameter_tune import Tune_lite
from parameter_tune import Tune_denoise
# from TQWT_std import tqwt
# from ITQWT_std import itqwt
from tqwt_tools import tqwt, itqwt

def get_slice(dat, index):
    '''
    retrieve the i-th block in original files
    '''
    return dat[index * 20480: (index + 1) * 20480]

def format_n(i):
    return '{:0>4d}'.format(i)


def find_freq(freqs):
    '''
    从突出频率中选取原频率及二倍频同时存在的频率
    '''
    ret = []
    for f in freqs:
        if np.min(np.abs(np.array(freqs) - 2 * f)) < 2:
            ret.append(f)
    return ret

def extract_signal(freq, dat):
    '''
    从频谱中提取突出信号，提取幅值大于周围信号平均值 * multiplier的信号
    '''
    width = 9
    half_width = width // 2
    multiplier = 3
    l = []
    thresholds = dat[:]
    tSum = np.sum(dat[:width])
    for i in range(half_width, len(dat) - half_width - 1):
        thresholds[i] = (tSum - dat[i]) * multiplier / (width - 1)
        tSum = tSum - dat[i - half_width] + dat[i + half_width + 1]
    for i in range(half_width, len(dat) - half_width - 1):
        if freq[i] >= 5000:
            break
        if dat[i] > thresholds[i]:
            l.append(freq[i])
        
    
    return find_freq(l) # 提取出可能为特征频率的值
    # if len(l) > 1 and np.abs(l[1] - 2 * l[0]) < 2:
    #     return l[0]
    # else:
    #     return None


def diagnose(s, b, a, fft_model, plot=True):
    '''
    诊断主函数
    输入：s: 振动信号
          b, a: 滤波器参数 (scipy.signal.butter)
          fft_model: 用于计算fft的模型 (fft_process)
    输出：特征频率
    '''
    q, r, l = Tune_lite(s[:5000]) # TQWT参数调优
    w = tqwt(s, q, r, l)
    recomposed = itqwt(w[:-1], q, r, len(s)).real
    # plt.subplot(2, 1, 1)
    # plt.plot(s)
    # plt.subplot(2, 1, 2)
    # plt.plot(recomposed)
    # plt.show()
    # recomposed = s
    level, wavelet, mode, substitute, method = Tune_denoise(recomposed) # 选择最优降噪参数
    wtdm = wtd.WaveletThresholdDenoise({'level': level, 'wavelet': wavelet, 'mode': mode, 'substitute': substitute, 'method': method})
    denoised = wtdm.wavelet_denoise([recomposed])[0] # 降噪
    filtered = denoised # ss.filtfilt(b, a, denoised) # 滤波
    hil = ss.hilbert(filtered) # hilbert 变换
    envelope = np.sqrt(np.power(filtered, 2) + np.power(hil, 2)) #计算包络
    envelope -= np.mean(envelope)
    fft_out = fft_model.fft_analysis(envelope) # 计算频谱
    if plot:
        plot_len = int(np.floor(800 / fft_out[1][-1] * len(fft_out[1])))
        plt.plot(fft_out[1][:plot_len], fft_out[0][:plot_len])
    return extract_signal(fft_out[1], fft_out[0]) # 从频谱提取故障频率

def freq_to_fault(freqs, rpm = 1, ratio = {'outer': 300, 'inner': 210, 'roller': 130}):
    '''
    将故障频率转译为故障位置
    '''
    ret = []
    for k in ratio.keys():
        for f in freqs:
            if np.abs(f / (ratio[k] * rpm) - 1) < 0.02 or np.abs(f / (2 * ratio[k] * rpm) - 1) < 0.02 or np.abs(f / (3 * ratio[k] * rpm) - 1) < 0.02:
                ret.append(k)
                break
    return ret

if __name__ == '__main__':
    # use '1', '2', '3' to get the three data sets
    # the next index is the bearing index

    fft_model = fft_process.FFTProcess({"fs": 20480})

    para=dict()
    para['level']=2
    para['wavelet']='db1'
    para['mode']='less'# soft hard greater less garrot
    para['substitute']=0
    para['method']='minimaxi'# sqtwolog rigsure heursure minimaxi
    wtdm = wtd.WaveletThresholdDenoise(para)



    datasets = ['1']
    b, a = ss.butter(4, 0.11, 'highpass')
    n = 10

    old_params = None
    pfile = open("py/params.txt")
    pstr = pfile.read().split('\n')
    if pstr[-1] == '':
        del pstr[-1]
    amps = np.float64(pstr)
    pfile.close()
    print(amps)
    old_params = amps
    s = simulate.simulate(20480, 2000, 50, 5000, [300, 210, 130], 50, amps, 0.3)
    cha_f = diagnose(s, b, a, fft_model, plot=True)
    faults = freq_to_fault(cha_f)
    fft_out = fft_model.fft_analysis(s)
    # plt.plot(fft_out[1], fft_out[0])
    plt.savefig('py/spectrum.png')
    plt.clf()
    print(cha_f)
    print(faults)
    while True:
        try:
            result = open('py/result.txt', 'w')
            break
        except:
            continue
    for f in ['outer', 'inner', 'roller']:
        if f in faults:
            result.write('1')
        else:
            result.write('0')

    result.close()     
