import scipy.io
import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt
import fft_process
import wavelet_threshold_denoise as wtd
import sys
import os
import TQWT

def get_slice(dat, index):
    '''
    retrieve the i-th block in original files
    '''
    return dat[index * 20480: (index + 1) * 20480]

def format_n(i):
    return '{:0>4d}'.format(i)


def find_freq(freqs):
    ret = []
    freqs = np.array(freqs)
    for f in freqs:
        if np.min(np.abs(freqs - 2 * f)) < 2:
            ret.append(f)
    return ret

def extract_signal(freq, dat):
    width = 9
    half_width = width // 2
    multiplier = 2.5
    l = []
    thresholds = dat[:]
    tSum = np.sum(dat[:width])
    for i in range(half_width, len(dat) - half_width - 1):
        thresholds[i] = tSum * multiplier / width
        tSum = tSum - dat[i - half_width] + dat[i + half_width + 1]
    for i in range(half_width, len(dat) - half_width - 1):
        if freq[i] >= 1000:
            break
        if dat[i] > thresholds[i]:
            l.append(freq[i])
    return find_freq(l)
    # if len(l) > 1 and np.abs(l[1] - 2 * l[0]) < 2:
    #     return l[0]
    # else:
    #     return None


def diagnose(s, wtdm, b, a, fft_model, tqwt, plot=True):
    c, w, N = tqwt.TQWT(s, 1.7, 3, 3)
    c = np.zeros(np.shape(c))
    recomposed = tqwt.ITQWT(c, w, N, 1.7, 3, 3)
    denoised = wtdm.wavelet_denoise([recomposed])[0]
    filtered = ss.filtfilt(b, a, denoised)
    hil = ss.hilbert(filtered)
    envelope = np.sqrt(np.power(filtered, 2) + np.power(hil, 2))
    fft_out = fft_model.fft_analysis(envelope)
    if plot:
        plt.plot(fft_out[1][:2000], fft_out[0][:2000])
    return extract_signal(fft_out[1], fft_out[0])

def freq_to_fault(freqs):
    rpm = 1900
    ratio = {'outer': 0.1217, 'inner': 0.1617, 'roller': 0.0559}
    ret = []
    for k in ratio.keys():
        for f in freqs:
            if np.abs(f / (ratio[k] * rpm) - 1) < 0.05:
                ret.append(k)
                break
    return ret

data = scipy.io.loadmat('bearing.mat')
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

result = open('result.txt', 'w')

datasets = ['1', '2', '3']
b, a = ss.butter(4, 0.1, 'highpass')
tqwt = TQWT.TQWTDenoise()

for d in datasets:
    os.mkdir("wavelet_denoise_fft_img/" + d)
    for id in range(data[d].shape[0]):
        os.mkdir("wavelet_denoise_fft_img/" + d + "/" + str(id))
        data20 = data[d][id]

        print("Processing dataset " + d + ", channel " + str(id) + ":")

        for i in range(data20.shape[0] // 20480 - 50, data20.shape[0] // 20480):
            s = get_slice(data20, i)
            cha_f = diagnose(s, wtdm, b, a, fft_model, tqwt, plot=True)
            faults = freq_to_fault(cha_f)
            result.write(d + ' ' + str(id) + ' ' + format_n(i) + ': ' + str(faults) + '\n')
            result.flush()
            plt.title('dataset=' + d + ' channel=' + str(id) + ' FFT: hilbert + denoised ' + format_n(i))
            plt.savefig('wavelet_denoise_fft_img/' + d + '/' + str(id) + '/' + 'fft_h_denoised' + format_n(i) + '.png')
            plt.clf()
            if i > data20.shape[0] // 20480 - 50:
                sys.stdout.write("\033[F")
            print(i + 1, "/", data20.shape[0] // 20480)