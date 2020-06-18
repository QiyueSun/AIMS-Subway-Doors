import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import fft_process
import wavelet_threshold_denoise as wtd
import sys

def get_slice(dat, index):
    '''
    retrieve the i-th block in original files
    '''
    return dat[index * 20480: (index + 1) * 20480]

def format_n(i):
    return '{:0>4d}'.format(i)

data = scipy.io.loadmat('bearing.mat')
# use '1', '2', '3' to get the three data sets
# the next index is the bearing index

data20 = data['1'][4]

fft_model = fft_process.FFTProcess({"fs": 20480})

para=dict()
para['level']=3
para['wavelet']='db3'
para['mode']='greater'# soft hard greater less garrot
para['substitute']=0
para['method']='sqtwolog'# sqtwolog rigsure heursure minimaxi
wtdm = wtd.WaveletThresholdDenoise(para)

for i in range(data20.shape[0] // 20480):
    s = get_slice(data20, i)
    denoised = wtdm.wavelet_denoise([s])[0]
    fft_out = fft_model.fft_analysis(denoised)
    plt.plot(fft_out[1], fft_out[0])
    plt.title('FFT: denoised ' + format_n(i))
    plt.savefig('wavelet_denoise_fft_img/fft_denoised' + format_n(i) + '.png')
    plt.clf()
    if i > 0:
        sys.stdout.write("\033[F")
    print(i + 1, "/", data20.shape[0] // 20480)