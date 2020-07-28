#!/usr/bin/python
# -*- coding: utf-8 -*-

import scipy.io
import scipy.signal as ss
import numpy as np
import matplotlib.pyplot as plt
import bearing
import fft_process
import pandas as pd

def loader(date, rpm):
    d = './val_data/val/' + str(date)
    return scipy.io.loadmat(d + '/' + str(rpm) + '_' + 'bearing.mat')

def get_bearing_info(channel):
    # returning type is a dictionary, {'type': 'direction'}
    # X-j: X-径向,  Y-z: Y-周向,  Z-z: Z-轴向,  X-z2: X-周向,  Y-z2: Y-轴向,  Z-j2: Z-径向
    mapping = {'1':{'NU214': 'X-j'}, '2':{'NU214': 'Y-z'}, '3':{'NU214': 'Z-z'},
                '4':{'NU214': 'X-45'}, '5':{'NU214': 'Y-45'}, '6':{'NU214': 'Z-45'},
                '7':{'6311': 'X-j'}, '8':{'6311': 'Y-z'}, '9':{'6311': 'Z-z'},
                '10':{'6311': 'X-z2'}, '11':{'6311': 'Y-z2'}, '12':{'6311': 'Z-j2'},
                }
    return mapping[str(channel)]

def get_slices(dat, n_slices=3):
    '''
    retrieve the i-th block in original files
    '''
    sample = 12800
    ret = []
    for i in range(n_slices):
        ret.append(dat[i * 20 * sample: (i * 20 + 1) * sample])
    return ret

if __name__ == "__main__":
        # For each folder, there are 5 .mat files, with rpm values of '422', '1804', '2600', '4100', '4712'.
        # Specify the date & rpm to get the signal data
        # Then, Specify the channel to get bearing information (type + direction)
        # Below is the avaliable parameter choices
        # 1.16 422+06, 1804+12,  2600+01, 4100+06, 4712+06
        # 1.17 422+06, 1804+06,  2600+06, 4100+（02, 04, 06）, 4712+（06, 07）
        # 1.18 422+（06, 07）, 1804+01,  2600+（01, 03, 04, 06, 07, 08, 09, 10, 11, 12）, 4100+（01, 04, 06, 07, 08, 09, 11, 12）, 4712+（05, 06, 07）
        # 1.19 422+06, 1804+05, 2600+（01, 02, 03, 04, 06, 07, 08, 12）, 4100+（01, 06, 07, 08, 11）, 4712+06
        # 1.20 422+06, 1804+（02, 07, 08, 10, 12）, 2600+（06, 07, 12）, 4100+（05, 06, 09）, 4712+（01, 02, 06, 07, 11）
        # 1.21 422+06, 1804+（06, 07, 08, 09, 12）, 2600+（03, 04, 05, 06, 07, 08, 09, 11, 12）, 4100+（01, 06, 10）, 4712+06
    date = 1.16
    rpm = 422
    channel = 6
    ds = ['1.16', '1.17', '1.18', '1.19', '1.20', '1.21']
    fs = [422, 1804, 2600, 4100, 4712]
    df = [(d, f) for d in ds for f in fs]
    ch = [['06'], ['12'], ['01'], ['06'], ['06'],
          ['06'], ['06'], ['06'], ['02', '04', '06'], ['06', '07'],
          ['06', '07'], ['01'], ['01', '03', '04', '06', '07', '08', '09', '10', '11', '12'], ['01', '04', '06', '07', '08', '09', '11', '12'], ['05', '06', '07'],
          ['06'], ['05'], ['01', '02', '03', '04', '06', '07', '08', '12'], ['01', '06', '07', '08', '11'], ['06'],
          ['06'], ['02', '07', '08', '10', '12'], ['06', '07', '12'], ['05', '06', '09'], ['01', '02', '06', '07', '11'],
          ['06'], ['06', '07', '08', '09', '12'], ['03', '04', '05', '06', '07', '08', '09', '11', '12'], ['01', '06', '10'], ['06']]
    data = loader(1.16, 422) # return a dictionary {'channel': signal}
    bearing_info = get_bearing_info(channel) # return a dictionary of bearing type + direction
    n_slice = 10
    fd = open('result.txt', 'w')
    for i in range(len(df)):
        data = loader(df[i][0], df[i][1])
        for c in ch[i]:
            sig = get_slices(data[c][0], n_slice)
            cf = sig[:]
            f = cf[:]
            b, a = ss.butter(4, df[i][1] / 25600, 'highpass')
            fftm = fft_process.FFTProcess({"fs": 25600})
            if c >= '07':
                ratio = {'roller': 0.06676, 'outer': 0.05095, 'inner': 0.08238}
            else:
                ratio = {'roller': 0.12278, 'outer': 0.11555, 'inner': 0.15111}
            for j in range(n_slice):
                cf[j] = bearing.diagnose(sig[j], b, a, fftm, plot=True)
                f[j] = bearing.freq_to_fault(cf[j], rpm=df[i][1], ratio=ratio)
                plt.savefig('fig/' + df[i][0] + '_' + str(df[i][1]) + '_' + c + '_' + str(j) + '.png')
                plt.clf()
            print(df[i][0] + ' ' + str(df[i][1]) + ' ' + c + str(f), file=fd)
            print(df[i][0] + ' ' + str(df[i][1]) + ' ' + c + str(f))

    ########################################
    # Data Visualization

    #     plt.plot(data20)
    #     plt.title('Full signal of a failed bearing')
    #     plt.savefig('failed.png')
    #     plt.clf()

    #     fft_model = fft_process.FFTProcess({"fs": 20480})

    #     slice1 = get_slice(data20, 0) # get the first 20480 readings

    #     plt.plot(slice1)
    #     plt.title('Slice of the first second')
    #     plt.savefig('begin.png')
    #     plt.clf()

    #     output = fft_model.fft_analysis(slice1)

    #     # print fft result
    #     plt.plot(output[1], output[0])
    #     plt.title('FFT output of first sec signal')
    #     plt.savefig('fft_begin.png')
    #     plt.clf()


    #     slice2 = get_slice(data['2'][0], 900)

    #     plt.plot(slice2)
    #     plt.title('A Slice right before the failure')
    #     plt.savefig('later.png')
    #     plt.clf()

    #     output = fft_model.fft_analysis(slice2)

    # plt.plot(output[1], output[0])
    # plt.title('FFT of the signal before failure')
    # plt.savefig('fft_later.png')