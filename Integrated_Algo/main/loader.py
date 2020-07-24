import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import fft_process

def get_slice(dat, index):
    '''
    retrieve the i-th block in original files
    '''
    return dat[index * 20480: (index + 1) * 20480]


data = scipy.io.loadmat('bearing.mat')
# use '1', '2', '3' to get the three data sets
# the next index is the bearing index

data20 = data['2'][0] # gets the first bearing in dataset 2
# contains 984 * 20480 readings

plt.plot(data20)
plt.title('Full signal of a failed bearing')
plt.savefig('failed.png')
plt.clf()

fft_model = fft_process.FFTProcess({"fs": 20480})

slice1 = get_slice(data20, 0) # get the first 20480 readings

plt.plot(slice1)
plt.title('Slice of the first second')
plt.savefig('begin.png')
plt.clf()

output = fft_model.fft_analysis(slice1)

# print fft result
plt.plot(output[1], output[0])
plt.title('FFT output of first sec signal')
plt.savefig('fft_begin.png')
plt.clf()


slice2 = get_slice(data['2'][0], 900)

plt.plot(slice2)
plt.title('A Slice right before the failure')
plt.savefig('later.png')
plt.clf()

output = fft_model.fft_analysis(slice2)

plt.plot(output[1], output[0])
plt.title('FFT of the signal before failure')
plt.savefig('fft_later.png')