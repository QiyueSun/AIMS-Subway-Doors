import numpy as np
import os
import glob
import scipy.io

dirs = ['../1st_test', '../2nd_test', '../4th_test/txt']

cnt = 1
mat_dict = {}
for d in dirs:
    files = glob.glob(d + '/*')
    files = sorted(files)

    data_list = []
    for f in files:
        data = np.genfromtxt(f).T
        data_list.append(data)
    data = np.concatenate(np.array(data_list), axis=1)
    print(data)
    mat_dict[str(cnt)] = data
    cnt = cnt + 1

scipy.io.savemat('bearing.mat', mat_dict)