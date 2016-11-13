# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 12　14:36



import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from OPENSHOE import PdrEkf, Setting, zupt_test

import XimuDataPreProcess

from log_process import seq_process

from DataChronic import DataChronic

if __name__ == '__main__':
    import os
    imu_time_app = list()
    uwb_time_app = list()
    for dir_name  in os.listdir('./'):
        if '-' in dir_name and (not '06' in dir_name) and (not '04' in dir_name):
            dc = DataChronic(dir_name)

            print(dir_name,dc.ImuSourceData[0,0],dc.ImuSourceData[-1,0],
                  dc.UwbData[0,0],dc.UwbData[-1,0])

            imu_time_app.append(dc.ImuSourceData[0,0])
            imu_time_app.append(dc.ImuSourceData[-1,0])

            uwb_time_app.append(dc.UwbData[0,0])
            uwb_time_app.append(dc.UwbData[-1,0])

    np.save('imutime',np.asarray(imu_time_app,dtype=float))
    np.save('uwbtime',np.asarray(uwb_time_app,dtype = float))


