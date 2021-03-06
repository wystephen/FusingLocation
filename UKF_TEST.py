# -*- coding:utf-8 -*-
# carete by steve at  2016 / 12 / 21　17:02
'''
Test UKF.py

'''

import os
import numpy as np

import matplotlib.pyplot as plt
from OPENSHOE.UKF import UKFIns
from OPENSHOE import Setting
import time
import DataChronic

if __name__ == '__main__':
    ex_dir_list = list()
    for dir_name in os.listdir('./'):
        if '-0' in dir_name and not '0.' in dir_name:
            ex_dir_list.append(dir_name)
            # if '03-' in dir_name:  # or '-0'in dir_name:
            #     print(dir_name)
            #     location = FusingLocation(dir_name)
            #     location.OnlyPF()
            #     location.Transform()
            #     location.Fusing(200)
            #
            #     plt.show()
            #
            #   3
            #   13
    print(ex_dir_list)
    ex_dir_list.sort()
    print(ex_dir_list)

    ticks = time.time()
    time_step = list()
    time_step.append(ticks)
    for i in [3]:
        dir_name = ex_dir_list[i]
        print(dir_name)
        # location = FusingLocation(dir_name, [0,1,2])
        # location.OnlyPF()
        # location.Transform()

        # location.Fusing(1000)
        # location.DeepFusing(1000)
        # location.MixFusing(1000, noise_sigma=1.0, evaluate_sigma=3.5)
        # plt.show()

        dc = DataChronic.DataChronic(dir_name,beacon_use=[0,1,2,3])

        dc.RunOpenshoe()
        dc.SynData()

        # UwbData[:,1:] = dc.UwbData[:,1:]/1000.0



        setting = Setting.settings()

        setting.Ts = np.mean(dc.ImuSourceData[1:, 0] - dc.ImuSourceData[0:-1, 0])
        # print("Ts:", setting.Ts)

        setting.min_rud_sep = int(1 / setting.Ts)

        setting.time_Window_size = 5
        setting.gamma = 6580

        setting.init_heading2 = setting.init_heading1
        setting.init_heading1 = setting.init_heading2 = 0.0
        ukfIns = UKFIns(settings=setting)

        ukf_result = np.zeros([dc.ImuSourceData.shape[0], 10])

        ukfIns.init_Nav_eq(dc.ImuSourceData[1:40, 1:7])

        for index in range(dc.ImuSourceData.shape[0]):
            try:
                if index > 10:
                    ukfIns.para.Ts = dc.ImuSourceData[index, 0] - \
                                     dc.ImuSourceData[index - 1, 0]
                ukf_result[index, 0] = dc.ImuSourceData[index, 0]

                ukf_result[index, 1:] = ukfIns.GetPosition(
                    dc.ImuSourceData[index, 1:7],
                    1
                ).reshape(9)
                print("index:", index, "x_h:", ukf_result[index, :])
            except np.linalg.LinAlgError as e:
                import matplotlib.pyplot as plt

                plt.figure(1)
                plt.plot(ukf_result[:, 1], ukf_result[:, 2], 'r-+')
                plt.grid(True)
                plt.show()
            finally:
                print(index)

        plt.figure(1)
        plt.plot(ukf_result[:, 1], ukf_result[:, 2], 'r-+')
        plt.grid(True)
        plt.show()
