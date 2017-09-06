import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import os
import OPENSHOE.zupt_test
import OPENSHOE.PdrEkf
import OPENSHOE.Setting

import mpl_toolkits.mplot3d.axes3d as p3

'''
    Quick read and save acc gyro and attitude by mag
'''

if __name__ == '__main__':
    dir_name = '/home/steve/Data/XIMU&UWB/5/'

    file_list = os.listdir(dir_name)

    for file_name in file_list:
        if 'Logged' in file_name:
            if 'InertialAndMa' in file_name:
                # process inertial and mag
                inertialData = np.loadtxt(dir_name + file_name,
                                          dtype=np.str,
                                          delimiter=',',
                                          skiprows=0)
                inertialData = inertialData[1:, :]
                inertialData = inertialData.astype(dtype=np.float)
                inertialData = inertialData[:, 1:]

                inertialData[:, 0:3] *= (np.pi / 180.0)
                inertialData[:, 3:6] *= 9.81

                tmp_data = inertialData * 1.0 # pay attention to mem operation of numpy
                inertialData[:, 0:3] = tmp_data[:, 3:6]
                inertialData[:, 3:6] = tmp_data[:, 0:3]
                print("inertial Data shape:",
                      inertialData.shape)


            elif 'Quaternion' in file_name:
                # process quaternion by mag
                magData = np.loadtxt(dir_name + file_name,
                                     dtype=np.str,
                                     delimiter=',',
                                     skiprows=0)
                magData = magData[1:, :]
                magData = magData.astype(dtype=np.float)
                magData = magData[:, 1:]
                print("quaternion Data shape :",
                      magData.shape)

    imu_lens = min(magData.shape[0], inertialData.shape[0])

    ImuData = np.zeros([imu_lens, magData.shape[1] + inertialData.shape[1]])

    ImuData[:, :inertialData.shape[1]] = inertialData[:imu_lens, :]
    ImuData[:, -magData.shape[1]:] = magData[:imu_lens, :]

    print(ImuData[0:3, :])

    np.savetxt(dir_name + 'ImuData.csv', ImuData, fmt='%.18e', delimiter=',')

    '''
    zupt
    '''
    t_setting = OPENSHOE.Setting.settings()
    # t_setting.Ts = 1.0/400.0
    t_setting.Ts = 1.0 / 128.0

    t_setting.sigma_a *= 0.8
    t_setting.sigma_g *= 0.8

    # t_setting.sigma_acc = 0.5 * np.ones([3,1])
    # t_setting.sigma_gyro = 0.5 * np.ones([3,1]) *np.pi / 180.0

    t_setting.init_heading = t_setting.init_heading1 = t_setting.init_heading2 = 0.0

    zupt_detector = OPENSHOE.zupt_test.zupte_test(t_setting)

    zupt_result = zupt_detector.GLRT_Detector(ImuData[:, 0:6])

    ins_filter = OPENSHOE.PdrEkf.ZUPTaidedIns(settings=t_setting)
    ins_filter.init_Nav_eq(ImuData[1:20, 0:6], ImuData[1:20, 0:6])

    u1 = ImuData[:, 0:6]
    u2 = u1

    zupt1 = zupt2 = zupt_result

    all_x = np.zeros([18, u1.shape[0]])

    for index in range(u1.shape[0]):
        if (index % 100 == 0):
            print("finished : " + str(float(index) / u1.shape[0]))
        # if index > 1:
        #     ins_filter.para.TS = out_data[index,0]-out_data[index-1,0]
        all_x[:, index] = ins_filter.GetPosition(
            u1[index, :],
            u2[index, :],
            zupt1[index, :],
            zupt2[index, :]
        ).reshape([18])

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.plot(all_x[0, :], all_x[1, :], all_x[2, :])
    # plt.show()
    plt.figure()
    plt.title("acc")
    plt.plot(ImuData[:, 0:3])
    plt.plot(zupt_result)

    plt.figure()
    plt.title("gyro")
    plt.plot(ImuData[:, 3:6])
    plt.plot(zupt_result)

    plt.show()
