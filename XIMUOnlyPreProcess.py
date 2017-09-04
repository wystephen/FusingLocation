
import numpy as np
import scipy as sp

import os

'''
    Quick read and save acc gyro and attitude by mag
'''

if __name__ == '__main__':
    dir_name = '/home/steve/Data/XIMU&UWB/2/'

    file_list = os.listdir(dir_name)



    for file_name in file_list:
        if 'Logged' in file_name:
            if 'InertialAndMa' in file_name:
                # process inertial and mag
                inertialData = np.loadtxt(dir_name+file_name,
                                          dtype=np.str,
                                          delimiter=',',
                                          skiprows=0)
                inertialData = inertialData[1:,:]
                inertialData = inertialData.astype(dtype=np.float)
                inertialData = inertialData[:,1:]
                inertialData[:,0:3] *=(np.pi / 180.0)
                inertialData[:,3:6] *=9.81
                print("inertial Data shape:",
                      inertialData.shape)


            elif 'Quaternion' in file_name:
                # process quaternion by mag
                magData = np.loadtxt(dir_name+file_name,
                                     dtype=np.str,
                                     delimiter=',',
                                     skiprows=0)
                magData = magData[1:,:]
                magData = magData.astype(dtype=np.float)
                magData = magData[:,1:]
                print("quaternion Data shape :",
                      magData.shape)


    imu_lens = min(magData.shape[0],inertialData.shape[0])

    ImuData = np.zeros([imu_lens,magData.shape[1]+inertialData.shape[1]])

    ImuData[:,:inertialData.shape[1]] = inertialData[:imu_lens,:]
    ImuData[:,-magData.shape[1]:] = magData[:imu_lens,:]

    print(ImuData[0:3,:])




