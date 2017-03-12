# -*- coding:utf-8 -*-
# ReadData at 17-3-12 上午8:59

import numpy as np

import matplotlib.pyplot as plt


import DataChronic

if __name__ == '__main__':
    dir_name = "/home/steve/Data/locate/5"
    dc = DataChronic.DataChronic(dir_name)
    dc.RunOpenshoe()
    dc.SmoothPath()
    dc.SynData()
    dc.OnlyPF()

    dc.UwbData[:,0] = dc.UwbData[:,0]*1000.0

    np.savetxt(dir_name+'beaconset.data',dc.BeaconSet)
    np.savetxt(dir_name+'beaconset.data.csv',dc.BeaconSet,delimiter=',')

    np.savetxt(dir_name+'UwbData.data',dc.UwbData)
    np.savetxt(dir_name+'UwbData.data.csv',dc.UwbData,delimiter=',')

    np.savetxt(dir_name+'UwbResult.data',dc.UWBResult)
    np.savetxt(dir_name+'UwbResult.data.csv',dc.UWBResult,delimiter=',')

    np.savetxt(dir_name+'ImuData.data',dc.ImuSourceData.astype(dtype=float))
    np.savetxt(dir_name+'ImuData.data.csv',dc.ImuSourceData.astype(dtype=float),delimiter=',')

    np.savetxt(dir_name+'Zupt.data',dc.zupt)
    np.savetxt(dir_name+'Zupt.data.csv',dc.zupt,delimiter=',')

    np.savetxt(dir_name+'ImuResultData.data',dc.ImuResultSyn)
    np.savetxt(dir_name+'ImuResultData.data.csv',dc.ImuResultSyn,delimiter=',')


    UwbData = dc.UwbData

    plt.figure(0)
    plt.plot(UwbData[:,1],'r')
    plt.plot(UwbData[:,2],'b')
    plt.plot(UwbData[:,3],'y')

    plt.grid(True)


    plt.figure(2)

    ImuResult = dc.ImuResultSyn
    plt.plot(ImuResult[:,0],ImuResult[:,1],'r-+')
    plt.grid(True)

    plt.figure(3)
    plt.plot(dc.ImuSourceData[:,0],'r')
    plt.plot(dc.UwbData[:,0],'b')
    plt.grid(True)

    plt.show()