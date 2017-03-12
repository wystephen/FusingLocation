# -*- coding:utf-8 -*-
# ReadData at 17-3-12 上午8:59

import numpy as np

import matplotlib.pyplot as plt


import DataChronic

if __name__ == '__main__':
    dir_name = "/home/steve/Data/locate/3"
    dc = DataChronic.DataChronic(dir_name)
    dc.RunOpenshoe()
    dc.SmoothPath()
    dc.SynData()

    UwbData = dc.UwbData

    plt.figure(0)
    plt.plot(UwbData[:,1],'r')
    plt.plot(UwbData[:,2],'b')
    plt.plot(UwbData[:,3],'y')

    plt.grid(True)


    plt.figure(2)

    ImuResult = dc.ImuResultSyn
    plt.plot(ImuResult[:,1],ImuResult[:,2],'r-+')
    plt.grid(True)

    plt.show()