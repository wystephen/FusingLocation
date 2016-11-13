# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 12　20:15

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    imutime = np.load('imutime.npy')
    uwbtime = np.load('uwbtime.npy')

    plt.figure(1)

    plt.plot(imutime+ np.mean(uwbtime)-np.mean(imutime),'r-+')
    plt.plot(uwbtime,'b-+')

    plt.figure(2)
    plt.plot(uwbtime-imutime-np.mean(uwbtime)+np.mean(imutime),'g-+')
    plt.grid(True)


    print('time offset:',np.mean(uwbtime)-np.mean(imutime))





    plt.show()
