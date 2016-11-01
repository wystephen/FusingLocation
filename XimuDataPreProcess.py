# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 01　21:26


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import os

class XimuDataPreProcess:
    def __init__(self,file_dir):
        file_lists = os.listdir(file_dir)
        # print(file_list)
        # print(type(file_lists))
        for file_name in file_lists:
            if 'Inertial' in file_name:
                print(file_name)
                # data_file  = open(file_dir + "/" + file_name,'r')
                # data_file = data_file.readlines()
                # print("len:",len(data_file))
                '''
                Reference code:
                lines = np.loadtxt('iris.csv',delimiter=',',dtype='str')
                df = lines[1:,:4].astype('float')
                Reference code 2(quicker):
                lines = [line.split(',') for line in open('iris.csv')]
                df = [[float(x) for x in line[:4]] for line in lines[1:]]

                '''
                self.data_index =  np.loadtxt(file_dir + "/" + file_name,delimiter=',',dtype='str')
                self.data_index = self.data_index[1:,:].astype('float')


                print(self.data_index.shape)
                print(self.data_index[0,:])

            elif 'Time' in file_name:
                print("time",file_name)






if __name__ == '__main__':
    '''
    Just for Test
    '''
    xdpp = XimuDataPreProcess("test2")