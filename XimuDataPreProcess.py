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
                # print(file_name)
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
                # self.data_index =  np.loadtxt(file_dir + "/" + file_name,delimiter=',',dtype='str')
                # self.data_index = self.data_index[1:,:].astype('float')

                the_lines = [line.split(',') for line in open(file_dir + '/' + file_name)]

                the_lines = the_lines[1:]
                self.data_index = np.asarray(the_lines,dtype=float)
                # print(self.data_index.shape)
                # print(self.data_index[0,:])

            elif 'Time' in file_name:
                # print("time",file_name)
                the_lines = [line.split(',') for line in open(file_dir + '/' + file_name)]

                the_lines = the_lines[1:]

                self.time_index = np.asarray(the_lines,dtype=int)
                # print(self.time_index.shape)
                # print(self.time_index[0,:])

        '''
        For synchronic.
        '''
        # test time array to time that units is s.
        import time
        self.sec_index = np.zeros([self.time_index.shape[0],2])
        for index in range(self.time_index.shape[0]):
            ISFORMAT = "%Y-%m-%d-%H-%M-%S"
            tmp_time_str = '{0}-{1}-{2}-{3}-{4}-{5}'.format(self.time_index[index,1],self.time_index[index,2],
                   self.time_index[index,3],self.time_index[index,4],self.time_index[index,5],
                   self.time_index[index,6])


            self.sec_index[index,0]=(time.mktime(time.strptime(tmp_time_str,ISFORMAT)))
            self.sec_index[index,1]= self.time_index[index,0]

        print(self.sec_index)
        # np.savetxt('sec_index.txt',self.sec_index.astype('int'))

        # Test synchronic
        # plt.figure(1)
        # plt.plot(self.sec_index[:,1],'b+-')
        # plt.grid(True)
        #
        # plt.figure(2)
        # plt.plot(self.sec_index[:,1],'r+-')
        #
        # plt.show()

        last_second_point = 0
        last_second_point = 0

        for i in range(self.sec_index.shape[0]):
            if i == self.sec_index.shape[0] or self.sec_index[i,0] < self.sec_index[i+1,0] - 0.8:
                # Don't use the first second's data.

                if last_second_point == 0:

                    last_second_point = i+1
                    continue
                index_offset = self.data_index[i,1] - self.data_index[last_second_point,1]

                print(1.0 / float(index_offset))

                last_second_point = i+1














if __name__ == '__main__':
    '''
    Just for Test
    '''
    xdpp = XimuDataPreProcess("test4")