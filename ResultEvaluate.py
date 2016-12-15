# -*- coding:utf-8 -*-
# carete by steve at  2016 / 12 / 14　16:50

import numpy as np
class ResultEvaluate:
    def __init__(self,keypointmatrix):
        '''

        :param keypointmatrix: n * 3 matrix,n is number of key points,each line is [x,y,time_stamp].
        '''
        self.KeyPointData = keypointmatrix


    def Distance2Line(self,point,timestamp):

        for i in range(self.KeyPointData.shape[0]):
            if timestamp < self.KeyPointData[i + 1, 2] and \
                            timestamp >= self.KeyPointData[i, 2]:
                tp1 = self.KeyPointData[i, 0:2]
                tp2 = self.KeyPointData[i + 1, 0:2]

                tp1 = tp1 - point
                tp2 = tp2 - point

                if np.linalg.norm(tp1 - tp2) > 0.1:
                    tmp_v = tp1[0] * tp2[1] - tp2[0] - tp1[1]
                    return np.abs(tmp_v) / np.linalg.norm(tp1 - tp2)

                else:
                    return (np.linalg.norm(tp1) + np.linalg.norm(tp2)) / 2.0

                print("Find :", i, timestamp, self.KeyPointData[i, 1], self.KeyPointData[i + 1, 2])
            else:
                return np.linalg.norm(point - self.KeyPointData[-1, 0:2])
