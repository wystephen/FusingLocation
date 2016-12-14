# -*- coding:utf-8 -*-
# carete by steve at  2016 / 12 / 14　16:50

class ResultEvaluate:
    def __init__(self,keypointmatrix):
        '''

        :param keypointmatrix: n * 3 matrix,n is number of key points,each line is [x,y,time_stamp].
        '''
        self.KeyPointData = keypointmatrix


    def Distance2Line(self,point,timestamp):

        for

