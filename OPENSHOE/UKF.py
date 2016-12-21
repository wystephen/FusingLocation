# -*- coding:utf-8 -*-
# carete by steve at  2016 / 12 / 19　11:52
'''
This module is use to compute imu path through ukf method.
'''
import numpy as np
import scipy as sp

import math

from OPENSHOE.Setting import settings


class UKFIns(object):
    '''

    '''
    def __init__(self,settings):
        '''

        '''
        self.para = settings

        self.R=np.zeros([6,6])


