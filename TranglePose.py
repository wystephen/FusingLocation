# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 14　11:12

import numpy as np
from scipy.optimize import minimize


class tranglepose:
    def __init__(self,beaconset,range_list):
        self.pose = [0,0,0]
        self.beaconset = beaconset
        self.range_list= range_list

        init_pose = [0.0,0.0,0.0]
        res = minimize(self.costfunction_multi_range,init_pose)

        print(res.x)

    def costfunction_single_range(self,pose):
        val = 0.0
        for i in range(self.beaconset.shape[0]):
            val += np.abs(np.linalg.norm(self.beaconset[i,:]-self.pose)
                    -self.range_list[i])
        return val

    def costfunction_multi_range(self,pose):
        val = 0.0
        for j in range(self.range_list.shape[0]):
            for i in range(self.beaconset.shape[0]):
                val += np.abs(np.linalg.norm(self.beaconset[i,:]-self.pose)
                              - self.range_list[j,i])
        return val * 1.0 / float(self.range_list.shape[0])




