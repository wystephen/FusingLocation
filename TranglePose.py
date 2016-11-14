# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 14　11:12

import numpy as np
from scipy.optimize import minimize


class tranglepose:
    def __init__(self, beaconset, range_list):
        self.pose = [5.2, -0.4, 2.0]
        self.beaconset = beaconset
        self.range_list = range_list
        print("size of beaconset", self.beaconset.shape, self.range_list.shape)

        print("cost func:", self.costfunction_multi_range(self.pose))

        print("range list ", self.range_list)
        res = minimize(self.costfunction_multi_range,
                       self.pose,
                       method='L-BFGS-B',
                       bounds=((-30, 30),
                               (-30, 30),
                               (1.00, 2.4)
                               ),
                       jac=False)

        print(res.x)

    def costfunction_single_range(self, pose):
        val = 0.0
        for i in range(self.beaconset.shape[0]):
            val += np.abs(np.linalg.norm(self.beaconset[i, :] - pose)
                          - self.range_list[i])
        return val

    def costfunction_multi_range(self, pose):
        val = 0.0
        for j in range(self.range_list.shape[0]):
            for i in range(self.beaconset.shape[0]):
                val += np.abs(np.linalg.norm(self.beaconset[i, :] - pose)
                              - self.range_list[j, i])
        print("val:", val
              / float(self.range_list.shape[0])
              / float(self.beaconset.shape[0]))
        return (val * 1.0
               / float(self.range_list.shape[0])
               / float(self.beaconset.shape[0]))
