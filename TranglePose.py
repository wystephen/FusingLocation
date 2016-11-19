# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 14　11:12

import numpy as np
from scipy.optimize import minimize


class trianglepose:
    def __init__(self, beaconset, range_list):
        self.pose = np.array([5.2, -0.4, 1.12])
        self.beaconset = beaconset
        self.range_list = range_list
        # print("size of beaconset", self.beaconset.shape, self.range_list.shape)

        # print("cost func:", self.costfunction_multi_range(self.pose))

        # print("range list ", self.range_list)
        res = minimize(self.costfunction_multi_range,
                       self.pose,
                       method='L-BFGS-B',
                       bounds=((-30, 30),
                               (-30, 30),
                               (1.00, 3.0)
                               ),
                       jac=False)

        print(res.x)
        self.pose = res.x

    def costfunction_single_range(self, pose):
        val = 0.0
        for i in range(self.beaconset.shape[0]):
            val += np.abs(np.linalg.norm(self.beaconset[i, :] - pose)
                          - self.single_range_list[i])
        return val

    def costfunction_multi_range(self, pose):
        val = 0.0
        for j in range(self.range_list.shape[0]):
            for i in range(self.beaconset.shape[0]):
                val += np.abs(np.linalg.norm(self.beaconset[i, :] - pose)
                              - self.range_list[j, i])
        return (val * 1.0
                / float(self.range_list.shape[0])
                / float(self.beaconset.shape[0]))

    def ComputePath(self, uwbdata):
        initial_pose = [0.0, 0.0, 0.0]
        OptResult = np.zeros([uwbdata.shape[0], 4])

        for i in range(OptResult.shape[0]):
            self.single_range_list = uwbdata[i, 1:]
            res = minimize(self.costfunction_single_range,
                           initial_pose,
                           # method='L-BFGS-B',
                           bounds=((-40, 40),
                                   (-40, 40),
                                   (1.0, 3.0)),
                           jac=False)
            initial_pose = res.x
            OptResult[i, 0] = uwbdata[i, 0]
            OptResult[i, 1:] = res.x

        return OptResult

    def TriComputePath(self, uwbdata):
        TriResult = np.zeros([uwbdata.shape[0], 4])

        for i in range(TriResult.shape[0]):
            TriResult[i, 0] = uwbdata[i, 0]
            TriResult[i, 1:] = self.Trilateration(
                self.beaconset[0:3, :],
                uwbdata[i, 1:]
            )

    def Trilateration(self, beaconset, r):
        '''

        :param r: Range to each beacon.
        :return:
        '''

        ex = beaconset[1, :] - beaconset[0, :]

        h = np.linalg.norm(ex)

        ex = ex / np.linalg.norm(ex)

        t1 = beaconset[2, :] - beaconset[0, :]
        i = ex.dot(t1)
        t2 = ex * i

        ey = t1 - t2
        t = np.linalg.norm(ey)

        if t > 0.0:
            ey = ey / t
            j = ey.dot(t1)
        else:
            j = 0.0

        ez = np.cross(ex, ey)

        r1 = r[0]
        r2 = r[1]
        r3 = r[3]

        x = (r1 * r1 - r2 * r2) / (2 * h) + h / 2.0
        y = (r1 * r1 - r3 * r2 + i * i) / (2.0 * j) + j / 2.0 - x * i / j
        z = r1 * r1 - x * x - y * y

        return np.asarray([x, y, z])
