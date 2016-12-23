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

    def __init__(self, settings):
        '''

        '''
        self.para = settings

        self.state_num = 9
        self.observe_num = 4

        self.R = np.zeros([6, 6])
        self.P = np.zeros([9,9])
        self.Q =

        self.x_h = np.zeros([9,1])


    def Navigation_euqtions(self, x_h, u1, quat1,  dt):
        '''

        :type x_h: np.array
        :param x_h:
        :param u1:
        :param u2:
        :param quat1:
        :param quat2:
        :param dt:
        :return:
        '''
        y = np.zeros([9, 1])
        # y = x_h

        w_tb = u1[3:6]
        v = np.linalg.norm(w_tb) * dt

        if math.fabs(v) > 1e-8:
            P = w_tb[0] * dt * 0.5
            Q = w_tb[1] * dt * 0.5
            R = w_tb[2] * dt * 0.5

            OMEGA = np.array([
                [0.0, R, -Q, P],
                [-R, 0.0, P, Q],
                [Q, -P, 0.0, R],
                [-P, -Q, -R, 0.0]
            ])

            q = (math.cos(v / 2.0) * np.diagflat([1.0, 1.0, 1.0, 1.0]) +
                 2.0 / v * math.sin(v / 2.0) * OMEGA).dot(quat1)
            q = q / np.linalg.norm(q)
        else:
            q = quat1

        ####################



        g_t = np.array([0, 0, 9.8173])
        g_t = np.transpose(g_t)

        # use rotation transform form imu to word
        Rb2t = self.q2dcm(q)
        f_t = Rb2t.dot(u1[0:3])


        acc_t = f_t + g_t

        A = np.diagflat(([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt

        B = np.zeros([6, 3])

        B[0:3, 0:3] = np.zeros([3, 3])
        B[3:6, 0:3] = np.diagflat([dt, dt, dt])

        # print(acc_t.shape)
        # print(B.dot(acc_t).shape)
        # print(A.dot(x_h[0:6]).shape)
        acc_t = acc_t.reshape(3, 1)

        # accumulate acc and pose.
        y[0:6] = A.dot(x_h[0:6]) + B.dot(acc_t)

        return y, q

    def dcm2q(self, R):
        """
        Transform from rotation matrix to quanternions.
        :param R:old rotation matrix
        :return:quanternion
        """
        T = 1.0 + R[0, 0] + R[1, 1] + R[2, 2]
        # print (T)


        # Really Big Change.
        # ToDo:Why there are some value is smallter than zero.
        if math.fabs(T) > 1e-3:
            S = 0.5 / math.sqrt(math.fabs(T))

            qw = 0.25 / S
            qx = (R[2, 1] - R[1, 2]) * S
            qy = (R[0, 2] - R[2, 0]) * S
            qz = (R[1, 0] - R[0, 1]) * S

        else:
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                S = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0

                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S

            elif R[1, 1] > R[2, 2]:
                S = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0

                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0

                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
                qz = 0.25 * S

        quart = np.array(np.transpose([qx, qy, qz, qw]))

        quart /= np.linalg.norm(quart)

        return quart

    def q2dcm(self, q):
        """

        :param q:
        :return:
        """
        p = np.zeros([6, 1])

        p[0:4] = q.reshape(4, 1) ** 2.0

        p[4] = p[1] + p[2]

        if math.fabs(p[0] + p[3] + p[4]) > 1e-18:
            p[5] = 2.0 / (p[0] + p[3] + p[4])
        else:
            p[5] = 0.0

        R = np.zeros([3, 3])

        R[0, 0] = 1 - p[5] * p[4]
        R[1, 1] = 1 - p[5] * (p[0] + p[2])
        R[2, 2] = 1 - p[5] * (p[0] + p[1])

        p[0] = p[5] * q[0]
        p[1] = p[5] * q[1]
        p[4] = p[5] * q[2] * q[3]
        p[5] = p[0] * q[1]

        R[0, 1] = p[5] - p[4]
        R[1, 0] = p[5] + p[4]

        p[4] = p[1] * q[3]
        p[5] = p[0] * q[2]

        R[0, 2] = p[5] + p[4]
        R[2, 0] = p[5] - p[4]

        p[4] = p[0] * q[3]
        p[5] = p[1] * q[2]

        R[1, 2] = p[5] - p[4]
        R[2, 1] = p[5] + p[4]

        return R