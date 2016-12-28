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

        self.R = np.zeros([self.observe_num, self.observe_num])
        self.P = np.zeros([self.state_num, self.state_num])
        self.Q = np.zeros([6, 6])

        self.K = np.zeros([self.state_num, self.observe_num])

        self.x_h = np.zeros([self.state_num])
        self.init_filter()
        self.init_vec(self.P)
        # Just for ukf

    def init_vec(self, P):
        """

        :param P:
        :return:
        """
        self.Id = np.diagflat(np.ones(self.P.shape[0]))
        return

    def init_Nav_eq(self, u1):
        '''
        :param u1:
        :param u2:
        :return:
        '''
        f_u = np.mean(u1[0, :])
        f_v = np.mean(u1[1, :])
        f_w = np.mean(u1[2, :])
        # print(f_u,f_v,f_w)
        roll = math.atan2(-f_v, -f_w)
        pitch = math.atan2(f_u, math.sqrt(f_v ** 2 + f_w ** 2))
        attitude = [roll, pitch, self.para.init_heading1]
        attitude = np.transpose(attitude)
        Rb2t = self.Rt2b(attitude)
        Rb2t = np.transpose(Rb2t)
        quat1 = self.dcm2q(Rb2t)
        x = np.zeros([9])
        x[0:3] = self.para.init_pos1
        x[6:9] = attitude

        self.x_h = x
        self.quat1 = quat1

        return x, quat1

    def init_filter(self):
        print(self.para.sigma_initial_pos ** 2)

        self.P[0:3, 0:3] = np.diagflat(np.transpose(self.para.sigma_initial_pos ** 2.0))
        self.P[3:6, 3:6] = np.diagflat(np.transpose(self.para.sigma_initial_vel ** 2.0))
        self.P[6:9, 6:9] = np.diagflat(np.transpose(self.para.sigma_initial_att ** 2.0))

        self.R = np.diagflat(np.transpose(np.ones([self.R.shape[0]])))
        self.R = self.R * self.para.sigma_initial_range_single

        self.Q[0:3, 0:3] = np.diagflat(np.transpose(self.para.sigma_acc ** 2.0))
        self.Q[3:6, 3:6] = np.diagflat(np.transpose(self.para.sigma_gyro ** 2.0))


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
        y = np.zeros([9])
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

        acc_t = acc_t.reshape(3)

        # accumulate acc and pose.
        y[0:6] = A.dot(x_h[0:6]) + B.dot(acc_t)

        return y, q

    def GetPosition(self, u1, zupt1):

        miu_z = np.zeros([self.x_h.shape[0] + u1.shape[0]])

        sigma_zz = np.zeros([self.P.shape[0] + self.Q.shape[0], self.P.shape[0] + self.Q.shape[0]])

        miu_z[0:9] = self.x_h.reshape(9)
        # miu_z[9:15] = u1.reshape(6)

        sigma_zz[0:9, 0:9] = self.P
        sigma_zz[9:15, 9:15] = self.Q

        # miu_z = (0.5 * miu_z.transpose() + 0.5 * miu_z)

        # miu_z = np.abs(miu_z)

        miu_z = np.sqrt(miu_z*miu_z.transpose())

        # print(sigma_zz)
        L = np.linalg.cholesky(sigma_zz)
        L_num = sigma_zz.shape[0]

        # alpha = np.zeros([1 + 2 * L_num])

        ka = 2.0

        miu_z_list = list()
        q_list = list()

        t_z = miu_z
        t_q = self.quat1
        t_z[0:9], t_q = self.Navigation_euqtions(miu_z[0:9], u1, t_q, self.para.Ts)
        miu_z_list.append(t_z)
        q_list.append(t_q)

        for i in range(L_num):
            t_z = miu_z
            t_z[0:9], t_q = self.comp_internal_states(miu_z[0:9], np.sqrt(L_num + ka) * L[0:9, i], self.quat1)
            t_z[0:9], t_q = self.Navigation_euqtions(t_z[0:9], u1 + np.sqrt(L_num + ka) * L[9:15, i], t_q, self.para.Ts)

            miu_z_list.append(t_z)
            q_list.append(t_q)

            t_z = miu_z
            t_z[0:9], t_q = self.comp_internal_states(miu_z[0:9], -np.sqrt(L_num + ka) * L[0:9, i], self.quat1)
            t_z[0:9], t_q = self.Navigation_euqtions(t_z[0:9], u1 - np.sqrt(L_num + ka) * L[9:15, i], t_q, self.para.Ts)

            miu_z_list.append(t_z)
            q_list.append(t_q)

        # sum up
        self.x_h = self.x_h * 0.0
        self.P = self.P * 0.0
        self.quat1 = self.quat1 * 0.0

        self.x_h += (ka / float(ka + L_num)) * miu_z_list[0][0:9]
        self.quat1 += np.sqrt(L_num + ka) * q_list[0]

        for i in range(1, 2 * L_num + 1):
            self.x_h += (1 / (2.0) / (L_num + ka)) * miu_z_list[i][0:9]
            self.quat1 += (1 / 2.0 / (L_num + ka)) * q_list[i]

        #keep quat1
        self.quat1 = self.quat1 / np.linalg.norm(self.quat1)

        self.P += (ka / (ka + L_num)) * (miu_z_list[0][0:9] - self.x_h).dot(
            (miu_z_list[0][0:9] - self.x_h).transpose())

        for j in range(1, 2 * L_num + 1):
            self.P += (1 / 2.0 / (ka + L_num)) * (miu_z_list[j][0:9] - self.x_h).dot(
                (miu_z_list[j][0:9] - self.x_h).transpose()
            )



        # Keep P;

        self.P = (self.P * 0.5 + self.P.transpose() * 0.5)

        return self.x_h










    def comp_internal_states(self, x_in, dx, q_in):
        '''

        :param x_in:
        :param dx:
        :param q_in:
        :param q_in2:
        :return:
        '''

        R = self.q2dcm(q_in)

        x_out = x_in + dx

        epsilon = dx[6:9]
        # print (dx)

        OMEGA = np.array([
            [0, -epsilon[2], epsilon[1]],
            [epsilon[2], 0.0, -epsilon[0]],
            [-epsilon[1], epsilon[0], 0.0]
        ])

        R = (np.diagflat([1.0, 1.0, 1.0]) - OMEGA).dot(R)

        q_out = self.dcm2q(R)

        return x_out, q_out

    '''
    auxilurate function.
    '''
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

    def Rt2b(self, ang):
        '''
        :
        :param ang:
        :return:
        '''
        cr = math.cos(ang[0])
        sr = math.sin(ang[0])

        cp = math.cos(ang[1])
        sp = math.sin(ang[1])

        cy = math.cos(ang[2])
        sy = math.sin(ang[2])

        R = np.array(
            [[cy * cp, sy * cp, -sp],
             [-sy * cr + cy * sp * sr, cy * cr + sy * sp * sr, cp * sr],
             [sy * sr + cy * sp * cr, -cy * sr + sy * sp * cr, cp * cr]]
        )
        return R
