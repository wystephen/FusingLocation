# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 04　15:05


import  pygame

import numpy as np

from ViewerModel.Beacon import BeaconWithRange
from ViewerModel.Agent import Robo
from ViewerModel.PF_FRAME import PF_Frame


from log_process import seq_process

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # --- Clobals ---
    # Colors

    '''
    Select file only according to first several letters.
    '''
    import os
    for dir_name in os.listdir('./'):
        if '04-' in dir_name:
            for the_file_name in os.listdir(dir_name):
                if '.data' in the_file_name:
                    beaconpose = np.loadtxt(dir_name+"/beaconset")
                    se = seq_process()
                    se.process_file(file_name=dir_name + '/'+the_file_name)
                    beacon_range = np.loadtxt("atrange.txt")
                    break
            break
    # print('Range size:',beacon_range.shape)
    beacon_range = beacon_range[:,1:]


    BLACK=(0,0,0)
    WHITE=(255,255,255)

    SCREEN_SIZE=[1680,980]

    OFFSET = [450,450] # piexels

    ScaleFactor = 30.0 #Real(m) to piexels


    pygame.init()

    screen = pygame.display.set_mode(SCREEN_SIZE)

    pygame.display.set_caption("OWN UWB TEST")

    allspriteslit = pygame.sprite.Group()

    clock = pygame.time.Clock()
    done = False

    tmp_beacon = BeaconWithRange(SCREEN_SIZE,OFFSET,ScaleFactor)
    tmp_beacon2 = BeaconWithRange(SCREEN_SIZE,OFFSET,ScaleFactor)
    tmp_beacon3 = BeaconWithRange(SCREEN_SIZE,OFFSET,ScaleFactor)
    tmp_beacon4 = BeaconWithRange(SCREEN_SIZE,OFFSET,ScaleFactor)

    # beaconpose = np.loadtxt("log/beaconset")
    print(beaconpose)
    tmp_beacon.SetPose(beaconpose[0,0],beaconpose[0,1])
    tmp_beacon2.SetPose(beaconpose[1,0],beaconpose[1,1])
    tmp_beacon3.SetPose(beaconpose[2,0],beaconpose[2,1])
    tmp_beacon4.SetPose(beaconpose[3,0],beaconpose[3,1])


    # Data Load

    # se = seq_process
    # se.process_file(se,file_name='log/log.data',out_aa='aarange.txt',out_at='atrange.txt')
    # beacon_range = np.loadtxt("atrange.txt")


    time_step = 0
    '''
        # Data preprocess 1:3d to 2d
    '''

    # z_offset = beaconpose[:,2] - 1.12
    z_offset = np.ones_like(beacon_range)
    z_offset *= 0.01

    # print("beacons 223",beacon_range[223,:])

    # z_offset.reshape([1,3])
    # beacon_range = beacon_range[:,3:6]
    beacon_range = beacon_range**2.0 - z_offset **2.0
    beacon_range = beacon_range ** 0.5
    # print("bbb 223:",beacon_range[223,:])
    beacon_range /= 1000.0
    print("bea range",beacon_range)


    '''
    # Data preprocess 2:gt cut
    '''
    # gt = gt[:,0:2]
    # print("gt shape ", gt.shape)

    '''
    #Error matrix
    '''
    # err = np.zeros(gt.shape[0])

    tmp_beacon.SetRangeMethond(1)
    tmp_beacon2.SetRangeMethond(1)
    tmp_beacon3.SetRangeMethond(1)
    tmp_beacon4.SetRangeMethond(1)

    tmp_robo = Robo(SCREEN_SIZE,OFFSET,ScaleFactor)

    pf = PF_Frame(SCREEN_SIZE,OFFSET,ScaleFactor,1300)

    BeaconSet = np.zeros([4,2])

    BeaconSet[0,:] = tmp_beacon.Pose
    BeaconSet[1,:] = tmp_beacon2.Pose
    BeaconSet[2,:] = tmp_beacon3.Pose
    BeaconSet[3,:] = tmp_beacon4.Pose

    pf.SetBeaconSet(BeaconSet)

    pygame.mouse.set_visible(False)
    pygame.mouse.set_visible(True)

    last_pose = np.zeros(2)

    IsPause = False

    while not done:
        pose = pygame.mouse.get_pos()
        # print("dis:",np.linalg.norm(np.asarray(pose)-last_pose))
        last_pose = np.asarray(pose)

        if time_step == 0:
            pf.InitialPose([0.0,0.0])
        if not IsPause:
            time_step += 1
        if time_step == beacon_range.shape[0]-1:
            time_step = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            elif event.type == pygame.KEYDOWN:
                print(event.key)
                if event.key == 115:
                    pf.InitialPose([((pose[0]-OFFSET[0])*1.0/ScaleFactor),((pose[1]-OFFSET[1])*1.0/ScaleFactor)])

                elif event.key == 100:
                    IsPause = not IsPause
        if IsPause:
            continue

        screen.fill(BLACK)
        '''
        Draw likelihood distribution.
        '''

        tmp_beacon.SetRange(beacon_range[time_step,0])
        tmp_beacon2.SetRange(beacon_range[time_step,1])
        tmp_beacon3.SetRange(beacon_range[time_step,2])
        tmp_beacon4.SetRange(beacon_range[time_step,3])

        tmp_beacon.Draw(screen)
        tmp_beacon2.Draw(screen)
        tmp_beacon3.Draw(screen)
        tmp_beacon4.Draw(screen)

        pf.Sample(0.5)
        pf.Evaluated(beacon_range[time_step,:])

        pf.ReSample()

        result = pf.GetResult()

        pf.Draw(screen)

        pygame.display.flip()

        clock.tick(100)

    pygame.quit()
