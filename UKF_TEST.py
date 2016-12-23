# -*- coding:utf-8 -*-
# carete by steve at  2016 / 12 / 21　17:02
'''
Test UKF.py

'''


if __name__ == '__main__':
    ex_dir_list = list()
    for dir_name in os.listdir('./'):
        if '-0' in dir_name and not '0.' in dir_name:
            ex_dir_list.append(dir_name)
            # if '03-' in dir_name:  # or '-0'in dir_name:
            #     print(dir_name)
            #     location = FusingLocation(dir_name)
            #     location.OnlyPF()
            #     location.Transform()
            #     location.Fusing(200)
            #
            #     plt.show()
            #
            #   3
            #   13
    print(ex_dir_list)
    ex_dir_list.sort()
    print(ex_dir_list)
    import time

    ticks = time.time()
    time_step = list()
    time_step.append(ticks)
    for i in [17]:
        dir_name = ex_dir_list[i]
        print(dir_name)
        # location = FusingLocation(dir_name, [0,1,2])
        # location.OnlyPF()
        # location.Transform()

        # location.Fusing(1000)
        # location.DeepFusing(1000)
        # location.MixFusing(1000, noise_sigma=1.0, evaluate_sigma=3.5)
        # plt.show()
        import DataChronic

        dc = DataChronic.DataChronic(dir_name,beacon_use=[0,1,2,3])

        dc.RunOpenshoe()
        dc.SynData()

        # UwbData[:,1:] = dc.UwbData[:,1:]/1000.0




