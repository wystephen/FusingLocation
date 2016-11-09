# -*- coding:utf-8 -*-
# carete by steve at  2016 / 10 / 12　10:25

import demjson
import numpy as np

class seq_process:
    def __init__(self):
        a=1
        # print("ini")
        # self.aarange = open('aarange.txt', 'w')
        # self.atrange = open('atrange.txt', 'w')
        #
        # self.aaseq = 0
        # self.atseq = 0
        #
        # self.aadis = np.zeros([3])
        # self.atdis = np.zeros([4])
    def file_pre_process(self,name):
        # print("frame")
        import os
        fa = open(name)
        tmp_log = open("log.data", 'w')

        all_file = fa.readline()
        print(len(all_file))

        last_i = 0
        for i in range(len(all_file)):
            # print(i)
            if all_file[i] == '\\' and all_file[i + 1] == 'n':
                tmp_log.write(all_file[last_i:i].replace('\"b\"','') + '\n')
                last_i = i + 2
                # print("new line")
        tmp_log.close()

    def process_file(self,file_name='LOG_2016_10_12_10_15_17.data',out_aa='aarange.txt',out_at='atrange.txt'):

        self.file_pre_process(file_name)
        logf = open('log.data', 'r')

        aarange = open(out_aa, 'w')
        atrange = open(out_at, 'w')

        aaseq = 0
        atseq = 0

        aadis = np.zeros([3])
        atdis = np.zeros([4])

        logf_all = logf.readlines()

        for ll in logf_all:
            if ll[0] == 'I' or  True:
                start = ll.find('{')
                if start == -1:
                    # print("ll is :",ll,"can not found a {")
                    return True
                ll = ll[start::]
            # print(ll)
            jdata = demjson.decode(ll)

            if jdata['type'] == 'a':
                if not (jdata['seq'] == aaseq):
                    if not (aaseq == 0):
                        aarange.write("{0} {1} {2}\n".format(aadis[0], aadis[1], aadis[2]))
                    aaseq = jdata['seq']

                if jdata['aid'] == 0:
                    if jdata['bid'] == 1:
                        aadis[0] = jdata['range']
                    else:
                        aadis[1] = jdata['range']
                else:
                    aadis[2] = jdata['range']

            elif jdata['type'] == 'c':
                # print(jdata['time'])
                if (not (jdata['seq'] == atseq)):
                    if not atseq == 0:
                        atrange.write("{0} {1} {2} {3} {4}\n".format(jdata['time'],atdis[0], atdis[1], atdis[2], atdis[3]))
                    atseq = jdata['seq']

                atdis[jdata['beacon_id']] = jdata['range']
            else:
                print("ERROR")

'''
Some error when use python 3.5 collectoed data.
'''
if __name__ == '__main__':
    se = seq_process()
    se.process_file(file_name='10-03-04-00-01/LOG_2016_11_7_15_50_12.data')