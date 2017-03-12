# -*- coding:utf-8 -*-
# carete by steve at  2017 / 03 / 08　17:30
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import DataChronic
if __name__ == '__main__':
    dc = DataChronic.DataChronic("../10-03-04-00-01")
    dc.RunOpenshoe()
    dc.SynData()
    print(dc.UwbData)

