# -*- coding:utf-8 -*-
# carete by steve at  2016 / 11 / 01　21:28


import serial

t = serial.Serial("com5",115600)

while True:
    print(t.read(10))