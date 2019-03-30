# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 08:41:50 2019

@author: itm
"""

#from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('E:/VAI92/Code')
#import my_linealT as trans 


x = [0, 0, 10, 18, 25, 3, -2, -14, -38, 16, 25, 44, 38, 2, -10, 1, 0, 1]
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for n in range(len(y)):
    y[n] = x[n] + 0.7*x[n - 1] - 0.4*x[n - 2]
    print(y[n])
