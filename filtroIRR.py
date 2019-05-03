

#from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('E:/VAI92/Code')
#import my_linealT as trans 


x = [0, 0, 0, 0, 0, 10, 18, 25, 3, -2, -14, -38, 16, 25, 44, 38, 2, -10, 1, 0, 1]
y = [0, 0, 0, 0, 0,  0,  0,  0, 0,  0,   0,  0,   0,  0,  0,  0, 0,  0,  0, 0, 0]

for n in range(len(y)):
    y[n] = (x[n] + x[n-1] + x[n-2] + x[n-3] + x[n-4])/5
    print(y[n])
