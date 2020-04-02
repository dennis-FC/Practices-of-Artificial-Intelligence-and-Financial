# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:05:02 2019

@author: user
"""

from PIL import Image
import numpy as np
from scipy import signal
#圖片雜訊用高斯模糊變圓潤
I = Image.open('pic.jpg')
W,H = I.size
data = np.asarray(I)
data2 = data.copy()
data = data.astype('float64')
noise = np.random.normal(0,20,(H,W,3))

data3 = data+noise;
data3[data3>255] = 255
data3[data3<0] = 0
data2 = data3.astype('uint8')
#I2 = Image.fromarray(data2,"RGB")
#I2.show()

x,y = np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
d = np.sqrt(x*x+y*y)
sigma,mu = 0.5,0.0
M = np.exp(-((d-mu)**2/(2.0*sigma**2)))
M = M/np.sum(M[:])
R = data2[:,:,0]
G = data2[:,:,1]
B = data2[:,:,2]
R2 = signal.convolve2d(R,M,boundary='symm',mode='same')
G2 = signal.convolve2d(G,M,boundary='symm',mode='same')
B2 = signal.convolve2d(B,M,boundary='symm',mode='same')
data4 = data2.copy()
data4[:,:,0] = R2.astype('uint8')
data4[:,:,1] = G2.astype('uint8')
data4[:,:,2] = B2.astype('uint8')

I2 = Image.fromarray(data4,"RGB")
I2.show()


