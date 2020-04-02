# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:16:13 2019

@author: user
"""
#用高斯模糊進行圖片模糊
from PIL import Image
import numpy as np
from scipy import signal
I = Image.open('pic.jpg')
W,H = I.size
data = np.asarray(I)
x,y = np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
d = np.sqrt(x*x+y*y)
sigma,mu = 1,0.0
M = np.exp(-((d-mu)**2/(2.0*sigma**2)))
M = M/np.sum(M[:])
R = data[:,:,0]
G = data[:,:,1]
B = data[:,:,2]
R2 = signal.convolve2d(R,M,boundary='symm',mode='same')
G2 = signal.convolve2d(G,M,boundary='symm',mode='same')
B2 = signal.convolve2d(B,M,boundary='symm',mode='same')
data2 = data.copy()
data2[:,:,0] = R2.astype('uint8')
data2[:,:,1] = G2.astype('uint8')
data2[:,:,2] = B2.astype('uint8')

I2 = Image.fromarray(data2,"RGB")
I2.show()