# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:17:16 2019

@author: user
"""
#圖片轉素描
from PIL import Image
import numpy as np
from scipy import signal

I = Image.open('pic.jpg')
W,H = I.size
data = np.array(I)

R = data[:,:,0]
G = data[:,:,1]
B = data[:,:,2]

Mx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype = int)
My = np.array([[-1,-2,-1],[0,0,0,],[1,2,1]],dtype = int)

R2 = signal.convolve2d(R,Mx,boundary='symm',mode='same')
G2 = signal.convolve2d(G,Mx,boundary='symm',mode='same')
B2 = signal.convolve2d(B,Mx,boundary='symm',mode='same')

R3 = signal.convolve2d(R,My,boundary='symm',mode='same')
G3 = signal.convolve2d(G,My,boundary='symm',mode='same')
B3 = signal.convolve2d(B,My,boundary='symm',mode='same')

R4 = R2**2+R3**2
G4 = G2**2+G3**2
B4 = B2**2+B3**2

S = R4+G4+B4 
i = np.argsort(S.reshape(1,-1))[0]
i1 = i[:288000]
i2 = i[288000:]
R = R.reshape(1,-1)
G = G.reshape(1,-1)
B = B.reshape(1,-1)

R[0][i1] = 255
R[0][i2] = 0
G[0][i1] = 255
G[0][i2] = 0
B[0][i1] = 255
B[0][i2] = 0

R = R.reshape((450,800))
G = G.reshape((450,800))
B = B.reshape((450,800))

data2 = data.copy()
threshold = 50
data2[:,:,0] = R.astype('uint8')
data2[:,:,1] = G.astype('uint8')
data2[:,:,2] = B.astype('uint8')



I2 = Image.fromarray(data2,"RGB")
I2.show()