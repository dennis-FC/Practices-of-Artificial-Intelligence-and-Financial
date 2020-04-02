# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 23:45:17 2019

@author: user
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import random

iris = datasets.load_iris()
np.random.seed(38)
idx = np.random.permutation(150)
feature = iris.data[idx,:]
target = iris.target[idx]

def kmeans(sample,K,maxiter):
    N = sample.shape[0]
    D = sample.shape[1]
    C = np.zeros((K,D))#k個中心點的座標
    L = np.zeros((N,1))# n個人的label
    L1 = np.zeros((N,1))
    dist = np.zeros((N,K))#距離
    idx = random.sample(range(N),K)
    C = sample[idx,:]
    iter = 0
    while(iter<maxiter):
        #所有sample跟k的中心點的距離
        for i in range(K):
            dist[:,i]=np.sum((sample-np.tile(C[i,:],(N,1)))**2,1)#1為方向(往右)  C[i,:] repeat N遍
        L1 = np.argmin(dist,1)#1為方向 算最小距離  
        if(iter>0 and np.array_equal(L,L1)):#沒更新了
            #print(iter)
            break
        L = L1
        for i in range(K):
            idx = np.nonzero(L==i)[0]
            if(len(idx)>0):
                C[i,:] = np.mean(sample[idx,:],0)#得到新的中心點 0為方向往下
        iter+=1
#        G1 = sample[L==0,:]
#        G2 = sample[L==1,:]
#        G3 = sample[L==2,:]
#        plt.plot(G1[:,0],G1[:,1],'r.',G2[:,0],G2[:,1],'g.',G3[:,0],G3[:,1],'b.',C[:,0],C[:,1],'kx')
#        plt.show()
    wicd = np.sum(np.sqrt(np.sum((sample-C[L,:])**2,1)))#每個人與各自中心點的距離，然後加總
    return C,L,wicd

def knn(test,train,target,k):#資料，字典，人的label，幾個
    N = train.shape[0]
    dist = np.sum((np.tile(test,(N,1))-train)**2,1)
    idx = sorted(range(len(dist)),key=lambda i:dist[i])[0:k]#排序拿index出來
    return Counter(target[idx]).most_common(1)[0][0] #預測出的結果

G = feature
C,L,wicd = kmeans(G,3,1000)
G1 = G[L==0,:]
G2 = G[L==1,:]
G3 = G[L==2,:]
plt.plot(G1[:,0],G1[:,1],'r.',G2[:,0],G2[:,1],'g.',G3[:,0],G3[:,1],'b.',C[:,0],C[:,1],'kx')
print("未經標準化:",wicd)

GA = (G-np.tile(np.mean(G,0),(G.shape[0],1)))/np.tile(np.std(G,0),(G.shape[0],1))#0,1為方向
C,L,wicd = kmeans(GA,3,1000)
C = C*np.std(feature,0)+np.mean(feature,0)#還原回正規化前
wicdGA = np.sum(np.sqrt(np.sum((G-C[L,:])**2,1)))#
print("standard score方法:",wicdGA)

GB = (G-np.tile(np.min(G,0),(G.shape[0],1)))/(np.tile(np.max(G,0),(G.shape[0],1))-np.tile(np.min(G,0),(G.shape[0],1)))
C,L,wicd = kmeans(GB,3,1000)
C = C*(np.max(feature,0)-np.min(feature,0))+np.min(feature,0)#還原回正規化前
wicdGB = np.sum(np.sqrt(np.sum((G-C[L,:])**2,1)))#算wicd
print("scaling方法:",wicdGB)



for i in range(10):
    print("k=",i+1)
    pred = []
    table = [[0,0,0],
             [0,0,0],
             [0,0,0]]
    for batch in range(0,150,1):#每次拿一筆當作test資料
            
            X_train = np.delete(feature, range(batch, batch+1), axis=0)
            y_train = np.delete(target, range(batch,batch+1), axis=0)
            X_test = feature[batch:batch+1,:]
            y_test = target[batch:batch+1]
            
            a = knn(X_test,X_train,target,i+1)
            pred.append(a)
            a = target[batch]
            b = pred[batch]
            table[a][b] = table[a][b]+1 #confusion matrix 

    print(pd.DataFrame(table))
    #print(pred)       
       
                
