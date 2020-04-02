# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import math

NN = np.load('data.npy')

def F3(t,A,B,C,beta,w,tc,phi):
    return A+(B*np.power(abs(tc - t),beta))*(1 + (C*np.cos((w *np.log(abs(tc-t)))+phi)))
#def F3(A,B,C,beta,w,tc,phi):
#    signal = np.zeros((1,tc))
#    for i in range(tc):
#        signal[0,i] = A+(B*np.power(tc - 1151,np.random.random.normal(0,1)))*(1 + (C*np.cos((w *np.log(tc-t))+phi)))
#    return signal

def E(t,b,A,B,C,beta,w,tc,phi):#t可能為向量 #和適度函數
    return np.sum(abs(b - F3(t,A,B,C,beta,w,tc,phi)))


n = 1200
ttc = random.randint(1151,1166)
print("ttc=",ttc)
b = np.log(NN[:1200].reshape((1200,1)))
t = np.zeros((n,1))
for i in range(n):
    t[i] = np.random.random()*100
#    b[i] = NN[i]

A = np.zeros((n,3))
for i in range(n):
    #for j in range(1151,1166):
    if (i!=ttc):
        A[i,0] = 1
        A[i,1] = np.power(abs(ttc-i),random.random())
        A[i,2] = (np.cos(random.uniform(5,15)*(np.log(abs(ttc-i)))+random.uniform(0,math.pi)))*np.power(abs(ttc-i),random.random())
    else:
        A[i,0] = 1
        A[i,1] = np.power(abs(ttc-i),random.random())
        A[i,2] = 0
        

x = np.linalg.lstsq(A,b)[0]
print(x)
x[2] = x[2]/x[1]

pop = np.random.randint(0,2,(10000,21))#一萬個人每個人有四十個基因(40個1 or 0)
fit = np.zeros((10000,1))

for generation in range(20):#每個世代
    print(generation)
    for i in range(10000):#算一萬個人適者還是不適者
        gene = pop[i,:]#第i個人的基因
        beta = (np.sum(2**np.array(range(4))*gene[:4]))/15;#二進位轉十進位
        w = (np.sum(2**np.array(range(3))*gene[4:7])+5);
        tc = (np.sum(2**np.array(range(4))*gene[7:11])+1151);
        phi = abs(np.sum(2**np.array(range(10))*gene[11:21])-395)/100;
        fit[i] = E(t,b,x[0],x[1],x[2],beta,w,tc,phi)
    sortf = np.argsort(fit[:,0])#最適者是第零個
    #print("sortf",sortf)
    pop = pop[sortf,:]
    #print("pop:",pop)
    for i in range(100,10000):#殺掉後面只留前一百個人，再用前一百人生成一萬人(交配)
        fid = np.random.randint(0,100)#產生爸爸
        mid = np.random.randint(0,100)#產生媽媽
        while mid==fid:#避免爸爸和媽媽一樣(因為沒意義)
            mid = np.random.randint(0,100)
        mask = np.random.randint(0,2,(1,21))
        son = pop[mid,:]#媽媽給兒子
        father = pop[fid,:]
        son[mask[0,:]==1] = father[mask[0,:]==1]#1改爸爸的
        pop[i,:] = son
    for i in range(1000):#突變0-->1 1-->0
        m = np.random.randint(0,10000)
        n = np.random.randint(0,21)
        pop[m,n]=1-pop[m,n]#第m個人的第n個基因會被改調
    
    
    
#    b = np.zeros((1200,1))
#    t = np.zeros((1200,1))
#    for i in range(1200):
#        t[i] = np.random.random()*100
##        b[i] = NN[i]

    AA = np.zeros((1200,3))
    b = np.log(NN[:1200].reshape((1200,1)))
    for i in range(1200):
    #for j in range(1151,1166):
        if (i!=tc):
            AA[i,0] = 1
            AA[i,1] = abs(tc-i)**beta
            AA[i,2] = (np.cos(w*(np.log(abs(tc-i)))+phi))*abs(tc-i)**beta
        else:
            AA[i,0] = 1
            AA[i,1] = abs(tc-i)**beta
            AA[i,2] = 0

    x = np.linalg.lstsq(AA,b)[0]
    #print(x)
    x[2] = x[2]/x[1]
        

for i in range(10000):#再排序一次 因為上面最後步驟是突變順序會有錯
    gene = pop[i,:]
    beta = (np.sum(2**np.array(range(4))*gene[:4]))/15;#二進位轉十進位
    w = (np.sum(2**np.array(range(3))*gene[4:7])+5);
    tc = (np.sum(2**np.array(range(4))*gene[7:11])+1151);
    phi = abs(np.sum(2**np.array(range(10))*gene[11:21])-395)/100;
    fit[i] = E(t,b,x[0],x[1],x[2],beta,w,tc,phi)
sortf = np.argsort(fit[:,0])
pop = pop[sortf,:]

gene = pop[0,:]#活最好的那個人
beta = (np.sum(2**np.array(range(4))*gene[:4]))/15;#二進位轉十進位
w = (np.sum(2**np.array(range(3))*gene[4:7])+5);
tc = (np.sum(2**np.array(range(4))*gene[7:11])+1151);
phi = abs(np.sum(2**np.array(range(10))*gene[11:21])-395)/100;
print('beta:',beta,'w:',w,'tc:',tc,'phi:',phi)

ANS = []
for i in range(1200):
    ANS.append((F3(i,x[0],x[1],x[2],beta,w,tc,phi)))

plt.plot(ANS)
plt.plot(np.log(NN[:1200]))
print("A=",x[0],"B=",x[1],"C=",x[1])
print("beta=",beta,"w=",w,"tc=",tc,"phi=",phi)