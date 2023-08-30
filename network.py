from distutils import ccompiler
import os, math
import sys
from tkinter.tix import CheckList
import numpy as np
import pandas as pd
import openmatrix as omx
from scipy.special import perm
from argparse import Namespace
import oslom
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# root
root = os.path.dirname(os.path.abspath('C://Users//zehan//Desktop//network//TransportationNetworks-master//README'))
#root = os.listdir('C://Users//zehan//Desktop//network//TransportationNetworks-master')
#print(root)

# all folders list - check root
folders = [x for x in os.listdir(root)[1:] if os.path.isdir(os.path.join(root, x))]
#print(folders)

# Importing the networks into a Pandas dataframe consists of a single line of code
# but we can also make sure all headers are lower case and without trailing spaces

netfile = os.path.join(root, 'SiouxFalls','SiouxFalls_net.tntp')
net = pd.read_csv(netfile, skiprows=8, sep='\t')

trimmed= [s.strip().lower() for s in net.columns]
net.columns = trimmed

# And drop the silly first andlast columns
net.drop(['~', ';'], axis=1, inplace=True)
print(net.head())

matrixforcsv = np.zeros((24,24))
netforcsv = np.zeros((76,3))
for i in range(net.shape[0]):
    m,n=net.loc[i,'init_node'], net.loc[i,'term_node']
    matrixforcsv[n-1,m-1]=net.loc[i,'length']
    matrixforcsv[m-1,n-1]=net.loc[i,'length']

np.savetxt('matrix.csv', matrixforcsv, delimiter = ',')

for i in range(net.shape[0]):
    m,n=net.loc[i,'init_node'], net.loc[i,'term_node']
    cap=net.loc[i,'length']
    netforcsv[i-1,0]=m
    netforcsv[i-1,1]=n
    netforcsv[i-1,2]=cap

np.savetxt('net.csv', netforcsv, delimiter = ',')


# calculate degree - average in-dgree and average out-degree
degree = net.shape[0]/24
print('degree is', degree)
kin = degree
kout = degree

# calculate clustering coefficient
def calculatecc(net):
    neighborEdge=[0 for i in range(24)]
    ki=0
    for i in range(1, 24):
        neib=[]
        for j in range(net.shape[0]):
            if net.iloc[j,0]==i:
                ki+=1
                n=net.iloc[j,1]
                neib.append(n)
        for k in range(net.shape[0]):
            if net.iloc[k,0] in neib and net.iloc[k,1] in neib:
                neighborEdge[i-1]+=1
    Ai=sum(neighborEdge)
    cc=Ai/(ki*(ki-1)/2)
    return cc

cc=calculatecc(net)
print('clustering coefficient is', cc)

# calculate closeness centrality
cce=23/3172
print('closeness centrality is', cce)

# calculate average path length
L=(1/(24*23))*3172
print('average path length is', L)

# calculate efficiency
e=(1/(24*23))*(32.5222)
print('efficiency is', e)

# plot1: P(K>k) degree
x = [1, 2, 3, 4, 5]
y = [0, 0, 0, 0, 0]
def calPlot1(net):
    degreeList=[0 for i in range(24)]
    for i in range(1, 24):
        for j in range(net.shape[0]):
            if net.iloc[j,0]==i or net.iloc[j,1]==i:
                degreeList[i-1]+=1
    for n in range(24):
        degreeList[n]=degreeList[n]/2
    return degreeList

degreeList=calPlot1(net)
print('degreeList is ',degreeList)
logy=[0,0,0,0]
for i in range(5):
    sum=0
    for k in range(24):
        if degreeList[k] > x[i]:
            sum+=1
    y[i]=sum/24
    if i!=4: logy[i]=math.log(y[i])   

#plt.plot([math.log(1),math.log(2),math.log(3),math.log(4)],logy)
# plt.plot(x,y)
# plt.title('In-degree and Out-degree distributions of urban road network in the linear scale')
# plt.ylabel('Cumulative probability')
# plt.xlabel('In-dgree or Out-degree')
# plt.grid()
# plt.show()

# regression
# def target_func(x,a):
#     return x**(-a)
def target_func(x,a):
    return e**(a*x)

popt, pcov=curve_fit(target_func,x,y)
# Cal R^2
calc_y=[target_func(i, popt[0]) for i in x]
res_y=np.array(y)-np.array(calc_y)
ss_res=np.sum(res_y**2)
ss_tot=np.sum((y-np.mean(y))**2)
r_square=1-(ss_res/ss_tot)

print ("a= %f  R2= %f"%(popt[0],r_square))

# calculate each node's clustering coefficient
def calculateccList(net):
    neighborEdge=[0 for i in range(24)]
    ccList=[0 for i in range(24)]
    for i in range(1, 24):
        ki=0
        neib=[]
        for j in range(net.shape[0]):
            if net.iloc[j,0]==i:
                ki+=1
                n=net.iloc[j,1]
                neib.append(n)
        for k in range(net.shape[0]):
            if net.iloc[k,0] in neib and net.iloc[k,1] in neib:
                neighborEdge[i-1]+=1
        Ai=(neighborEdge[i-1])
        ccList[i-1]=Ai/(ki*(ki-1)/2)
    return ccList
ccList=calculateccList(net)
print('cclist is ',ccList)

#plot2 CC
x2=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
y2=[0,0,0,0,0,0,0,0,0,0]
logy2=[0,0,0,0,0,0,0,0,0,0]
def calPlot2(net):
    for i in range(10):
        sum=0
        for j in range(24):
            if ccList[j]>x2[i]:
                sum+=1
        y2[i]=sum/24
    return y2
y2=calPlot2(net)
print(y2)
for i in range(10):
    if y2[i]!=0: 
        logn=math.log(y2[i])
    logy2[i]=logn
# plt.plot(x2,logy2)
# plt.title('Clustering coefficient of networks')
# plt.ylabel('Log Cumulative probability')
# plt.xlabel('Clustering coefficient')
# plt.grid()
# plt.show()

#plot3  WCC

wccList=[0.0476190476190476,
0.0400000000000000,
0.0588235294117647,
0.0625000000000000,
0.0555555555555556,
0.0500000000000000,
0.0666666666666667,
0.0555555555555556,
0.0588235294117647,
0.0714285714285714,
0.100000000000000,
0.0769230769230769,
0.100000000000000,
0.166666666666667,
0.125000000000000,
0.0666666666666667,
0.0769230769230769,
0.0769230769230769,
0.0909090909090909,
0.111111111111111,
0.333333333333333,
0.200000000000000,
0.500000000000000,
0]

x3=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
y3=[0,0,0,0,0,0,0,0,0,0]
logy3=[0,0,0,0,0,0,0,0,0,0]
def calPlot3(net):
    for i in range(10):
        sum=0
        for j in range(24):
            if wccList[j]>x3[i]:
                sum+=1
        y3[i]=sum/24
    return y3
y3=calPlot3(net)
print(y3)
for i in range(10):
    if y3[i]!=0: 
        logn=math.log(y3[i])
    logy3[i]=logn
print(logy3)
# plt.plot(x3,logy3)
# plt.title('Weighted Closeness of networks')
# plt.ylabel('Log Cumulative probability')
# plt.xlabel('Weighted Closeness')
# plt.grid()
# plt.show()

#plot4 correlation
#cc and degree
plt.scatter(degreeList,ccList)
plt.xlim((-1,6))
plt.ylim((-0.1,1))
plt.show()