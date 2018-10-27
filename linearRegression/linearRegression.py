#-*- coding:utf-8 -*-

import numpy as np
from pylab import *

def train_wb(X, y):
    if np.linalg.det(X.T * X) !=0 :
        wb = ((X.T.dot(X).I).dot(X.T)).dot(y)
        return wb

def test(x, wb):
    return x.T.dot(wb)

def getdata():
    x = []; y = []
    file = open("ex0.txt", 'r')
    for line in file.readlines():
        temp = line.strip().split("\t")
        x.append([float(temp[0]), float(temp[1])])
        y.append(float(temp[2]))
    return(np.mat(x), np.mat(y).T)

def draw(x, y, wb):
    a = np.linspace(0, np.max(x))
    b = wb[0] + a*wb[1]
    plot(x, y, '.')
    plot(a, b)
    show()

X, y = getdata()
wb = train_wb(X,y)
print(wb)
draw(X[:,1], y, wb.tolist())