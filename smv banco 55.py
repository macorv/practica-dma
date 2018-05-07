# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 09:08:08 2018

@author: Fernando
"""
import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

home=pd.read_excel('C://Users//Fernando//Desktop//homeb.xlsx')

print(home)

home.describe()

home.columns

x=home[['pred','finde', 'feriado', 'posferiado', 'habil',
       'primer_habil', 'ultimo_habil']]
y=home['obs']
print(x)

clf = svm.SVR()
clf.fit(x, y) 
clf = svm.SVR(C=99999999.0,  epsilon=50000, gamma=30,  kernel='rbf' )

clf.fit(x, y)

ee=clf.predict(x)

#%%

t=[i for i in range(0,len(ee))]

 
bg_color = 'black'

fg_color = 'white'

fig = plt.figure(1,figsize=(5,5), facecolor=bg_color, edgecolor=fg_color)

plt.figure(1,figsize=(5,5), axisbg='red')

axes = plt.axes((0.1, 0.1, 0.8, .8), axisbg=bg_color)

axes.xaxis.set_tick_params( color=fg_color, labelcolor=fg_color)

axes.yaxis.set_tick_params( color=fg_color, labelcolor=fg_color)

axes.axhline(y=0, color='w')

axes.axvline(x=0, color='w') 

axes.grid(True, which='both')

 

for spine in axes.spines.values():

    spine.set_color(fg_color)

   

#plot1=plt.plot(ee, y, c='y', linewidth=0.5,axes=axes)

plot2=plt.scatter(ee,y,c='y', marker='.')

plot3=plt.scatter(x['pred'],y, c='w',marker='.')

 

plot2=plt.plot(t, ee, c='y', linewidth=0.5,axes=axes)

plot3=plt.plot(t, y, c='w', linewidth=0.5,axes=axes)

plot3=plt.plot(t, x['pred'], c='r', linewidth=0.5,axes=axes)
