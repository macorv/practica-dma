
import numpy as np

import pandas as pd

import seaborn.apionly as sns

import matplotlib.pyplot as plt

 

from sklearn.cross_validation import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVR

 

from sklearn.model_selection import train_test_split

home_banking = pd.read_excel(r'/Users/martin/Documents/GitHub/practica-dma/datasets/homeb.xlsx')

 

from sklearn import svm

home_banking.columns

X=home_banking[[ 'pred','finde', 'feriado','posferiado', 'habil', 'primer_habil', 'ultimo_habil']]

y=home_banking['obs']

 

 

clf=svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=10000, gamma=1,kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

clf.fit(X, y)

ee=clf.predict(X)


#%%

 

 

 


#Genero el tiempo

 

t=[i for i in range(0,len(ee))]

 

 

 

 

 

bg_color = 'white'

fg_color = 'black'

fig = plt.figure(1,figsize=(5,5),facecolor=bg_color, edgecolor=fg_color)

plt.figure(1,figsize=(5,5))

axes = plt.axes((0.1, 0.1, 0.8, .8), axisbg=bg_color)

axes.xaxis.set_tick_params(color=fg_color, labelcolor=fg_color)

axes.yaxis.set_tick_params(color=fg_color, labelcolor=fg_color)

axes.axhline(y=0, color='w')

axes.axvline(x=0, color='w') 

axes.grid(True, which='both')

 

for spine in axes.spines.values():

    spine.set_color(fg_color)

   

#plot1=plt.plot(ee, y, c='y', linewidth=0.5,axes=axes)

#plot2=plt.scatter(ee,y,c='y',marker='.')

#plot3=plt.scatter(X['pred'],y,c='b',marker='.')

 

plot2=plt.plot(t, ee, c='y', linewidth=0.5,axes=axes)

plot3=plt.plot(t, y, c='b', linewidth=0.5,axes=axes)

plot4=plt.plot(t, X['pred'], c='r', linewidth=0.5,axes=axes)
