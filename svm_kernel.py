from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt
import mglearn

X,y = make_blobs(centers=4,random_state=8)
linear_svc=LinearSVC().fit(X,y)
y=y%2 # 0->0, 1->1, 2->0

line=np.linspace(-7,10) # X-Achse generieren

mglearn.discrete_scatter(X[:,0],X[:,1],y)
#mglearn.plots.plot_2d_separator(linear_svc,X)

plt.plot(line, -(line*linear_svc.coef_[0,0]+linear_svc.intercept_[0])/linear_svc.coef_[0,1],"y+")
plt.plot(line, -(line*linear_svc.coef_[1,0]+linear_svc.intercept_[1])/linear_svc.coef_[1,1],"b+")
plt.plot(line, -(line*linear_svc.coef_[2,0]+linear_svc.intercept_[2])/linear_svc.coef_[2,1],"r+")
plt.plot(line, -(line*linear_svc.coef_[3,0]+linear_svc.intercept_[3])/linear_svc.coef_[3,1],"g+")

plt.xlabel("Merkmal 0")
plt.ylabel("Merkmal 1")
