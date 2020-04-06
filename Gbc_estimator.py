import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
X,y=make_circles(noise=0.25,factor=0.5,random_state=1)

y_named=np.array(["blue","red"])[y] # der 0llen und 1en werden blue und red zugewiesen!!

X_train,X_test,y_train_named,y_test_named,y_train,y_test=train_test_split(X,y_named,y,random_state=0)
gbrt=GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train_named)