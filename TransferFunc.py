from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt
import mglearn

line=np.linspace(-3,3,100) # 100 x-Werte zwischen -3 und 3
plt.plot(line,np.tanh(line),label="tanh")
plt.plot(line,np.maximum(line,0),label="relu")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("relu(x),tanh(x)")
plt.show()
