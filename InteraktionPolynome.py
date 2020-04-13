from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# 04.04.2020

X, y = mglearn.datasets.make_wave(n_samples=100)
plt.figure(1)
plt.plot(X, y, 'bo', label="wave dataset")

line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)  # reshape(-1,1) ist das gleich wie line.flatten()
regTree = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
regLin = LinearRegression().fit(X, y)
plt.plot(line, regTree.predict(line), label="Entscheidungsbaum")
plt.plot (line, regLin.predict(line), label='Lineare Regression')
plt.ylabel("Ausgabe der Regression")
plt.xlabel("Eingasmerkmal ")
plt.legend(loc='best')

# transformation eines "kontinuierlichen Merkmals in ein kategorisches Merkmal" - Diskretisierung/Klassenbildung
bins = np.linspace(-3, 3, 11)  # definiere 10 Klassen zwischen -3 und 3
which_bin = np.digitize(X, bins=bins )  # welcher Datenpunkt von X gehört jetzt zu welcher Klasse
# array([[ 4],
#       [10],
#       [ 8],
#      [ 6],
#       [ 2]], dtype=int32)

encoder = OneHotEncoder(sparse=False, categories='auto')
encoder.fit(which_bin)  # eindeutige Werte finden, die in which_bin auftreten
X_binned = encoder.transform(which_bin)  # Matrix: 10 Kategorien, jede Zeile Datenpunkt gehört zu welcher Kategorie
# array([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  ->(4)
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],  ->(10)
#        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  ->(8)
#        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  ->(6)
#        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]]) ->(2)


plt.figure(2)
line_binned = encoder.transform(np.digitize(line, bins=bins))

reg_binned = LinearRegression().fit(X_binned, y)
reg_binned = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)

plt.plot(line, reg_binned.predict(line_binned), label='Regression mit Binning')
plt.plot(line, reg_binned.predict(line_binned), label='Entscheidungsbaum mit Binning')
plt.plot(X, y, 'bo', label="wave dataset")

# plt.vlines(bins,-3,3,linewidth=1,alpha=0.2)
plt.ylabel("Ausgabe der Regression")
plt.xlabel("Eingasmerkmal")
plt.legend(loc='best')

plt.figure(3)
#X_product=np.hstack([X_binned,X*X_binned])
#plt.show()