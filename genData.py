import numpy as np
import matplotlib.pyplot as mp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

#Definition Funktion, X-Bereich
x_range=[]
x_min=-5.0
x_max=5.0
x_delta=0.1

x1=np.arange(x_min,x_max,x_delta) # numpy array erzeugen
y1=[(i**4+5*i**3-7*i**2-6*i+30) for i in x1] #Funktion berechnen
x1=x1.tolist() #numpy array wieder in Liste umwandeln

#mp.plot(x1,y1) # Funtion plotten

data=np.array([x1,y1]).T    #transponierte Matrix aus x1 und y1 generieren:
                            # erste Spalte = x Wert
                            # zweite Spalte = y

# X als numpy array erstellen, 100 Zeilen, 1 Spalte
X=np.array(x1).reshape(100,1)

# Datensatz aufsplitten in Test- und Trainingsdaten
X_train,X_test,y_train,y_test=train_test_split(X,y1)

# einmal Entscheidungsbaum und dann Linear-Regression anlegen und trainieren
tree=DecisionTreeRegressor(max_depth=3).fit(X_train,y_train)
linear_reg=LinearRegression().fit(X_train,y_train)

pred_tree=tree.predict(X)
pred_lr=linear_reg.predict(X)


mp.plot(data[:,0],data[:,1]) #plot aus Daten generieren, erste Spalte X, zweite Spalte y
mp.plot(X,pred_tree)
mp.plot(X,pred_lr)







