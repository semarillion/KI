import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


KNN_values=np.array([0,0,0])
MAX_KNN=8
y_delta=[]

Score_Lasso_train=[]
Score_Lasso_test=[]
Ridge_train=[]
Ridge_test=[]
score_MLP_train=[]
score_MLP_test=[]
score_train_GBR=[]
score_test_GBR=[]
rate_diff=[]
#auto = pd.read_csv("cars_1.csv")
#pd_auto = pd.read_csv("cars_1.csv",sep=";",decimal=",",names=["Leistung","Leistungsgewicht",\
#                                                                    "Drehmoment","Beschleunigung","Geschwindigkeit"])

# Tabelle mit Grundaten aus Karten
pd_auto_2 = pd.read_csv("cars_2.csv",sep=";",decimal=",",names=["Type","Leistung","Leistungsgewicht",\
                                                                    "Drehmoment","Beschleunigung","Geschwindigkeit"])
np_autos=np.array(pd_auto_2)

# Tabelle mit Grunddaten + möglichem cw-Wert
pd_auto_3 = pd.read_csv("cars_3.csv",sep=";",decimal=",",names=["Type","Leistung","Leistungsgewicht",\
                                                                    "Drehmoment","Beschleunigung","cw-Wert","Geschwindigkeit"])
np_autos_3=np.array(pd_auto_3)


X=np_autos_3[:,1:6] # Eingangsvektor, Spalten 1-5 (mit cw Wert)
y=np_autos_3[:,6:]  # Ausgangsvektor , Spalte 6

# Trainingsdaten und Testdaten, jeweils für X und X_nn ermitteln
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42) # Daten in Trainings- und Testdaten aufteilen

#Clustering mittels kmeans
kmeans=KMeans(n_clusters=5).fit(X)


#KNN
for i in range(1,MAX_KNN):
    index=i-1
    knn_reg=KNeighborsRegressor(n_neighbors=i).fit(X_train,y_train)
    KNN_act=np.array([i,knn_reg.score(X_train,y_train),knn_reg.score(X_test,y_test)]) # Ergebnisse im np speichern
    KNN_values=np.vstack((KNN_values,KNN_act))                                         # und dann stapeln

KNN_values = np.delete(KNN_values, (0), axis=0) # erste anglegte Zeile wieder löschen



# Linear regression
LinReg=LinearRegression().fit(X_train,y_train)

# Ridge Regression
for i in range (0,6):
    ALPHA = 10/(10 ** i)
    RidReg=Ridge(alpha=ALPHA).fit(X_train,y_train)
    Ridge_train.append([ALPHA,RidReg.score(X_train,y_train)])
    Ridge_test.append([ALPHA,RidReg.score(X_test,y_test)])


# Lasso
for i in range(0,6):
    ALPHA=10/(10**i)
    lasso_reg=Lasso(alpha=ALPHA,max_iter=100000).fit(X_train,y_train)
    Score_Lasso_train.append([ALPHA,lasso_reg.score(X_train,y_train)])
    Score_Lasso_test.append([ALPHA,lasso_reg.score(X_test,y_test)])

# MLP
x_scaler=MinMaxScaler(feature_range=(0.1,0.9)) # Min/Max scaler auswählen
y_scaler=MinMaxScaler(feature_range=(0.1,0.9)) # Min/Max scaler auswählen

x_scaler.fit(X_train)                       # x-scaler mit trainingsdaten fitten
X_train_nn = x_scaler.transform(X_train)    # und auf trainisdaten als auch
X_test_nn = x_scaler.transform(X_test)      # test-daten anwenden

y_scaler.fit(y_train)                           # das gleich dann mit y
y_train_nn=y_scaler.transform(y_train).ravel()  # trainingsdaten und dann "glätten"
y_test_nn=y_scaler.transform(y_test).ravel()    # Test-Daten

for NODES in range(10,100,2):
    MLP_reg=MLPRegressor(hidden_layer_sizes=(NODES,2),max_iter=1000,solver="lbfgs",random_state=42,\
                         activation="tanh",alpha=0.01,learning_rate_init=0.01).fit(X_train_nn,y_train_nn)
    score_MLP_train.append([NODES,MLP_reg.score(X_train_nn,y_train_nn)])    # scores abspeichern
    score_MLP_test.append([NODES,MLP_reg.score(X_test_nn,y_test_nn)])       # scores für test daten abspeichern

#maximalen score über die nodes berechnen, für trainings- und Testdaten
node,rate_train=zip(*score_MLP_train)   # entpacken - nodes und scores der Trainingsdaten
node,rate_test=zip(*score_MLP_test)     # entpacken - nodes und scores der Testdaten
for i in range(0,len(rate_train)):
    rate_diff.append(rate_train[i]-rate_test[i])    # Liste der Differenzen generieren - später Darstellung mit plt

#Berechnung zur Darstellung Balkendiagramme, wieviel er beim Schätzen daneben liegt
inx=0
for idx in range(0,len(X_test_nn)):
    y_diff=y_test[idx]-y_scaler.inverse_transform([MLP_reg.predict([X_test_nn[idx]])]).ravel()
    y_delta.append(y_diff.tolist()[0])

# GradientBootRegressor
for i in range(0,20):
    ALPHA=1/(10**i)
    if ALPHA>=1.0:
        ALPHA=0.99
    TREES=10+20*i
    GBR=GradientBoostingRegressor(alpha=ALPHA,n_estimators=20).fit(X_train,y_train.ravel())
    score_train_GBR.append([ALPHA,GBR.score(X_train,y_train.ravel())])
    score_test_GBR.append([ALPHA,GBR.score(X_test,y_test.ravel())])


# plot data
# KNN
plt.figure(1)
plt.plot(KNN_values[:,0],KNN_values[:,1],"g")   # Trainingsdaten
plt.plot(KNN_values[:,0],KNN_values[:,2],"r")   # Testdaten
plt.axis( [1,MAX_KNN-1,0.5,1])                      # Achsen Bereiche, x,y
plt.grid(True)                              # Gitter einschalten
plt.legend(("Trainingsdaten","Testdaten"))  # Legende
plt.xlabel("KNN")
plt.ylabel("score [%]")
plt.title("k-nearest-neighbour (KNN)")
plt.show()


# MLP
x_train_data, y_train_data = zip(*score_MLP_train)
x_test_data, y_test_data = zip(*score_MLP_test)
plt.figure(2)
plt.plot(x_train_data,y_train_data,"g")                     # Trainingsdaten
plt.plot(x_test_data,y_test_data,"r")                       # Testdaten
plt.plot(x_test_data,rate_diff,"b")
plt.axis( [10,100,0.1,1])                                   # Achsen Bereiche, x,y
plt.grid(True)                                              # Gitter einschalten
plt.legend(("Trainingsdaten","Testdaten","Differenz Score"))      # Legende
plt.xlabel("Nodes")
plt.ylabel("score [%]")
plt.title("Multi-layer Perceptron (MLP)")
plt.show()

# Ridge
x_Ridge_train,y_Ridge_train=zip(*Ridge_train)
x_Ridge_test,y_Ridge_test=zip(*Ridge_test)
plt.figure(3)
plt.plot(x_Ridge_train,y_Ridge_train,"g")                   # Trainingsdaten
plt.plot(x_Ridge_test,y_Ridge_test,"r")                     # Testdaten
plt.axis([max(x_Ridge_test),min(x_Ridge_test),0.5,1.0])     # Achsen Bereiche, x,y
plt.grid(True)                                              # Gitter einschalten
plt.legend(("Trainingsdaten","Testdaten"))                  # Legende
plt.xlabel("Alpha")
plt.ylabel("score [%]")
plt.xscale("log")
plt.title("Ridge")
plt.show()

# Lasso
x_Lasso_train,y_Lasso_train=zip(*Score_Lasso_train)
x_Lasso_test,y_Lasso_test=zip(*Score_Lasso_test)
plt.figure(4)
plt.plot(x_Lasso_train,y_Lasso_train,"g")                   # Trainingsdaten
plt.plot(x_Lasso_test,y_Lasso_test,"r")                     # Testdaten
plt.axis([max(x_Lasso_test),min(x_Lasso_test),0.5,1.0])     # Achsen Bereiche, x,y
plt.grid(True)                                              # Gitter einschalten
plt.legend(("Trainingsdaten","Testdaten"))                  # Legende
plt.xlabel("Alpha")
plt.ylabel("score [%]")
plt.xscale("log")
plt.title("Lasso")
plt.show()


# GBR

x_GBR_train,y_GBR_train=zip(*score_train_GBR)
x_GBR_test,y_GBR_test=zip(*score_test_GBR)
plt.figure(5)
plt.plot(x_GBR_train,y_GBR_train,"g")                          # Trainingsdaten
plt.plot(x_GBR_test,y_GBR_test,"r")                     # Testdaten
plt.axis([max(x_GBR_test),min(x_GBR_test),0.5,1.0])     # Achsen Bereiche, x,y
plt.grid(True)                                              # Gitter einschalten
plt.legend(("Trainingsdaten","Testdaten"))                  # Legende
plt.xlabel("Alpha")
plt.ylabel("score [%]")
plt.xscale("log")
plt.title("Gradient Boost regressor (GBR)")
plt.show()

plt.figure(6)
plt.grid(True)                              # Gitter einschalten
plt.bar(range(0,len(y_test)),y_delta,color="green")
plt.legend(["Abweichung Testdaten"])
plt.ylabel("delta km/h")
plt.xlabel("Anzahl Testdaten")
plt.show()


