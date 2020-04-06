from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn

cancer=load_breast_cancer()

X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=0)

mlp=MLPClassifier(random_state=42).fit(X_train,y_train)
print("ohne Skalierung der Daten ergebn sich folgende scores")
print("Genauigketi Trainingsdaten",mlp.score(X_train,y_train))
print("Genauigkeit der Tesdaten",mlp.score(X_test,y_test))

#umskalieren der Daten
mean_on_train=X_train.mean(axis=0)
std_on_train=X_train.std(axis=0)
X_train_scaled=(X_train-mean_on_train)/std_on_train
X_test_scaled=(X_test-mean_on_train)/std_on_train
mlp_scaled=MLPClassifier(random_state=42,max_iter=1000,alpha=1).fit(X_train_scaled,y_train)
print("\nmit Skalierung der Daten ergebn sich folgende scores")
print("Genauigketi Trainingsdaten",mlp_scaled.score(X_train_scaled,y_train))
print("Genauigkeit der Tesdaten",mlp_scaled.score(X_test_scaled,y_test))