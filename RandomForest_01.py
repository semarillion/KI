from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn


X,y=make_moons(n_samples=100,noise=0.25,random_state=3)

X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42)

# n_estimators = Anzahl der maximal zu estellenden Baäume
# random_state = wenn reproduzierbare Ergebnisse gewünscht sollte der Paramter gleich sein
score_train=[]
score_test=[]

Range=[1,2,3,4,5,6,7,8,9,20,50,100] # Liste mit Anzahl der Bäume
for estimator in Range:
    forest=RandomForestClassifier(n_estimators=estimator,random_state=2).fit(X_train,y_train)
    score_train.append(forest.score(X_train,y_train))
    score_test.append(forest.score(X_test,y_test))

# Daten ausgeben, Score der Trainings- und Testdaten
plt.plot(Range,score_train,label="training")
plt.plot(Range,score_test,label="test")
plt.xlabel("Anzahl der Bäume")
plt.ylabel("score = f(Anzahl der Bäume")
plt.legend()









