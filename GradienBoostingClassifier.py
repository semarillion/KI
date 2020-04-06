from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn

cancer=load_breast_cancer()

X_train,X_test,y_train,y_test=train_test_split(cancer.data, cancer.target, random_state=42)

rfcl=RandomForestClassifier(random_state=0,max_depth=5,n_estimators=4).fit(X_train,y_train)
gbrt=GradientBoostingClassifier(random_state=0,max_depth=1,learning_rate=0.1,n_estimators=4).fit(X_train,y_train)


print("RandomForestClassifier:")
print("Genauigkeit der Trainingsdaten %1.3f"%rfcl.score(X_train,y_train))
print("Genauigkeit der Testdaten %1.3f"%rfcl.score(X_test,y_test))
print("GradientBoostingClassifier:")
print("Genauigkeit der Trainingsdaten %1.3f"%gbrt.score(X_train,y_train))
print("Genauigkeit der Testdaten %1.3f"%gbrt.score(X_test,y_test))


