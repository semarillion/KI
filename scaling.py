import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
cancer = load_breast_cancer()

svm=SVC(C=100)

X_train,X_test,y_train,y_test=train_test_split(cancer.data, cancer.target,random_state=1)

scaler=MinMaxScaler()

scaler.fit(X_train)

X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

svm.fit(X_train_scaled,y_train)

