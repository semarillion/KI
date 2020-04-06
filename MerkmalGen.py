from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
data=pd.read_csv("adult.data",header=None,index_col=False,names=['age','workclass','fnlwgt','education',
                            'education-num','marital-status','occupation','relationship','race','gender','capital-gain',
                            'capital-loss','hours-per-week','native country','income'])

# reduziere aus der original Daten nur die folgenden Spalten
data=data[['age','workclass','education','gender','hours-per-week','occupation','income']]

# wandle kategorische Merkmal in kontinuierliche Merkmale um -> Spaltenanzahl wird größer
data_dummies=pd.get_dummies(data)

# extrahiere aus gesamger Tabelle nur die Eingangsdaten - X-Vektor
features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']

#und überführe in numpy array
X=features.values

# extrahiere aus gesamter Tabelle nur die Ausgangdaten und überführe in numpy array
y=data_dummies['income_ >50K'].values

# teile die Daten in trainings- und Testdaten auf
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

LogReg=LogisticRegression(solver='lbfgs',max_iter=1000)
LogReg.fit(X_train,y_train)

print(LogReg.score(X_test,y_test))
