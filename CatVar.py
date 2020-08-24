import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data=pd.read_csv("adult.data",header=None,index_col=False,names=['age','workclass','fnlwgt','education','education-num','marital-status',\
                                                                      'occupation','relationship','race','gender','capital-gain','capital-loss',\
                                                                      'hours-per-week','native-country','income'])

data=data[['age','workclass','education','gender','hours-per-week','occupation','income']]

data_dummies=pd.get_dummies(data) #bearbeitung kategorischer Merkmale

features=data_dummies.loc[:, 'age':'occupation_ Transport-moving'] #nimm nur die Eingangsvariablen
X=features.values                                                   # und wandle in NumpyArray um

y=data_dummies['income_ >50K'].values # Ergebnis Spalte, nur eine Spalte wird benötigt, da Inhalt indirekt schon drin steht

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
logreg=LogisticRegression()
logreg.fit(X_train,y_train)


