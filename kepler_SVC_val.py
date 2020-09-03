# Prognose ob Exoplant anhand der cumulative data base der NASA

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import kepler_SVC_val_aux as kepler_aux

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

mapping = {'CONFIRMED' : 0, 'FALSE POSITIVE' : 1, 'CANDIDATE' : 2}
col={0:'g',1:'r',2:'y'}

exo_planet_raw = pd.read_csv('exo_planets.csv', sep=',', na_values=('nan'))  # read csv file

# delete columns
del exo_planet_raw['kepler_name']
del exo_planet_raw['rowid']
del exo_planet_raw['koi_teq_err1']
del exo_planet_raw['koi_teq_err2']
del exo_planet_raw['koi_tce_delivname']

# fill missing values with previous content of cell
exo_planet_raw = exo_planet_raw.fillna(method='pad')

# drop lines with NaN
exo_planet_raw = exo_planet_raw.dropna(axis=0)

# generate input (X) and output (y) data
out = (exo_planet_raw.loc[:, 'koi_disposition']).replace(to_replace=mapping)
features = exo_planet_raw.loc[:, 'koi_fpflag_nt' :]

# move pandas data frame to numpay array
X = features.values
y = out.values


# generate training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

# select scaler
#scaler = MinMaxScaler(feature_range=(0.01, 0.99))
scaler=StandardScaler()

# fit training data to the model
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled= scaler.transform(X)

# simple grid test via for loop over C and gamma values
best_score,best_paramters=kepler_aux.SimpleGridSearch(X_train_scaled,y_train,X_test_scaled,y_test)
print('bester score',best_score)
print('best parameters',best_paramters)


X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=42)


# fit training data to the model
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_trainval_scaled = scaler.transform(X_trainval)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

print('get best paramters on X_train_scaled, y_train')
best_score,best_paramters=kepler_aux.SimpleGridSearch(X_train_scaled,y_train,X_val_scaled,y_val)
print('bester score validation data:',best_score)
print('best parameters',best_paramters)

print('apply best parameteres on X_test_scaled, y_test')
svc=SVC(**best_paramters)
svc.fit(X_trainval_scaled,y_trainval)
svc_score=svc.score(X_test_scaled,y_test)
print('score test data:',svc_score)






# principal component analysis - 2 componentes
#pca=PCA(n_components=2,tol=0.001,random_state=0)    # create object and define two maincomponents to reflect the data
#pca.fit(X_scaled)                                   # train model with scaled input
#X_pca=pca.transform(X_scaled)                       # do the transformation
#
## principal component analysis - 3 componentes
#pca_3D=PCA(n_components=3,tol=0.001,random_state=0)    # create object and define three maincomponents to reflect the data
#pca_3D.fit(X_scaled)                                   # train model with scaled input
#X_pca_3D=pca_3D.transform(X_scaled)                    # do the transformation
#
## plot 2d picture
#plt.figure(1)
#plt.scatter(X_pca[:,0],X_pca[:,1],c=[col[x] for x in y])
#plt.xlabel('1st component')
#plt.ylabel('1st component')
#plt.title("pca")
#plt.show()
#
## plot 3d picture
#fig = plt.figure(2)
#ax = plt.axes(projection='3d')
#ax.set_xlabel('1st component')
#ax.set_ylabel('2nd component')
#ax.set_zlabel('3rd component')
#ax.scatter3D(X_pca_3D[:,0],X_pca_3D[:,1],X_pca_3D[:,2],c=[col[x] for x in y])
#
#fig = plt.figure(3)
#plt.subplot(121)
#plt.scatter(features.columns,pca.components_[0],c='b')
#plt.xlabel('feature')
#plt.ylabel('variant')
#plt.title("1. Hauptkomponente:\n"\
#          +str(features.columns[pca.components_[0].argmax()])+'\n'\
#          +str(features.columns[pca.components_[0].argmin()]))
#plt.subplot(122)
#plt.scatter(features.columns,pca.components_[1],c='g')
#plt.xlabel('feature')
#plt.ylabel('variant')
#plt.title("2. Hauptkomponente:\n"\
#          +str(features.columns[pca.components_[1].argmax()])+'\n'\
#          +str(features.columns[pca.components_[1].argmin()]))
#plt.show()