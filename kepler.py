import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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

#scaler = MinMaxScaler(feature_range=(0.01, 0.99))
                                                    #LinearClassifier:
                                                    #Prognose Trainingsdaten: 86.7698 %
                                                    #Prognose Testdaten: 86.7838 %
                                                    #MLP
                                                    #Prognose Trainingsdaten: 87.7178 %
                                                    #Prognose Testdaten: 87.3693 %


scaler=StandardScaler()
                                                    #LinearClassifier:
                                                    #Prognose Trainingsdaten: 86.9232 %
                                                    #Prognose Testdaten: 87.0765 %

                                                    #MLP
                                                    #Prognose Trainingsdaten: 97.7834 %
                                                    #Prognose Testdaten: 86.1564 %

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled= scaler.transform(X)

linear_svm = LinearSVC(C=100000,\
                       max_iter=10000,\
                       dual=False).fit(X_train_scaled, y_train)

mlp = MLPClassifier(solver='lbfgs',\
                    random_state=0,\
                    activation='tanh',\
                    alpha=0.01,\
                    learning_rate_init=0.1).fit(X_train_scaled, y_train)

linear_svm_cross = cross_val_score(linear_svm,X_scaled,y,cv=3)
mlp_svm_cross = cross_val_score(mlp,X_scaled,y,cv=3)

print("LinearClassifier:")
print('Prognose Trainingsdaten:', round(linear_svm.score(X_train_scaled, y_train) * 100, 4), '%')
print('Prognose Testdaten:', round(linear_svm.score(X_test_scaled, y_test) * 100, 4), '%\n')
print('Cross Validierung:',linear_svm_cross)

print("MLP")
print('Prognose Trainingsdaten:', round(mlp.score(X_train_scaled, y_train) * 100, 4), '%')
print('Prognose Testdaten:', round(mlp.score(X_test_scaled, y_test) * 100, 4), '%\n')
pront('Cross Validierung:',mlp_svm_cross)


# principal component analysis - 2 componentes
pca=PCA(n_components=2,tol=0.001,random_state=0)    # create object and define two maincomponents to reflect the data
pca.fit(X_scaled)                                   # train model with scaled input
X_pca=pca.transform(X_scaled)                       # do the transformation

# principal component analysis - 3 componentes
pca_3D=PCA(n_components=3,tol=0.001,random_state=0)    # create object and define three maincomponents to reflect the data
pca_3D.fit(X_scaled)                                   # train model with scaled input
X_pca_3D=pca_3D.transform(X_scaled)                    # do the transformation

# plot 2d picture
plt.figure(1)
plt.scatter(X_pca[:,0],X_pca[:,1],c=[col[x] for x in y])
plt.xlabel('1st component')
plt.ylabel('1st component')
plt.title("pca")
plt.show()

# plot 3d picture
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('1st component')
ax.set_ylabel('2nd component')
ax.set_zlabel('3rd component')
ax.scatter3D(X_pca_3D[:,0],X_pca_3D[:,1],X_pca_3D[:,2],c=[col[x] for x in y])