import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import aux_running
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import running_pd_data
import numpy as np

col={0:'b',1:'g',2:'r',3:'c',4:'m',5:'y'}

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Aktivitätstyp,Datum,Favorit,Titel,Distanz,Kalorien,Zeit,Ø HF,Max. HF,Aerober TE,
# Ø Schrittfrequenz (Laufen),Max. Schrittfrequenz (Laufen),Ø Pace,Beste Pace,Positiver Höhenunterschied,
# Negativer Höhenunterschied,Ø Schrittlänge,Durchschnittliches vertikales Verhältnis,Ø vertikale Bewegung,
# Training Stress Score®,Grit,Flow,Kletterzeit,Grundzeit,Min. Temp.,Oberflächenpause,Dekompression,
# Beste Rundenzeit,Anzahl der Läufe,Max. Temp.")
running = pd.read_csv('Activities (1).csv',sep=',',na_values='--')
running_red=running[['Distanz','Kalorien','Zeit','Ø Herzfrequenz','Ø Schrittfrequenz (Laufen)','Ø Pace',
                     'Positiver Höhenunterschied','Negativer Höhenunterschied','Ø Schrittlänge']]

running_red=running_red.dropna() # drop rows with NaN

# convert strings (e.g from time, pace and calories to numbers
for idx in list(running_red.index):
    temp=running_red.loc[idx,'Zeit']
    running_red.at[idx,'Zeit']=aux_running.TimeToSec(temp)
    temp = running_red.loc[idx, 'Ø Pace']
    running_red.at[idx, 'Ø Pace'] = aux_running.TimeToSec(temp)
    temp = running_red.loc[idx, 'Kalorien']
    running_red.at[idx, 'Kalorien'] = aux_running.Calories(temp)

#running_red=running_pd_data.GenData() # pseudo data

running_red_cluster=running_red.copy()

running_np=running_red.values                   # move pandas data frame to numpy array
scaler=MinMaxScaler()                           # create MinMax scaler object
scaler.fit(running_np)                          # and fit the model to the data
running_np_scaled=scaler.transform(running_np)  # transform the data, values (0...1)

# Hauptkomponenten-Zerlegung
pca=PCA(n_components=2)                 # create object and define two maincomponents to reflect the data
pca.fit(running_np_scaled)              # train model with scaled input
X_pca=pca.transform(running_np_scaled)  # do the transformation

# print data as scatter plot
plt.figure(1)
plt.scatter(X_pca[:,0],X_pca[:,1],c='b')
plt.title("pca - Klassifikation")
plt.show()

# clustering via k-means
kmeans = KMeans(n_clusters=6)       # define kmeans object with 6 cluster
kmeans.fit(X_pca)                   # and fit the opbject to the data

#fig, ax = plt.subplots()
#ax.scatter(X_pca[:,0], X_pca[:,1],c=list(kmeans.labels_))
#ax.legend(labels=(0,1,2,3))
#for i, txt in enumerate(list(running_red.index)):
#    ax.annotate(txt, (X_pca[:,0][i], X_pca[:,1][i]))

# print data with cluster information as well as index of run
fig, ax = plt.subplots()
for i, txt in enumerate(list(running_red.index)):
    ax.scatter(X_pca[:, 0][i], X_pca[:, 1][i],c=col[kmeans.labels_[i]])
    ax.annotate(txt, (X_pca[:,0][i], X_pca[:,1][i]))

# add cluster to pandas data frame
running_red_cluster['Cluster']=list(kmeans.labels_)



# write pandas data frame to excel file
#running_red_cluster.to_excel(r'C:\\Users\\Arno\\PycharmProjects\\KI\\export_red_Cluster.xlsx', index = True)

