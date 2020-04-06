import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from collections import Counter

cnr=Counter()
txt_dict= {0:"noise"}

pd_rezept = pd.read_csv("CupcakeVsMuffin.csv",sep=",",names=["Flour","Milk","Sugar",\
                                                                    "Butter","Egg","Baking powder","Vanilla",\
                                                                    "salt","Type"])
np_rezept=np.array(pd_rezept)

Zutaten=list(pd_rezept.keys())
Zutaten.remove("Type")


X=np_rezept[:,0:7] # Eingangsvektor "Zutaten"
y=np_rezept[:,8:]  # Ausgangsvektor "cupcake oder Muffin"

# Liste für Farben erzeugen: Muffin = blue, Cupcake = green
y_name=[]
for i in y:
    if i=="Muffin":
        y_name.append("b")
    else:
        y_name.append("g")

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42) # Daten in Trainings- und Testdaten aufteilen

# Trainings und Testdaten skalieren
scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

# Alle Daten skalieren
scaler_all = StandardScaler()
scaler_all.fit(X)
X_scaled = scaler_all.transform(X)

pca=PCA(n_components=2)
pca.fit(X)
X_pca=pca.transform(X)

pca_scaled=PCA(n_components=2)
pca_scaled.fit(X_scaled)
X_pca_scaled=pca_scaled.transform(X_scaled)

LogReg=LogisticRegression(C=1000,solver="newton-cg")
LogReg.fit(X_pca_scaled,y.flatten())

line=np.linspace(-1.5,1.5,50) # X-Achse generieren

plt.figure(1)
plt.scatter(X_pca_scaled[:,0],X_pca_scaled[:,1],c=y_name)
plt.plot(line, -(line*LogReg.coef_.flatten()[0]+LogReg.intercept_[0])/LogReg.coef_.flatten()[1],c="red")
plt.title("pca - Klassifikation")
plt.show()

plt.figure(2)
#xTicks=["Flour","Milk","Sugar","Butter","Egg","Baking powder","Vanilla"]
xTicks=Zutaten
plt.matshow(pca_scaled.components_,cmap="plasma")
plt.yticks([0,1],["Erste Hautpkomponente","Zweite Hauptkomponente"])
plt.xticks(range(len(xTicks)),xTicks,rotation=60)
plt.xlabel("Merkmal")
plt.ylabel("Hautpkomponente")
plt.title("Heat map - pca")
plt.colorbar()
plt.show()


plt.figure(4)
plt.title("k-means clustering")
kmeans=KMeans(n_clusters=2).fit(X_scaled)
plt.scatter(X_pca_scaled[:,0],X_pca_scaled[:,1],c=list(kmeans.labels_))
plt.show()

plt.figure(5)
plt.title("Agglomeerative clustering")
agg=AgglomerativeClustering(n_clusters=3)
assignment=agg.fit(X_scaled)
plt.scatter(X_pca_scaled[:,0],X_pca_scaled[:,1],c=list(assignment.labels_))
plt.show()

plt.figure(6)
plt.title("DBSCAN")
dbscan=DBSCAN(eps=2.5,min_samples=5)
clusters=dbscan.fit(X_scaled)

cnt = Counter()                     # will feststellen, wieviele Cluster ermittelt wurden
for cl in list(clusters.labels_):
    cnt[cl]+=1
cl=list(cnt.keys())

z=0
X_pca_sep=[]
for i in range(0,len(cl)):                          # Listen-array mit Anzahl der Cluster anlegen
    X_pca_sep.append([])
    txt_dict[i+1]="cluster "+str(i)                 # und Label-Text für späteres Diagramm anlegen

for i in (clusters.labels_):                        # in Abhängigkeit, zu welchem cluster ein Datensatz gehört
    X_pca_sep[i+1].append(list(X_pca_scaled[z]))    # Datensätze an Liste anhängen, um später die Listen separat
    z+=1                                            # für jeden Cluster einzeln auszugeben

for i in range(0,len(cl)):                          # gehe jetzt über jede Cluster gruppe
    x, y = zip(*X_pca_sep[i])                       # entpacke X-Daten und y-Daten
    plt.scatter(x,y,label=txt_dict[i])              # ausgabe und Legende mit jeweiligen Text
    plt.legend()
plt.show()

