import matplotlib.pyplot as mp
import numpy as np
import FirstNeuralNetwork
from random import randint

d2=0.99
d1=0.01
max_y=[]
y_norm=[]
x_norm=[]
x_norm_in=[]
x_test_dia=[]
y_test_dia=[]
y_test_dia_2=[]
y_test_dia_3=[]
y_test_dia_4=[]

x_min=-15#int(input("Minimaler x-Wert der Funktion"))
x_max=7#int(input("Minimaler x-Wert der Funktion"))
x_delta=0.1
#float(input("Schrittweite"))
x_test=30
x_test_data=[]

epoche=5000

# reale Funktion als Graph darstellen
x1=np.arange(x_min,x_max,x_delta)
y1=[(i**4+12*i**3-3*i**2+16*i+30) for i in x1]


# Normieren auf 0.1..0.9 mit Schrittweite x_delta
for i in np.arange(x_min,x_max,x_delta):
    x_norm.append( (i-x_min)*(d2-d1)/(x_max-x_min)+d1)

# ermittle die größten Y-Werte (als absolut Werte) und normiere die Y-Werte zwischen 0.1..0.9
max_y_dia=max(y1)           # max und
min_y_dia=min(y1)           # min werte für Diagramm ermitteln
y_norm=[(i-min_y_dia)*(d2-d1)/(max_y_dia-min_y_dia)+d1 for i in y1]

# eine Figur, zwei Plots in einer Spalter untereinander
fig, (ax1, ax2) = mp.subplots(1,2, figsize=(10,4))
ax1.plot(x1,y1, 'go')           # greendots
ax2.plot(x_norm,y_norm, 'b*')  #

# Titel und Achsenbeschriftung und -skalierung
ax1.set_title('Eingangsfunktion'); ax2.set_title('Normierte Funktion')
ax1.set_xlabel('X');  ax2.set_xlabel('X')  # x label
ax1.set_ylabel('Y');  ax2.set_ylabel('Y')  # y label
ax1.set_xlim(x_min, x_max) ;  ax2.set_xlim(0, 1)          # x axis limits
ax1.set_ylim(min_y_dia,max_y_dia);  ax2.set_ylim(0, 1)   # y axis limits
mp.show()

# normierte X-Daten aufbereiten - ein Konstante einfügen und neue Liste generieren
for i in x_norm:
    x_norm_in.append([0.99,i])

#neuronales Netz konfigurieren
input_nodes=2
output_nodes=1
hidden_nodes=50
learning_rate=0.3

## und dann erstellen
n=FirstNeuralNetwork.neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)


# test daten für neuronales Netz generieren
for i in range(0,x_test):
    random_value=randint(0, 9999) / 10000
    x_test_data.append([0.99,random_value])
    x_test_dia.append(random_value)


# Neuronales Netz trainieren
for i in range(0,epoche+1):                   # durchlauf der Epoche
    for k in range(0,len(x_norm_in)):       # durchlaufen der Trainingsdaten
        n.train(x_norm_in[k],y_norm[k])

    if i==0:
        for i in x_test_data:                       # Netz mit testdaten
            y_test_out=n.query(x_test_data)         # abfragen
        y_test_out=y_test_out.tolist()              # und numpy array in Liste umwandeln
        for i in y_test_out[0]:                     # verschachtelte Liste
            y_test_dia.append(i)                    # dann nochmal in ordendliche Liste umwandeln

    if i==(epoche//100):
        for i in x_test_data:                       # Netz mit testdaten
            y_test_out=n.query(x_test_data)         # abfragen
        y_test_out=y_test_out.tolist()              # und numpy array in Liste umwandeln
        for i in y_test_out[0]:                     # verschachtelte Liste
            y_test_dia_2.append(i)                   # dann nochmal in ordendliche Liste umwandeln

    if i==(epoche//10):
        for i in x_test_data:                       # Netz mit testdaten
            y_test_out=n.query(x_test_data)         # abfragen
        y_test_out=y_test_out.tolist()              # und numpy array in Liste umwandeln
        for i in y_test_out[0]:                     # verschachtelte Liste
            y_test_dia_3.append(i)                  # dann nochmal in ordendliche Liste umwandeln

    if i==(epoche//1):
        for i in x_test_data:                       # Netz mit testdaten
            y_test_out=n.query(x_test_data)         # abfragen
        y_test_out=y_test_out.tolist()              # und numpy array in Liste umwandeln
        for i in y_test_out[0]:                     # verschachtelte Liste
            y_test_dia_4.append(i)                  # dann nochmal in ordendliche Liste umwandeln

# test daten darstellen
# eine Figur, zwei Plots in einer Spalter in einem diagram
mp.figure(figsize=(10,4), dpi=100)
mp.plot(x_norm,y_norm,"go")
mp.plot(x_test_dia,y_test_dia,"b*")
mp.plot(x_test_dia,y_test_dia_2,"r*")
mp.plot(x_test_dia,y_test_dia_3,"y*")
mp.plot(x_test_dia,y_test_dia_4,"k*")