import pandas as pd
import matplotlib.pyplot as plt



# Aktivitätstyp,Datum,Favorit,Titel,Distanz,Kalorien,Zeit,Ø HF,Max. HF,Aerober TE,
# Ø Schrittfrequenz (Laufen),Max. Schrittfrequenz (Laufen),Ø Pace,Beste Pace,Positiver Höhenunterschied,
# Negativer Höhenunterschied,Ø Schrittlänge,Durchschnittliches vertikales Verhältnis,Ø vertikale Bewegung,
# Training Stress Score®,Grit,Flow,Kletterzeit,Grundzeit,Min. Temp.,Oberflächenpause,Dekompression,
# Beste Rundenzeit,Anzahl der Läufe,Max. Temp.")

running = pd.read_csv('activities.csv',sep=',')
running_red=running[['Datum','Distanz','Kalorien','Zeit','Ø HF','Ø Schrittfrequenz (Laufen)','Ø Pace',]]


a=running_red['Ø HF'].value_counts()
counts=a.values.tolist()
puls=a.keys().tolist()
plt.plot(counts,puls,'or')