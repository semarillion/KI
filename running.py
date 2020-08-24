import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import aux_running
import numpy as np


#pd.set_option('display.max_rows', 200)
#pd.options.display.max_columns = 20
pd.options.display.max_rows = None

# Aktivitätstyp,Datum,Favorit,Titel,Distanz,Kalorien,Zeit,Ø HF,Max. HF,Aerober TE,
# Ø Schrittfrequenz (Laufen),Max. Schrittfrequenz (Laufen),Ø Pace,Beste Pace,Positiver Höhenunterschied,
# Negativer Höhenunterschied,Ø Schrittlänge,Durchschnittliches vertikales Verhältnis,Ø vertikale Bewegung,
# Training Stress Score®,Grit,Flow,Kletterzeit,Grundzeit,Min. Temp.,Oberflächenpause,Dekompression,
# Beste Rundenzeit,Anzahl der Läufe,Max. Temp.")

running = pd.read_csv('activities.csv',sep=',',na_values='--')
running_red=running[['Distanz','Kalorien','Zeit','Ø HF','Ø Schrittfrequenz (Laufen)','Ø Pace',
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

running_np=running_red.values # move pandas data frame to numpy array
scaler=MinMaxScaler()
scaler.fit(running_np)
running_np_scaled=scaler.transform(running_np)






#running_red.to_excel(r'C:\\Users\\Arno\\PycharmProjects\\KI\\export_red.xlsx', index = False)
#running.to_excel(r'C:\\Users\\Arno\\PycharmProjects\\KI\\export_full.xlsx', index = False)
