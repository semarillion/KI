import pandas as pd

def GenData():
     data={'Distanz': [12,14,15,20],\
                   'Kalorien': [500,500,500,500],\
                   'Zeit': [3000,3000,3000,3000],\
                   'Ø HF': [168,168,168,168],\
                   'Ø Schrittfrequenz (Laufen)': [168,168,168,168,],\
                   'Ø Pace': [290,300,310,320],\
                   'Positiver Höhenunterschied':[0,10,20,30],\
                   'Negativer Höhenunterschied':[0,10,20,30],\
                   'Ø Schrittlänge':[1.01,1.01,1.01,1.01]}

     data = pd.DataFrame(data, columns = ['Distanz','Kalorien','Zeit','Ø HF','Ø Schrittfrequenz (Laufen)','Ø Pace',
                          'Positiver Höhenunterschied','Negativer Höhenunterschied','Ø Schrittlänge'], index=[0,1,2,3])
     return data