import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import aux_running
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import running_pd_data
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

mapping = {'CONFIRMED': 0, 'FALSE POSITIVE': 1, 'CANDIDATE':2}

exo_planet_raw = pd.read_csv('exo_planets.csv',sep=',')  # read csc file

#delete columns
del exo_planet_raw['kepler_name']
del exo_planet_raw['rowid']
del exo_planet_raw['koi_teq_err1']
del exo_planet_raw['koi_teq_err2']
del exo_planet_raw['koi_tce_delivname']

# generate input (X) and output (y) data
out=(exo_planet_raw.loc[:,'koi_disposition']).replace(to_replace=mapping)
features=exo_planet_raw.loc[:,'koi_fpflag_nt':]

# move pandas data frame to numpay array
X=features.values()
y=out.values()

# generate training and test data
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
