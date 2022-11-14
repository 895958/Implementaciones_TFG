

import pandas as pd
import numpy as np

df = pd.read_csv('../RGB1_split_v1.csv',delimiter=';')
df = df.dropna()
df.head(5)

sujetos = pd.DataFrame(df, columns=['Sujeto'])
rutas = pd.DataFrame(df, columns=['Ruta'])
acciones = pd.DataFrame(df, columns=['Accion'])

sujetos_np = sujetos.to_numpy().squeeze()
rutas_np = rutas.to_numpy().squeeze()
acciones_np = acciones.to_numpy().squeeze()

X = np.vstack((sujetos_np,rutas_np)).T
y = acciones_np

from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()
logo.get_n_splits(X,y,sujetos_np)

for train_index, test_index in logo.split(X,y,sujetos_np):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
# from sklearn.model_selection import StratifiedShuffleSplit
# strat = StratifiedShuffleSplit(n_splits=1)

# for train_index, test_index in strat.split(X,y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

df_train = pd.DataFrame(np.vstack((X_train.T,y_train)).T)
df_train.columns = ['Sujeto', 'Ruta', 'Accion']
df_train.to_csv("train_jdelser.csv", index=False)

df_test = pd.DataFrame(np.vstack((X_test.T,y_test)).T)
df_test.columns = ['Sujeto', 'Ruta', 'Accion']
df_test.to_csv("test_jdelser.csv", index=False)