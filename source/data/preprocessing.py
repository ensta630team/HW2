import numpy as np 
import pandas as pd
from datetime import datetime

def to_X_y(y, p):
    """
    Transforma una serie de tiempo en una matriz de características (X) y un vector objetivo (Y).
    Es útil para preparar datos para modelos autorregresivos (AR).

    Argumentos:
        y (np.ndarray): La serie de tiempo original.
        p (int): El número de rezagos (lags) a utilizar como características.

    Retorna:
        tuple: Una tupla con la matriz X (regresores) y el vector Y (objetivo).
    """
    nobs = len(y)
    # Inicializa X con una columna de unos para el término de intercepto.
    X = np.ones(nobs - p)
    
    # Itera para crear y añadir cada rezago como una nueva columna en X.
    for i in range(p):
        # Crea el rezago (ej. y_{t-1}, y_{t-2}, ..., y_{t-p}).
        nuevorezago = y[p-1-i : nobs-1-i]
        X = np.column_stack([X, nuevorezago])
    
    # El vector Y contiene los valores actuales de la serie (y_t), a partir del período 'p'.
    Y = y[p:]
    
    return X, Y

def split_dataset(dataset: dict, cutoff_date_str: str) -> dict:
    """
    Divide un diccionario de series de tiempo en dos subconjuntos (ej. entrenamiento y prueba)
    basándose en una fecha de corte.

    Argumentos:
        dataset (dict): Diccionario que contiene las series de tiempo.
        cutoff_date_str (str): La fecha de corte en formato 'YYYY-MM-DD'.

    Retorna:
        tuple: Una tupla con dos diccionarios, el primero con datos hasta la fecha de corte
               y el segundo con datos posteriores.
    """
    # Copia el diccionario para no modificar el original.
    dataset_cp = dataset.copy()
    
    # Ajusta las series para alinear sus longitudes, eliminando el primer dato
    # que se perdió al calcular diferencias logarítmicas.
    dataset_cp['t'] = dataset_cp['t'][1:]
    dataset_cp['i_t'] = dataset_cp['i_t'][1:]
    
    # Convierte el diccionario a un DataFrame de pandas para facilitar la manipulación.
    df = pd.DataFrame(dataset_cp)
    df['t'] = pd.to_datetime(df['t'])

    # Filtra el DataFrame para crear los dos conjuntos de datos.
    df_0 = df[df['t'] <= cutoff_date_str] # Datos de entrenamiento/históricos.
    df_1 = df[df['t'] > cutoff_date_str]  # Datos de prueba/nuevos.

    # Convierte los DataFrames de vuelta a diccionarios y los retorna.
    return df_0.to_dict(orient='list'), df_1.to_dict(orient='list')