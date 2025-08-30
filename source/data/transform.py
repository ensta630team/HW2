import numpy as np

def create_lagged_dataset(serie: np.ndarray, lag: int = 1, add_intercept=True) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforma una serie de tiempo en un dataset supervisado con rezagos.
    """
    X, y = [], []

    if isinstance(serie, list):
        serie = np.array(serie)

    for i in range(lag, len(serie)):
        # [y_{t-1}, y_{t-2}, ...]
        rezagos = serie[i - lag:i][::-1]
        X.append(rezagos)
        
        y.append(serie[i])

    X = np.array(X) 
    y = np.array(y)
    
    if add_intercept:
        X = np.c_[np.ones(X.shape[0]), X]

    return X, y

def get_windows(serie: np.ndarray, n_ventanas: int, tamano_ventana: int) -> np.ndarray:
    """
    Extrae sub-series (ventanas) aleatorias de un tamaño fijo de una serie de tiempo.
    
    Argumentos:
        serie (np.ndarray): La serie de tiempo original.
        n_ventanas (int): El número de ventanas a extraer.
        tamano_ventana (int): El largo de cada ventana.

    Retorna:
        np.ndarray: Un array 2D donde cada fila es una ventana extraída.
    """
    largo_serie = len(serie)
    if tamano_ventana > largo_serie:
        raise ValueError("El tamaño de la ventana no puede ser mayor que el largo de la serie.")
    
    # Determina todos los posibles puntos de inicio para una ventana.
    max_ventanas_posibles = largo_serie - tamano_ventana + 1
    posibles_inicios = np.arange(max_ventanas_posibles)
    
    # Elige aleatoriamente los puntos de inicio para las ventanas deseadas (con reemplazo).
    inicios_elegidos = np.random.choice(a=posibles_inicios, size=n_ventanas, replace=True)

    # De forma vectorizada, crea los índices para todas las ventanas y las extrae de la serie.
    # Esto es mucho más eficiente que un bucle for.
    indices_ventanas = inicios_elegidos[:, np.newaxis] + np.arange(tamano_ventana)
    ventanas = serie[indices_ventanas]
    
    return ventanas