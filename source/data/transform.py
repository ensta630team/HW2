import numpy as np

def create_var_dataset(data: np.ndarray, lag: int, add_intercept: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforma un conjunto de series de tiempo en un dataset supervisado
    para un modelo VAR(p=lag).

    Argumentos:
    data (np.ndarray): Array de forma (T, n) con T observaciones y n variables.
    lag (int): Número de rezagos (p) a incluir.
    add_intercept (bool): Si es True, añade una columna de unos para el intercepto.

    Retorna:
    tuple: (X, Y)
        X (np.ndarray): Matriz de regresores con forma (T-lag, np+1).
        Y (np.ndarray): Matriz de variables objetivo con forma (T-lag, n).
    """
    X, Y = [], []
    
    # El bucle comienza en 'p' porque necesitamos 'lag' observaciones pasadas.
    for i in range(lag, len(data)):
        # La variable objetivo Y_t es el vector de todas las variables en el tiempo i.
        target_vector = data[i]
        Y.append(target_vector)
        
        # El vector de regresores se construye con los 'lag' rezagos de TODAS las variables.
        # Tomamos las 'lag' filas anteriores: de data[i-lag] a data[i-1]
        # Las aplanamos para crear un único vector de regresores.
        # [y1_{t-1}, y2_{t-1}, ..., yn_{t-1}, y1_{t-2}, ..., yn_{t-lag}]
        lags_matrix = data[i-lag:i]
        
        # Invertimos el orden de las filas para que el primer rezago (t-1) venga primero.
        regressor_vector = np.flip(lags_matrix, axis=0).flatten()
        X.append(regressor_vector)

    X = np.array(X)
    Y = np.array(Y)
    
    if add_intercept:
        # Añadimos la columna de unos al principio de la matriz X.
        X = np.c_[np.ones(X.shape[0]), X]
        
    return X, Y

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