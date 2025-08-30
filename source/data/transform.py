import numpy as np

def create_lagged_dataset(serie: np.ndarray, lag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforma una serie de tiempo en un dataset supervisado con rezagos.

    Args:
        serie (np.ndarray): Un array de NumPy 1D que representa la serie de tiempo.
        n_rezagos (int): El número de pasos de tiempo anteriores (rezagos) que se usarán como
                         características de entrada (X).

    Returns:
        tuple[np.ndarray, np.ndarray]: Una tupla que contiene:
            - X (np.ndarray): Matriz 2D donde cada fila contiene los 'n_rezagos' valores
                              anteriores.
            - y (np.ndarray): Array 1D con el valor objetivo correspondiente a cada fila de X.
    """
    # Se inicializan listas vacías para almacenar los datos
    X, y = [], []

    # Se asegura de que la serie sea un array de NumPy
    if isinstance(serie, list):
        serie = np.array(serie)

    # El bucle comienza en el primer punto donde hay suficientes datos pasados (n_rezagos)
    # y termina al final de la serie.
    for i in range(lag, len(serie)):
        # La ventana de características (X) es la subsecuencia desde [t - n_rezagos] hasta [t-1]
        rezagos = serie[i - lag:i]
        X.append(rezagos)
        
        # El objetivo (y) es el valor en el tiempo actual [t]
        y.append(serie[i])

    # Se convierten las listas a arrays de NumPy para operaciones eficientes
    return np.array(X), np.array(y)


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