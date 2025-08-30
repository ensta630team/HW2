import numpy as np

def calculate_autocovariance(serie: np.ndarray, lag: int) -> np.ndarray:
    """
    Calcula las autocovarianzas de una serie de tiempo de forma eficiente y vectorizada.

    Args:
        serie (np.ndarray): Un array de NumPy 1D que representa la serie de tiempo.
        lag (int): El número de rezagos (lags) para los cuales calcular la autocovarianza.
                     El resultado incluirá el rezago 0 (la varianza).

    Returns:
        np.ndarray: Un array 1D con las autocovarianzas desde el rezago 0 hasta `lag - 1`.
    """
    if lag > len(serie):
        raise ValueError("El orden no puede ser mayor que la longitud de la serie.")

    n = len(serie)
    
    # 1. Centrar la serie (restar la media). Esto se hace una sola vez.
    serie_centrada = serie - np.mean(serie)
    
    # 2. Usar np.correlate para calcular la suma de productos cruzados para todos los rezagos.
    #    'full' calcula la correlación completa.
    autocorr_completa = np.correlate(serie_centrada, serie_centrada, mode='full')
    
    # 3. La autocovarianza es el resultado de la correlación dividido por n.
    #    Seleccionamos solo los rezagos no negativos (la segunda mitad del resultado).
    autocovarianzas = autocorr_completa[n - 1 : n + lag -1] / n
    
    return autocovarianzas