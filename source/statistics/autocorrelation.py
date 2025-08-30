import numpy as np

def calculate_acf(data: np.ndarray, lags: int) -> np.ndarray:
    """
    Calcula la Función de Autocorrelación (ACF) para una serie de tiempo.

    Args:
        data (np.ndarray): La serie de tiempo (1D array).
        lags (int): El número máximo de rezagos para calcular.

    Returns:
        np.ndarray: Un array con los valores de la ACF para los rezagos 0 a `lags`.
    """
    n = len(data)
    if n <= lags:
        raise ValueError("El número de lags debe ser menor que la longitud de los datos.")

    acf_values = np.zeros(lags + 1)
    mean = np.mean(data)
    # El denominador de la ACF es la varianza de la serie.
    variance = np.sum((data - mean)**2)

    # Evita la división por cero si la serie es constante.
    if variance == 0: 
        return np.zeros(lags + 1)

    # Por definición, la autocorrelación en el rezago 0 es siempre 1.
    acf_values[0] = 1.0

    # Itera para cada rezago desde 1 hasta 'lags'.
    for k in range(1, lags + 1):
        # El numerador es la autocovarianza para el rezago k.
        numerator = np.sum((data[k:] - mean) * (data[:-k] - mean))
        acf_values[k] = numerator / variance

    return acf_values

def calculate_pacf(data: np.ndarray, lags: int) -> np.ndarray:
    """
    Calcula la Función de Autocorrelación Parcial (PACF) para una serie de tiempo.
    Usa el método de resolución de ecuaciones de Yule-Walker para cada lag.

    Args:
        data (np.ndarray): La serie de tiempo (1D array).
        lags (int): El número máximo de rezagos para calcular.

    Returns:
        np.ndarray: Un array con los valores de la PACF para los rezagos 0 a `lags`.
                    El valor para el rezago 0 es NaN por convención.
    """
    n = len(data)
    if n <= lags:
        raise ValueError("El número de lags debe ser menor que la longitud de los datos.")

    # La PACF requiere los valores de la ACF para su cálculo.
    acf_values = calculate_acf(data, lags)
    pacf_values = np.zeros(lags + 1)
    
    # Por convención, la PACF en el rezago 0 no está definida.
    pacf_values[0] = np.nan 

    if lags >= 1:
        # La PACF en el rezago 1 es idéntica a la ACF en el rezago 1.
        pacf_values[1] = acf_values[1]

    # Calcula la PACF para los rezagos k = 2 hasta 'lags'.
    for k in range(2, lags + 1):
        # Construye la matriz R (matriz de Toeplitz de autocorrelaciones)
        # para el sistema de ecuaciones de Yule-Walker.
        R = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                R[i, j] = acf_values[abs(i - j)]
        
        # Construye el vector r de autocorrelaciones.
        r = acf_values[1 : k + 1]

        try:
            # Resuelve el sistema R * phi = r para encontrar los coeficientes phi
            # de un modelo AR(k) hipotético ajustado a los datos.
            phi_coeffs = np.linalg.solve(R, r)
            # La PACF en el rezago k es, por definición, el último coeficiente (phi_k)
            # de este modelo AR(k).
            pacf_values[k] = phi_coeffs[-1]
        except np.linalg.LinAlgError:
            # Maneja el caso de una matriz singular, que puede ocurrir con datos degenerados.
            pacf_values[k] = np.nan
            print(f"Advertencia: No se pudo calcular PACF para el rezago {k} debido a una matriz singular.")

    return pacf_values