import numpy as np

def get_covariance_matrix(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, white: bool = False) -> np.ndarray:
    """
    Calcula la matriz de Varianza-Covarianza para los coeficientes de un modelo.
    """
    n, k = X.shape
    residuos = y_true - y_pred
    
    try:
        inv_XTX = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        raise ValueError("La matriz X'X es singular. Revisa si hay multicolinealidad.")

    if white:
        # Errores Estándar Robustos (White)
        Omega = np.diag(residuos**2)
        cov_matrix = inv_XTX @ (X.T @ Omega @ X) @ inv_XTX
    else:
        # Varianza estimada del error del modelo (sigma^2)
        sigma_cuadrado_hat = (residuos.T @ residuos) / (n - k)
        # Matriz de Varianza-Covarianza
        cov_matrix = sigma_cuadrado_hat * inv_XTX

    # DEVUELVE LA MATRIZ DIRECTAMENTE
    return np.nan_to_num(cov_matrix, nan=0.0)

def get_standard_error(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, white: bool = False) -> np.ndarray:
    """
    Calcula los errores estándar (la raíz de la DIAGONAL de la matriz de covarianza).
    """
    # Llama a la función anterior para obtener la matriz completa
    cov_matrix = get_covariance_matrix(X, y_true, y_pred, white=white)
    
    # Extrae la diagonal (varianzas) y LUEGO calcula la raíz cuadrada
    standard_errors = np.sqrt(np.diag(cov_matrix))
    return standard_errors