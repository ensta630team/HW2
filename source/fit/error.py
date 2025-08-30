import numpy as np
import pandas as pd

def get_standard_error(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, white: bool = False) -> np.ndarray:
    """
    Calcula los errores estándar para los coeficientes de un modelo de regresión lineal.

    Puede calcular tanto los errores estándar clásicos (bajo el supuesto de homocedasticidad)
    como los errores estándar robustos a heterocedasticidad de White.

    Args:
        X (np.ndarray): La matriz de regresores (variables independientes).
        y_true (np.ndarray): El vector de las observaciones reales (variable dependiente).
        y_pred (np.ndarray): El vector de las predicciones del modelo.
        white (bool): Si es True, calcula los errores estándar robustos de White.
                      Si es False (por defecto), calcula los errores estándar clásicos.

    Returns:
        np.ndarray: Un array 1D con los errores estándar estimados para cada coeficiente en X.
    """

    n, k = X.shape
    
    # Rresiduos del modelo
    residuos = y_true - y_pred
    
    try:
        inv_XTX = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        raise ValueError("La matriz X'X es singular y no se puede invertir. Revisa si hay multicolinealidad perfecta.")

    if white:
        # Errores Estándar Robustos (White)
        Omega = np.diag(residuos**2)
        # Fórmula: Var(β̂) = (X'X)⁻¹ (X'ΩX) (X'X)⁻¹
        matriz_cov_robusta = inv_XTX @ (X.T @ Omega @ X) @ inv_XTX
        varianza_beta = np.diag(matriz_cov_robusta)
    else:
        # Calculo de Errores Estándar (homocedasticidad).
        # Fórmula: s² = Σ(eᵢ²)/(n-k)
        sigma_cuadrado_hat = (residuos.T @ residuos) / (n - k)
        # Fórmula: Var(β̂) = s² * (X'X)⁻¹
        matriz_cov_clasica = sigma_cuadrado_hat * inv_XTX
        varianza_beta = np.diag(matriz_cov_clasica)

    errores_estandar = np.sqrt(varianza_beta)

    return errores_estandar