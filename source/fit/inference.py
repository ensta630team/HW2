import numpy as np
from scipy.stats import f as f_distribution
from source.fit.error import get_standard_error

def test_F(model, X, y_true, R, q, robust=False):
    """
    Realiza un test F para una hipótesis lineal (Rβ = q) usando un modelo ya ajustado.

    Esta función es un "wrapper" que automatiza la predicción y el cálculo
    de la matriz de covarianza necesaria para el test.

    Args:
        model: Una instancia de un modelo de regresión ya ajustado (ej. OLS).
               Debe tener un método `.predict(X)` y un atributo `.beta`.
        X (np.ndarray): La matriz de regresores (variables independientes), incluyendo el intercepto.
        y_true (np.ndarray): El vector de las observaciones reales.
        R (np.ndarray): La matriz que define las J restricciones lineales (dimensiones J x k).
        q (np.ndarray): El vector de valores para la hipótesis nula (J elementos).
        robust (bool): Si es True, usa errores robustos a heterocedasticidad (White).

    Returns:
        tuple: Una tupla con (estadistico_f, p_valor).
    """
    # Obtener predicciones y coeficientes del modelo ajustado
    y_pred = model.predict(X)
    beta_estimado = model.beta

    # Obtener dimensiones y calcular residuos
    n, k = X.shape
    J = R.shape[0]  # Número de restricciones

    # Calcular la matriz de varianza-covarianza de los coeficientes (V)
    s_hat = get_standard_error(X, y_true, y_pred, white=robust)
    V = np.diag(s_hat**2)

    # Calcular la matriz de varianza-covarianza de la combinación lineal Rβ
    var_Rb = R @ V @ R.T

    # alcular el estadístico F
    Rb_menos_q = R @ beta_estimado - q
    estadistico_f = (Rb_menos_q.T @ np.linalg.inv(var_Rb) @ Rb_menos_q) / J

    # Calcular el p-valor usando la distribución F
    p_valor = 1 - f_distribution.cdf(estadistico_f, dfn=J, dfd=n - k)

    return estadistico_f, p_valor