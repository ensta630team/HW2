import numpy as np
from scipy.stats import f as f_distribution
from source.fit.error import get_covariance_matrix

def test_F(model, X, y_true, R, H0, robust=False):
    # ... (código anterior) ...
    y_pred = model.predict(X)
    beta_estimado = model.beta

    n, k = X.shape
    J = R.shape[0]

    # --- LÍNEA CORREGIDA ---
    # Llamamos a la función que devuelve la matriz de covarianza completa
    V = get_covariance_matrix(X, y_true, y_pred, white=robust)

    Rb = R @ beta_estimado
    var_Rb = R @ V @ R.T
    
    # ... (resto del código es correcto) ...
    
    try:
        inv_var_Rb = np.linalg.inv(var_Rb)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
        
    fobs = ((Rb - H0).T @ inv_var_Rb @ (Rb - H0)) / J
    p_valor = 1 - f_distribution.cdf(fobs, dfn=J, dfd=n - k)

    return fobs, p_valor