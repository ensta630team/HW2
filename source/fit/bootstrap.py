# HW2/source/fit/bootstrap.py

import numpy as np
import sympy as sp
import statsmodels.api as sm
from source.models.var import VAR
from source.statistics.impulse_response import compute_IRF_VAR
from source.data.transform import create_lagged_dataset
from source.models.ols import OLS
from source.fit.inference import test_F

def _generate_one_bootstrap_sample(
    original_series: np.ndarray, 
    initial_model: OLS, 
    lags: int
) -> np.ndarray:
    """
    Genera una única serie de tiempo remuestreada basada en los residuos de un modelo inicial.
    (Versión mejorada de la función 'remuestracion' original).
    """
    # 1. Obtener residuos y coeficientes del modelo original
    residuals = original_series[lags:] - initial_model.predict(initial_model.X)
    beta_hat = initial_model.beta
    
    # 2. Mantener las primeras 'lags' observaciones como valores iniciales
    resampled_series = original_series[:lags].tolist()
    
    n_residuals = len(residuals)
    
    # 3. Generar el resto de la serie de forma autorregresiva
    for i in range(len(original_series) - lags):
        # Muestrear un residuo con reposición
        random_residual = residuals[np.random.randint(0, n_residuals)]
        
        # Tomar los 'lags' valores anteriores de la *nueva* serie generada
        previous_values = np.array(resampled_series[i : i + lags][::-1])
        
        # Crear el vector de regresores [1, y_{t-1}, y_{t-2}, ...]
        predictor = np.insert(previous_values, 0, 1)

        # Predecir el siguiente valor y añadir el residuo aleatorio
        next_value = predictor @ beta_hat + random_residual
        resampled_series.append(next_value)
        
    return np.array(resampled_series)


def run_bootstrap_simulation(
    original_series: np.ndarray, 
    lags: int, 
    n_reps: int, 
    R: np.ndarray = None, 
    H0: np.ndarray = None
) -> dict:
    """
    Orquesta la simulación de bootstrap para un modelo autorregresivo.

    Argumentos:
        original_series (np.ndarray): La serie de tiempo original.
        lags (int): El número de rezagos del modelo AR.
        n_reps (int): El número de replicaciones de bootstrap a generar.
        R, H0 (np.ndarray, opcional): Matriz y vector para pruebas de hipótesis.

    Retorna:
        dict: Un diccionario con la distribución de 'betas' y 'f_stats' de la simulación.
    """
    # 1. Ajustar el modelo OLS inicial sobre los datos originales
    X_orig, y_orig = create_lagged_dataset(original_series, lag=lags)
    initial_model = OLS()
    initial_model.fit(X_orig, y_orig)

    bootstrapped_betas = []
    bootstrapped_f_stats = []

    for _ in range(n_reps):
        # Generar una nueva serie de tiempo
        new_sample = _generate_one_bootstrap_sample(original_series, initial_model, lags)
        
        # Ajustar un modelo OLS a la nueva serie
        X_boot, y_boot = create_lagged_dataset(new_sample, lag=lags)
        boot_model = OLS()
        boot_model.fit(X_boot, y_boot)
        bootstrapped_betas.append(boot_model.beta)

        # Si se proporcionan R y H0, calcular el estadístico F
        if R is not None and H0 is not None:
            f_stat, _ = test_F(boot_model, X_boot, y_boot, R, H0)
            bootstrapped_f_stats.append(f_stat)

    return {
        'betas': np.array(bootstrapped_betas),
        'f_stats': np.array(bootstrapped_f_stats)
    }


def calculate_bootstrap_ci(
    param_distribution: np.ndarray, 
    original_estimate: float, 
    alphas: list = [0.10, 0.05, 0.01]
) -> dict:
    """
    Calcula los intervalos de confianza de Efron (percentil) y Hall (pivotal)
    a partir de una distribución de parámetros de bootstrap.

    Argumentos:
        param_distribution (np.ndarray): Array 1D con la distribución de un parámetro (e.g., betas[:, 1]).
        original_estimate (float): El coeficiente estimado con la muestra original.
        alphas (list): Lista de niveles de significancia para los intervalos.

    Retorna:
        dict: Un diccionario anidado con los intervalos 'efron' y 'hall' para cada alpha.
    """
    results = {'efron': {}, 'hall': {}}
    
    for alpha in alphas:
        # Intervalo de Efron (percentil simple)
        lower_efron = np.percentile(param_distribution, (alpha / 2) * 100)
        upper_efron = np.percentile(param_distribution, (1 - alpha / 2) * 100)
        results['efron'][alpha] = (lower_efron, upper_efron)

        # Intervalo de Hall (pivotal)
        diff_dist = param_distribution - original_estimate
        lower_hall = original_estimate - np.percentile(diff_dist, (1 - alpha / 2) * 100)
        upper_hall = original_estimate - np.percentile(diff_dist, (alpha / 2) * 100)
        results['hall'][alpha] = (lower_hall, upper_hall)
        
    return results


def get_bootstrap_critical_values(f_stat_distribution: np.ndarray, alphas: list = [0.10, 0.05, 0.01]) -> dict:
    """
    Obtiene los valores críticos empíricos de una distribución de estadísticos F de bootstrap.

    Argumentos:
        f_stat_distribution (np.ndarray): Array 1D con los estadísticos F de la simulación.
        alphas (list): Lista de niveles de significancia.

    Retorna:
        dict: Un diccionario con el valor crítico para cada nivel de alpha.
    """
    critical_values = {}
    for alpha in alphas:
        critical_values[alpha] = np.percentile(f_stat_distribution, (1 - alpha) * 100)
        
    return critical_values


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================

# Intervalos de Confianza Bootstraps
def bootstrap_IRF_intervals(X, p, H, n_boot=1000, alpha=0.05):
    """
    Bootstrap completo que re-estima el VAR en cada réplica
    """
    T, n = X.shape
    IRF_boot = []
    
    # Estimar modelo original para obtener coeficientes iniciales
    model = sm.tsa.VAR(X)
    results = model.fit(p)
    Phi_original = [results.coefs[i] for i in range(p)]
    omega_original = results.sigma_u
    residuals_original = results.resid
    
    for i in range(n_boot):
        try:
            # Generar residuos bootstrap (resampling con reemplazo)
            indices = np.random.choice(T-p, size=T-p, replace=True)
            residuals_boot = residuals_original[indices]
            
            # Generar datos bootstrap recursivamente
            X_boot = np.zeros_like(X)
            X_boot[:p] = X[:p]  # Valores iniciales
            
            for t in range(p, T):
                # Componente determinística (constantes, etc.)
                deterministic = results.intercept if hasattr(results, 'intercept') else 0
                
                # Componente autorregresiva
                ar_component = 0
                for lag in range(1, p+1):
                    ar_component += np.dot(Phi_original[lag-1], X_boot[t-lag])
                
                X_boot[t] = deterministic + ar_component + residuals_boot[t-p]
            
            # Re-estimar VAR con datos bootstrap
            
            # Calcular IRF ========
            VARaux = VAR(inp_dim=X_boot.shape[1], p=p)
            VARaux.fit(X_boot)
            omega_boot = VARaux.omega_hat
            Psi_boot = VARaux._compute_psi_sequence(H)
            IRF_sequence = [compute_IRF_VAR(Psi_boot, omega_boot, s) for s in range(H)]
            IRF_boot.append(IRF_sequence)
            
        except Exception as e:
            print(f"Error en la réplica {i+1}: {e}")
            continue
    
    # Calcular intervalos de confianza
    IRF_boot = np.array(IRF_boot)
    lower_percentile = 100 * alpha/2
    upper_percentile = 100 * (1 - alpha/2)

    IRF_lower = np.percentile(IRF_boot, lower_percentile, axis=0)
    IRF_upper = np.percentile(IRF_boot, upper_percentile, axis=0)
    
    return IRF_lower, IRF_upper