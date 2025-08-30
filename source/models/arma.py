import pandas as pd
import numpy as np

from source.models.base import TimeSeriesModel
from source.optimize.likelihood import maximum_likelihood_estimation
from source.models.ar import AutoRegressive
from source.models.ma import MovingAverage
from tqdm import tqdm


class ARMA:
    """
    Implementa un modelo Autorregresivo de Media Móvil (ARMA).
    
    Define un proceso ARMA(p,q), que combina componentes AR y MA.
    La ecuación del proceso es:
    y_t = c + φ₁*y_{t-1} + ... + φp*y_{t-p} + u_t + θ₁*u_{t-1} + ... + θq*u_{t-q}
    """
    def __init__(self,
                 c: float = 0.0,
                 sigma: float = 1.0,
                 phi_params=None,
                 theta_params=None,
                 **kwargs):
        """
        Inicializa el modelo ARMA.
        
        Argumentos:
            c (float): Término constante del modelo.
            sigma (float): Desviación estándar del término de error (ruido blanco).
            phi_params (list o np.ndarray): Vector de parámetros AR (coeficientes φ).
            theta_params (list o np.ndarray): Vector de parámetros MA (coeficientes θ).
        """
        # --- Configuración del componente AR ---
        # Se crea un objeto AutoRegressive para manejar la parte AR del modelo.
        self.ar = AutoRegressive(c, sigma, params_distribution=phi_params, **kwargs)

        # --- Configuración del componente MA ---
        # Se crea un objeto MovingAverage para manejar la parte MA del modelo.
        self.ma = MovingAverage(c, sigma, theta_params, **kwargs)
        
        # --- Configuración General ---
        self.c = c
        self.sigma = sigma
        self._unconditional_std = None # Caché para la desviación estándar incondicional.

    @property
    def params(self) -> dict:
        """Retorna un diccionario con los parámetros phi y theta."""
        return {'phi': self.ar.phi, 'theta': self.ma.theta}
    
    @property
    def p(self) -> int:
        """Retorna el orden (p) del componente AR."""
        return self.ar.p

    @property
    def q(self) -> int:
        """Retorna el orden (q) del componente MA."""
        return self.ma.q
        
    @property
    def phi(self) -> np.ndarray:
        """Retorna los coeficientes (phi) del componente AR."""
        return self.ar.phi

    @property
    def theta(self) -> np.ndarray:
        """Retorna los coeficientes (theta) del componente MA."""
        return self.ma.theta
    
    def _is_stationary(self) -> bool:
        """
        Verifica la estacionariedad. Un proceso ARMA es estacionario
        si y solo si su componente AR es estacionario.
        """
        return self.ar._is_stationary()

    def get_unconditional_mean(self) -> float:
        """
        Calcula la media incondicional, que solo depende del componente AR.
        """
        return self.ar.get_unconditional_mean()

    def get_unconditional_std(self) -> float:
        """
        Calcula la desviación estándar incondicional del proceso ARMA.
        Se calcula usando la representación MA(inf) y la Función de Impulso-Respuesta (IRF).
        """
        if self._unconditional_std is not None:
            return self._unconditional_std
        if not self._is_stationary():
            return np.inf
        # Si no hay parte AR o MA, delega el cálculo al componente correspondiente.
        if self.p == 0:
            return self.ma.get_unconditional_std()
        if self.q == 0:
            return self.ar.get_unconditional_std()
            
        # Aproxima la suma infinita con un horizonte grande (H=1000).
        H = 1000
        psi_coeffs = self.get_irf(H=H)
        # Var(y_t) = sigma^2 * sum(psi_j^2)
        variance = self.sigma**2 * np.sum(psi_coeffs**2)
        self._unconditional_std = np.sqrt(variance)
        return self._unconditional_std

    def sample(self, n_samples: int, 
               initial_values: np.ndarray = None, 
               burn_in: int = 0, 
               **kwargs) -> np.ndarray:
        """
        Genera una muestra de una serie de tiempo del proceso ARMA(p,q).
        
        Argumentos:
            n_samples (int): Longitud final de la serie de tiempo.
            burn_in (int): Número de muestras iniciales a descartar.
        
        Retorna:
            np.ndarray: La serie de tiempo generada.
        """
        total_samples = n_samples + burn_in
        
        # Genera todos los shocks (errores) de una vez para eficiencia.
        shocks = np.random.normal(loc=0, scale=self.sigma, size=total_samples)
        y_sample = np.zeros(total_samples)

        if initial_values is not None:
            y_sample[:self.p] = initial_values

        # El bucle debe empezar después del máximo de los retardos p y q.
        max_lag = max(self.ar.p, self.ma.q)

        # Itera en el tiempo para construir la serie.
        for t in range(max_lag, total_samples):
            # Parte AR: producto de phi con los valores pasados de y.
            past_y = y_sample[t-self.ar.p:t]
            ar_component = np.dot(self.ar.phi, past_y[::-1]) if self.ar.p > 0 else 0
            
            # Parte MA: producto de theta con los shocks pasados.
            past_shocks = shocks[t-self.ma.q:t]
            ma_component = np.dot(self.ma.theta, past_shocks[::-1]) if self.ma.q > 0 else 0
            
            # Combina todos los componentes para obtener el valor actual.
            y_sample[t] = self.c + ar_component + shocks[t] + ma_component
            
        return y_sample[burn_in:]
    
    def get_irf(self, H: int = 20) -> np.ndarray:
        """
        Calcula la Función de Impulso-Respuesta (IRF) para un proceso ARMA(p,q).
        
        Argumentos:
            H (int): Horizonte para el cual calcular la IRF.
        
        Retorna:
            np.ndarray: Un array con los valores de la IRF.
        """
        # Prepara vectores extendidos para los coeficientes.
        phi_ext = np.zeros(H)
        theta_ext = np.zeros(H)

        if self.p > 0:
            phi_ext[1:self.p + 1] = self.phi
        if self.q > 0:
            theta_ext[1:self.q + 1] = self.theta

        irf_values = np.zeros(H)
        irf_values[0] = 1 # El efecto de un shock en el momento cero es 1.

        # Calcula recursivamente los valores de la IRF.
        for j in range(1, H):
            ar_part = np.dot(phi_ext[1:j + 1], irf_values[j-1::-1]) if self.p > 0 else 0
            ma_part = theta_ext[j] if self.q > 0 else 0
            irf_values[j] = ar_part + ma_part
            
        return irf_values
    
    def forecast_arma(self, y_data, h: int):
        """
        Realiza pronósticos fuera de muestra para un modelo ARMA ajustado.

        Argumentos:
            y_data (np.ndarray): La serie de tiempo histórica.
            h (int): El horizonte de pronóstico (número de períodos a predecir).

        Retorna:
            tuple: Pronósticos, límite inferior y límite superior del intervalo de confianza.
        """
        phi = self.phi if self.p > 0 else np.array([])
        theta = self.theta if self.q > 0 else np.array([])
        sigma = self.sigma
        n = len(y_data)
        
        # --- 1. Reconstruir los errores históricos a partir de los datos. ---
        errors = np.zeros(n)
        for t in range(max(self.p, self.q), n):
            y_past = y_data[t-self.p:t][::-1]
            u_past = errors[t-self.q:t][::-1]
            ar_part = np.dot(phi, y_past) if self.p > 0 else 0
            ma_part = np.dot(theta, u_past) if self.q > 0 else 0
            errors[t] = y_data[t] - self.c - ar_part - ma_part

        # --- 2. Realizar el pronóstico de forma recursiva. ---
        forecasts = np.zeros(h)
        y_extended = np.concatenate([y_data, np.zeros(h)])
        errors_extended = np.concatenate([errors, np.zeros(h)])

        for i in range(h):
            t = n + i
            y_past = y_extended[t-self.p:t][::-1]
            u_past = errors_extended[t-self.q:t][::-1] # Para el futuro, los errores esperados son cero.

            ar_part = np.dot(phi, y_past) if self.p > 0 else 0
            ma_part = np.dot(theta, u_past) if self.q > 0 else 0
            
            # El pronóstico puntual se calcula y se guarda.
            forecast_point = self.c + ar_part + ma_part
            forecasts[i] = forecast_point
            y_extended[t] = forecast_point # Se usa para el siguiente pronóstico.

        # --- 3. Calcular los intervalos de confianza. ---
        # La varianza del error de pronóstico depende de la IRF.
        irf_coeffs = self.get_irf(H=h)
        
        # La varianza del error de pronóstico se acumula con el tiempo.
        forecast_error_var = np.cumsum(irf_coeffs**2) * sigma**2
        forecast_se = np.sqrt(forecast_error_var)
        
        # Intervalo de confianza al 95% (z ≈ 1.96).
        z_score = 1.96
        lower_bound = forecasts - z_score * forecast_se
        upper_bound = forecasts + z_score * forecast_se

        return forecasts, lower_bound, upper_bound

    def __str__(self) -> str:
        """Genera un resumen del modelo en formato de texto."""
        is_stationary = self._is_stationary()
        header = f"ARMA({self.p}, {self.q}) Model Summary"
        separator = "=" * 50
        
        unc_mean = self.get_unconditional_mean()
        unc_std = self.get_unconditional_std()

        summary_lines = [
            separator, f"{header:^50}", separator,
            f"{'Is Stationary':<25}: {is_stationary}",
            f"{'Mu (Unc. Mean)':<25}: {unc_mean:.4f}" if unc_mean is not None else "N/A",
            f"{'Sigma (Unc. Std.)':<25}: {unc_std:.4f}" if unc_std is not None else "N/A",
            f"{'Error Std Dev (sigma)':<25}: {self.sigma:.4f}",
            f"{'Constant (c)':<25}: {self.c:.4f}",
            "-" * 50, "AR Coefficients (phi):"
        ]
        
        if self.p > 0:
            for i, coef in enumerate(self.phi):
                summary_lines.append(f"  phi_{i+1:<4} = {coef: >.4f}")
        else:
            summary_lines.append("  (No AR coefficients)")
            
        summary_lines.extend(["-" * 50, "MA Coefficients (theta):"])
        
        if self.q > 0:
            for i, coef in enumerate(self.theta):
                summary_lines.append(f"  theta_{i+1:<4} = {coef: >.4f}")
        else:
            summary_lines.append("  (No MA coefficients)")
        
        summary_lines.append(separator)
        return "\n".join(summary_lines)