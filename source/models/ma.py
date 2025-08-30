import numpy as np
from source.utils.sampling import initialize_params
from source.models.base import TimeSeriesModel

class MovingAverage(TimeSeriesModel):
    """
    Implementa un modelo de Media Móvil (MA).

    Define un proceso MA(q) y proporciona métodos para analizar sus propiedades
    y generar muestras de datos.

    La ecuación del proceso es:
    y_t = c + u_t + θ₁*u_{t-1} + ... + θq*u_{t-q}
    """
    def __init__(self,
                 c: float = 0.0,
                 sigma: float = 1.0,
                 params_distribution=None,
                 **kwargs):
        """
        Inicializa el modelo MA.

        Argumentos:
            c (float): El término constante del modelo, que es igual a la media incondicional.
            sigma (float): La desviación estándar del término de error (ruido blanco).
            params_distribution (list o np.ndarray): Vector de parámetros MA (coeficientes θ).
        """
        # Vector de parámetros de media móvil (theta_1, theta_2, ...)
        self.theta = initialize_params(params_distribution, **kwargs)
        # q: orden del proceso de media móvil
        self.q = len(self.theta) if self.theta is not None else 0

        super().__init__(c, sigma)

    def _is_stationary(self) -> bool:
        """
        Verifica la estacionariedad. Un proceso MA(q) de orden finito
        es siempre estacionario por definición.
        """
        return True

    def get_unconditional_mean(self) -> float:
        """
        Calcula la media incondicional. Para un modelo MA, la media es simplemente la constante 'c'.
        """
        return self.c

    def get_unconditional_std(self) -> float:
        """
        Calcula la desviación estándar incondicional del proceso MA.
        La varianza se calcula como: σ² * (1 + θ₁² + θ₂² + ... + θq²).
        """
        if self.theta is None: return None
        
        theta_sq_sum = np.sum(self.theta**2)
        variance = self.sigma**2 * (1 + theta_sq_sum)
        return np.sqrt(variance)

    def is_invertible(self) -> bool:
        """
        Verifica la invertibilidad del proceso.
        Un proceso MA es invertible si todas las raíces de su polinomio
        característico se encuentran fuera del círculo unitario.
        """
        if self.q == 0:
            return True

        # Coeficientes del polinomio: [1, theta_1, theta_2, ..., theta_q]
        coeffs = np.concatenate(([1], self.theta))
        roots = np.roots(coeffs)

        # La invertibilidad requiere que el módulo de todas las raíces sea > 1.
        return np.all(np.abs(roots) > 1)

    def get_irf(self, H: int = 20) -> np.ndarray:
        """
        Calcula la Función de Impulso-Respuesta (IRF) para un proceso MA(q).
        La IRF de un MA es finita y sus valores son [1, θ₁, ..., θq, 0, ...].
        """
        irf_values = np.zeros(H)
        irf_values[0] = 1  # El impacto de un shock en el momento 0 es 1.

        # El número de coeficientes a usar es el mínimo entre el horizonte (H-1) y el orden q.
        num_coeffs_to_use = min(H - 1, self.q)
        if num_coeffs_to_use > 0:
            irf_values[1:num_coeffs_to_use + 1] = self.theta[:num_coeffs_to_use]

        return irf_values

    def sample(self, n_samples: int, burn_in: int = 100, **kwargs) -> np.ndarray:
        """
        Genera una muestra de una serie de tiempo del proceso MA(q).

        Argumentos:
            n_samples (int): Longitud deseada de la serie de tiempo final.
            burn_in (int): Número de muestras iniciales a generar y descartar.
        """
        total_samples = n_samples + burn_in

        # 1. Generar todos los shocks (ruido blanco) de una vez.
        shocks = np.random.normal(loc=0, scale=self.sigma, size=total_samples)

        # 2. Inicializar el array para la serie de tiempo.
        y_sample = np.zeros(total_samples)

        # 3. Iterar en el tiempo para construir la serie.
        # El bucle empieza en 'q' porque se necesitan 'q' shocks pasados para el primer cálculo.
        for t in range(self.q, total_samples):
            # Obtener los últimos 'q' shocks (desde u_{t-1} hasta u_{t-q}).
            past_shocks = shocks[t-self.q:t]

            # Calcular la suma ponderada de los shocks pasados (componente MA).
            # Se invierten los shocks para alinear con el orden de theta (θ₁, θ₂, ...).
            ma_component = np.dot(self.theta, past_shocks[::-1])

            # Calcular y_t usando la fórmula del proceso MA(q).
            y_sample[t] = self.c + shocks[t] + ma_component

        # 4. Descartar el período de "burn-in" y retornar la muestra final.
        return y_sample[burn_in:]