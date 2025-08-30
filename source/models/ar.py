import numpy as np
from source.data.sampling import initialize_params
from source.statistics import calculate_irf
from scipy.linalg import solve_discrete_lyapunov
from source.models.base import TimeSeriesModel

class AutoRegressive(TimeSeriesModel):
    """
    Define un modelo de series de tiempo autorregresivo AR(p).
    """
    def __init__(self, 
                 c: float = 0.0, 
                 sigma: float = 1.0,
                 params_distribution=None,
                 **kwargs):
        
        # Constructor de la clase. Inicializa los parámetros del modelo.
        
        # Vector de parámetros autorregresivos (phi_1, phi_2, ...)
        self.phi = initialize_params(params_distribution, **kwargs)
        # p: orden del proceso autorregresivo
        self.p = len(self.phi) if self.phi is not None else 0

        super().__init__(c, sigma)

    def _is_stationary(self) -> bool:
        """
        Verifica si el proceso es estacionario.
        Un proceso es estacionario si todos los eigenvalores de su matriz 
        acompañante tienen un módulo menor a 1.
        Usa un valor en caché para no recalcularlo repetidamente.
        """
        if self._is_stationary_cached is None:
            self._is_stationary_cached = self._check_stationarity()
        return self._is_stationary_cached

    def get_unconditional_mean(self) -> float:
        """
        Calcula la media incondicional (el valor promedio a largo plazo) del proceso.
        Solo está definida para procesos estacionarios.
        """
        if self.phi is None or not self._is_stationary:
            return None
        
        phi_sum = np.sum(self.phi)

        # Si la suma de los coeficientes es 1, hay una raíz unitaria y la media no está definida.
        if np.isclose(phi_sum, 1.0):
            return np.inf
        
        self._unconditional_mean = self.c / (1 - phi_sum)
        return self._unconditional_mean
    
    def get_autocovariance(self):
        if self._is_stationary is None:
            self._check_stationarity()()
        
        if not self._is_stationary:
            return None
        
        F = self._build_F_matrix()
        Ip2 = np.eye(self.p**2,self.p**2)
        FxF = np.kron(F, F)
        autocov = self.sigma**2 * np.linalg.inv(Ip2-FxF)
        self.autocov = autocov[0:self.p,0]

        return self.autocov
    
    def get_unconditional_std(self) -> float:
        """
        Calcula la desviación estándar incondicional del proceso.
        Resuelve las ecuaciones de Yule-Walker para encontrar la varianza (gamma_0).
        """
        if not self._is_stationary():
            return np.inf
        
        # Usa un valor en caché para evitar recalcular.
        if self._unconditional_std is None:
            gamma_0 = self._solve_yule_walker_for_variance()
            # Asegura que la varianza no sea negativa por errores numéricos.
            self._unconditional_std = np.sqrt(gamma_0) if gamma_0 > 0 else 0.0
        
        return self._unconditional_std

    def get_irf(self, H: int = 20, method='exact') -> np.ndarray:
        """
        Calcula la Función de Impulso-Respuesta (IRF).
        Muestra el efecto de un shock o impulso a lo largo de H periodos.
        """
        if not self._is_stationary:
            raise ValueError("El proceso no es estacionario. La IRF no converge.")
        
        F = self._build_F_matrix()
        irf_values = calculate_irf(F, H, self.phi, method=method)
        return irf_values

    def sample(self, n_samples: int, initial_values: np.ndarray = None, burn_in: int = 100) -> np.ndarray:
        '''
        Genera una muestra (una serie de tiempo simulada) del proceso AR(p).
        
        La ecuación del proceso es:
        y_t = c + phi_1*y_{t-1} + ... + phi_p*y_{t-p} + u_t
        donde u_t es ruido blanco con desviación estándar sigma.

        Argumentos:
            n_samples (int): Número de muestras a generar.
            initial_values (np.ndarray, opcional): Valores iniciales para la serie. 
                                                   Si es None, se usan ceros o valores de la distribución incondicional.
            burn_in (int): Número de muestras iniciales a descartar para asegurar la estabilidad del proceso.

        Retorna:
            Un array de numpy con la serie de tiempo generada.
        '''
        total_samples = n_samples + burn_in
        y_sample = np.zeros(total_samples)
        wnoise = np.random.normal(0, self.sigma, total_samples)

        # Define los valores iniciales de la serie.
        if initial_values is not None:
            y_sample[:self.p] = initial_values
        elif self._is_stationary:
            # Si es estacionario, los saca de la distribución incondicional.
            y_sample[:self.p] = np.random.normal(self._unconditional_mean, self._unconditional_std, self.p)

        # Genera la serie de tiempo iterativamente.
        for t in range(self.p, total_samples):
            y_prev = y_sample[t-self.p:t]
            # Calcula el valor actual basado en los valores anteriores y un shock aleatorio.
            # Invierte los valores previos para que coincidan con el orden de phi en el producto punto.
            y_sample[t] = self.c + np.dot(self.phi, y_prev[::-1]) + wnoise[t]
        
        # Descarta el período de "burn-in" y retorna la muestra final.
        return y_sample[burn_in:]
    
    # ======================================================
    # Métodos privados de la clase
    # ======================================================

    def _build_F_matrix(self) -> np.ndarray:
        """
        Construye la matriz acompañante (F) para la representación 
        de espacio de estados del proceso AR(p).
        """
        if self.p == 0:
            return np.empty((0, 0))
            
        F = np.zeros((self.p, self.p))
        # La primera fila contiene los coeficientes phi.
        F[0, :] = self.phi
        # La subdiagonal se llena de unos para desplazar los valores en el tiempo.
        if self.p > 1:
            np.fill_diagonal(F[1:], 1)
        
        return F
    
    def _solve_yule_walker_for_variance(self) -> float:
        """
        Resuelve la ecuación de Lyapunov discreta para obtener la matriz de covarianza.
        """
        if not self._is_stationary:
            return np.inf

        if self.p == 0:
            return self.sigma**2
            
        F = self._build_F_matrix()
        V = np.zeros((self.p, self.p))
        # La varianza del shock solo afecta al primer elemento.
        V[0, 0] = self.sigma**2

        # Resuelve Gamma = F * Gamma * F' + V usando la función de scipy.
        gamma_matrix = solve_discrete_lyapunov(F, V)
        # La varianza incondicional del proceso (gamma_0) es el primer elemento de la matriz de covarianza.
        return gamma_matrix[0, 0]
 
    def _check_stationarity(self) -> bool:
        """
        Implementación que revisa si los eigenvalores de la matriz F 
        son todos menores a 1 en módulo.
        """
        if self.p == 0:
            return True
            
        F = self._build_F_matrix()
        eigenvalues = np.linalg.eigvals(F)
        return np.all(np.abs(eigenvalues) < 1)