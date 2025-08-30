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
        
        # Vector de parámetros autorregresivos (phi_1, phi_2, ...)
        self.phi = initialize_params(params_distribution, **kwargs)
        # p: orden del proceso autorregresivo
        self.p = len(self.phi) if self.phi is not None else 0
        
        # Inicializa la autocovarianza como no calculada.
        self.autocov = None

        super().__init__(c, sigma)

    def _is_stationary(self) -> bool:
        """
        Verifica si el proceso es estacionario usando un valor en caché.
        """
        if self._is_stationary_cached is None:
            self._is_stationary_cached = self._check_stationarity()
        return self._is_stationary_cached

    def get_unconditional_mean(self) -> float:
        """
        Calcula la media incondicional del proceso.
        """
        if self.phi is None or not self._is_stationary():
            return None
        
        phi_sum = np.sum(self.phi)
        if np.isclose(phi_sum, 1.0):
            return np.inf
        
        self._unconditional_mean = self.c / (1 - phi_sum)
        return self._unconditional_mean
    
    def get_autocovariance(self) -> np.ndarray:
        """
        Calcula las primeras p autocovarianzas (γ₀ a γₚ₋₁) del proceso.
        Usa una fórmula matricial y guarda el resultado en caché.
        """
        # Si ya se ha calculado, devuelve el valor guardado.
        if self.autocov is not None:
            return self.autocov
        
        if not self._is_stationary():
            return None
        
        # Usa la fórmula vec(Γ₀) = (I - F⊗F)⁻¹vec(Σᵤ) para resolver.
        F = self._build_F_matrix()
        Ip2 = np.eye(self.p**2)
        FxF = np.kron(F, F)
        
        # Vector de varianzas del shock (solo V[0,0] es no nulo).
        vec_V = np.zeros(self.p**2)
        vec_V[0] = self.sigma**2
        
        # Resuelve para el vector de autocovarianzas.
        autocov_vec = np.linalg.inv(Ip2 - FxF) @ vec_V
        
        # Extrae las primeras p autocovarianzas (γ₀, γ₁, ..., γₚ₋₁).
        self.autocov = autocov_vec[0:self.p]

        return self.autocov

    def extend_autocovariance(self, order: int) -> np.ndarray:
        """
        Extiende la secuencia de autocovarianzas hasta el orden (rezago) especificado.

        Primero calcula las 'p' autocovarianzas base y luego las extiende
        recursivamente usando las ecuaciones de Yule-Walker.

        Args:
            order (int): El número total de autocovarianzas a devolver (de γ₀ a γₒᵣᏧₑᵣ₋₁).

        Returns:
            np.ndarray: Un array con la secuencia de autocovarianzas.
        """
        # 1. Asegurarse de tener las autocovarianzas base.
        base_autocov = self.get_autocovariance()
        if base_autocov is None:
            raise ValueError("No se pueden calcular autocovarianzas para un proceso no estacionario.")

        # 2. Si el orden pedido no es mayor que p, devolver las que ya tenemos.
        if order <= self.p:
            return base_autocov[:order]

        # 3. Extender recursivamente usando la ecuación de Yule-Walker.
        # γₖ = Σ(φᵢ * γₖ₋ᵢ)
        autocov_extended = list(base_autocov)
        for k in range(self.p, order):
            # Se toman las p autocovarianzas anteriores.
            past_gammas = np.array(autocov_extended[k - self.p : k])
            # Se invierten para el producto punto con [φ₁, ..., φₚ].
            next_gamma = np.dot(self.phi, past_gammas[::-1])
            autocov_extended.append(next_gamma)

        return np.array(autocov_extended)

    def get_unconditional_std(self) -> float:
        """
        Calcula la desviación estándar incondicional del proceso.
        """
        # Llama al método unificado para obtener las autocovarianzas.
        autocovs = self.get_autocovariance()
        
        if autocovs is None:
            return np.inf

        # La varianza (gamma_0) es el primer elemento.
        gamma_0 = autocovs[0]
        
        self._unconditional_std = np.sqrt(gamma_0) if gamma_0 > 0 else 0.0
        return self._unconditional_std

    def get_irf(self, H: int = 20, method='exact') -> np.ndarray:
        """
        Calcula la Función de Impulso-Respuesta (IRF).
        """
        if not self._is_stationary():
            raise ValueError("El proceso no es estacionario. La IRF no converge.")
        
        F = self._build_F_matrix()
        irf_values = calculate_irf(F, H, self.phi, method=method)
        return irf_values

    def sample(self, n_samples: int, initial_values: np.ndarray = None, burn_in: int = 0, seed: int = None) -> np.ndarray:
        '''
        Genera una muestra (una serie de tiempo simulada) del proceso AR(p).
        '''
        if seed is not None:
            np.random.seed(seed)
        
        total_samples = n_samples + burn_in
        y_sample = np.zeros(total_samples)
        wnoise = np.random.normal(0, self.sigma, total_samples)
        
        mean_val = self.get_unconditional_mean()
        std_val  = self.get_unconditional_std()

        if initial_values is not None:
            y_sample[:self.p] = initial_values
        elif self._is_stationary():
            y_sample[:self.p] = np.random.normal(mean_val, std_val, self.p)

        for t in range(self.p, total_samples):
            y_prev = y_sample[t-self.p:t]
            y_sample[t] = self.c + np.dot(self.phi, y_prev[::-1]) + wnoise[t]
        
        return y_sample[burn_in:]
    
    # ======================================================
    # Métodos privados de la clase
    # ======================================================

    def _build_F_matrix(self) -> np.ndarray:
        """
        Construye la matriz acompañante (F).
        """
        if self.p == 0:
            return np.empty((0, 0))
            
        F = np.zeros((self.p, self.p))
        F[0, :] = self.phi
        if self.p > 1:
            np.fill_diagonal(F[1:], 1)
        
        return F
    
    def _check_stationarity(self) -> bool:
        """
        Revisa si los eigenvalores de la matriz F son menores a 1 en módulo.
        """
        if self.p == 0:
            return True
            
        F = self._build_F_matrix()
        eigenvalues = np.linalg.eigvals(F)
        return np.all(np.abs(eigenvalues) < 1)