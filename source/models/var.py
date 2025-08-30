import numpy as np 

from source.models.base import TimeSeriesModel
from source.data.sampling import initialize_params

class VAR(TimeSeriesModel):

    def __init__(self,  
                 inp_dim=1,
                 p=1,
                 c=None, 
                 sigma=None,
                 params_distribution=None,
                 **kwargs):

        if isinstance(c, int):
            print('Repitiendo el valor de c')
            c = np.ones(inp_dim)*c

        if isinstance(sigma, int):
            print('Repitiendo el valor de sigma')
            sigma = np.ones(inp_dim)*sigma

        self.inp_dim = inp_dim
        self.p = p
        self.c = c
        self._is_stationary_cached = None
        
        phi_aux = initialize_params(params_distribution, p=p*inp_dim**2)
        self.phi = np.reshape(phi_aux, [p, inp_dim, inp_dim])

        # super().__init__(c=c, sigma=sigma)

    def get_unconditional_mean(self) -> float:
        """
        Calcula la media incondicional del proceso.
        """
        return None

    def get_unconditional_std(self) -> float:
        """
        Calcula la desviación estándar incondicional del proceso.
        """
        return None

    def get_irf(self, H: int = 20) -> np.ndarray:
        """
        Calcula la Función de Impulso-Respuesta (IRF).
        """
        return None

    def sample(self, n_samples: int, initial_values: np.ndarray = None, burn_in: int = 100) -> np.ndarray:
        """
        Genera una muestra simulada de la serie de tiempo.
        """
        return None

    def _is_stationary(self) -> bool:
        """
        Verifica si el proceso es estacionario usando un valor en caché.
        """
        if self._is_stationary_cached is None:
            self._is_stationary_cached, _ = self._check_stationarity()
        return self._is_stationary_cached
    
    def _check_stationarity(self) -> bool:
        """
        Revisa si los eigenvalores de la matriz F son menores a 1 en módulo.
        """
        if self.p == 0:
            return True
            
        F = self._build_F_matrix()
        eigenvalues = np.linalg.eigvals(F)
        return np.all(np.abs(eigenvalues) < 1), eigenvalues
    
    def fit(self, X, y):
        Pi = np.concatenate(self.phi, 0)
        Pi_c = np.zeros([1, Pi.shape[1]])
        Pi = np.concatenate([Pi_c, Pi, ]).T
        return Pi
    
    ## METODOS PRIVADOS
    def _build_F_matrix(self):
        if self.p == 0:
            return np.empty((0, 0))
        I = np.identity(self.inp_dim*(self.p-1))
        zeros = np.zeros([self.inp_dim*self.p-self.inp_dim, 
                          self.inp_dim])
        top    = np.hstack(self.phi)
        bottom = np.hstack([I, zeros])
        F = np.vstack([top, bottom])
        return F