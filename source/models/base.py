import numpy as np
from abc import ABC, abstractmethod
from source.display.print import check_num


class TimeSeriesModel(ABC):
    """
    Clase base abstracta (plantilla) para modelos de series de tiempo univariadas.
    
    Esta clase define una interfaz común para todos los modelos (AR, MA, ARMA),
    asegurando que todos compartan un conjunto esencial de métodos y propiedades.
    """

    def __init__(self, 
                 c: 0.0, 
                 sigma: 1.0):
        """
        Inicializador de la clase base.
        
        Argumentos:
            c (float): El término constante del modelo.
            sigma (float): La desviación estándar del término de error (ruido blanco).
        """
        if sigma <= 0:
            raise ValueError("Sigma (desviación estándar del error) debe ser un valor positivo.")
        
        # Atributos comunes a todos los modelos de series de tiempo.
        self.c = c
        self.sigma = sigma

        # Inicializa variables para guardar en caché resultados de cálculos.
        self._is_stationary_cached = None
        self._unconditional_mean   = None
        self._unconditional_std    = None

        # Calcula las propiedades incondicionales al crear el objeto.
        self._unconditional_mean = self.get_unconditional_mean()
        self._unconditional_std = self.get_unconditional_std()

    @abstractmethod
    def get_unconditional_mean(self) -> float:
        """
        Calcula la media incondicional del proceso.
        Este método DEBE ser implementado por cada subclase.
        """
        pass

    @abstractmethod
    def get_unconditional_std(self) -> float:
        """
        Calcula la desviación estándar incondicional del proceso.
        Este método DEBE ser implementado por cada subclase.
        """
        pass

    @abstractmethod
    def get_irf(self, H: int = 20) -> np.ndarray:
        """
        Calcula la Función de Impulso-Respuesta (IRF).
        Este método DEBE ser implementado por cada subclase.
        """
        pass

    @abstractmethod
    def sample(self, n_samples: int, initial_values: np.ndarray = None, burn_in: int = 100) -> np.ndarray:
        """
        Genera una muestra simulada de la serie de tiempo.
        Este método DEBE ser implementado por cada subclase.
        """
        pass

    @abstractmethod
    def _is_stationary(self) -> bool:
        """
        Verifica si el modelo es estacionario.
        Este método DEBE ser implementado por cada subclase.
        """
        pass

    def __str__(self) -> str:
        """
        Genera un resumen del modelo en formato de texto.
        Adapta la salida para modelos AR o MA según los atributos que encuentre.
        """
        is_stationary = self._is_stationary()

        # Determina si el modelo es AR o MA para mostrar el resumen correcto.
        if 'p' in self.__dict__.keys():
            header = f"AR({self.p}) Model Summary"
            order = self.p
            coefs = self.phi
            coefname = 'phi'
            
        if 'q' in self.__dict__.keys():
            header = f"MA({self.q}) Model Summary"
            order = self.q
            coefs = self.theta
            coefname = 'theta'
                    
        separator = "=" * 50

        # Construye las líneas del resumen.
        summary_lines = [
            separator,
            f"{header:^50}",
            separator,
            f"{'Model Order':<25}: {check_num(order)}",
            f"{'Intercept (c)':<25}: {check_num(self.c)}",
            f"{'Error Std Dev (sigma)':<25}: {check_num(self.sigma)}",
            f"{'Is Stationary':<25}: {is_stationary}",
            f"{'Mu (Unc. Mean):':<25}: {check_num(self._unconditional_mean)}",
            f"{'Sigma (Unc. Std.)':<25}: {check_num(self._unconditional_std)}",
        ]
        
        summary_lines.append("-" * 50)
        summary_lines.append(f"Coefficients ({coefname}):")
        
        # Agrega los coeficientes del modelo al resumen.
        if order > 0:
            for i, coef in enumerate(coefs):
                summary_lines.append(f"  {coefname}_{i+1:<4} = {coef: >5.2f}")
        else:
            summary_lines.append("  (No coefficients)")
            
        summary_lines.append(separator)
        
        return "\n".join(summary_lines)