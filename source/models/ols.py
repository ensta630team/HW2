import numpy as np

class OLS:
    """
    Implementa un modelo de regresión por Mínimos Cuadrados Ordinarios (OLS).

    Esta clase permite ajustar un modelo lineal a los datos y realizar predicciones.
    Maneja automáticamente la inclusión de un término de intercepto.
    """
    def __init__(self):
        """
        Inicializa el modelo OLS.
        """
        # Se inicializan los coeficientes como None hasta que se ajuste el modelo.
        self.beta = None       # Vector completo de coeficientes (incluyendo intercepto)
        self.intercept_ = None # Valor del intercepto (beta_0)
        self.coef_ = None      # Coeficientes de las variables (beta_1, beta_2, ...)

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Añade una columna de unos (intercepto) al inicio de la matriz de características X.
        """
        return np.c_[np.ones(X.shape[0]), X]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Ajusta el modelo OLS a los datos de entrenamiento.

        Calcula los coeficientes beta óptimos usando la ecuación normal:
        β = (X'X)⁻¹X'y

        Argumentos:
            X (np.ndarray): Matriz de características (variables independientes).
            y (np.ndarray): Vector de la variable objetivo (variable dependiente).
        """
        # Añade la columna de unos a la matriz X para el término del intercepto.
        X_con_intercepto = self._add_intercept(X)

        # Calcula los coeficientes beta usando la fórmula de la ecuación normal.
        factor_0 = np.linalg.inv(np.matmul(X_con_intercepto.T, X_con_intercepto))
        factor_1 = np.matmul(X_con_intercepto.T, y)
        self.beta = np.matmul(factor_0, factor_1)

        # Separa el intercepto y los demás coeficientes para mayor comodidad.
        self.intercept_ = self.beta[0]
        self.coef_ = self.beta[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones utilizando el modelo OLS ya ajustado.

        Argumentos:
            X (np.ndarray): Las muestras de entrada para las cuales hacer predicciones.

        Retorna:
            np.ndarray: Las predicciones calculadas.
        
        Lanza:
            ValueError: Si el modelo aún no ha sido ajustado con el método 'fit'.
        """
        if self.beta is None:
            raise ValueError("El modelo aún no ha sido ajustado. Llama primero al método 'fit'.")

        # Añade la columna de intercepto a los nuevos datos.
        X_con_intercepto = self._add_intercept(X)

        # Realiza las predicciones: y_pred = X * β
        y_pred = np.matmul(X_con_intercepto, self.beta)
        return y_pred