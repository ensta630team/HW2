import numpy as np
from numpy.polynomial.polynomial import polyfromroots

# Diccionario que mapea nombres de distribuciones a funciones lambda para generar parámetros aleatorios.
# Esto permite crear parámetros de forma flexible a partir de diferentes distribuciones de probabilidad.
DISTRIBUTION_GENERATORS = {
    'normal': lambda **kwargs: np.random.normal(
        loc=kwargs.get('mean', 0.0),      # Media por defecto = 0
        scale=kwargs.get('std', 1.0),       # Desv. Estándar por defecto = 1
        size=kwargs['p']),
    'exponential': lambda **kwargs: np.random.exponential(
        scale=kwargs.get('scale', 1.0),   # Escala por defecto = 1
        size=kwargs['p']),
    'uniform': lambda **kwargs: np.random.uniform(
        low=kwargs.get('low', 0.0),       # Límite inferior por defecto = 0
        high=kwargs.get('high', 1.0),     # Límite superior por defecto = 1
        size=kwargs['p']),
    'beta': lambda **kwargs: np.random.beta(
        a=kwargs.get('alpha', 1.0),       # Alpha por defecto = 1
        b=kwargs.get('beta', 1.0),        # Beta por defecto = 1
        size=kwargs['p']),
    'gamma': lambda **kwargs: np.random.gamma(
        shape=kwargs['shape'], 
        scale=kwargs.get('scale', 1.0), 
        size=kwargs['p']),
    'poisson': lambda **kwargs: np.random.poisson(
        lam=kwargs.get('lambda', 1.0), 
        size=kwargs['p'])
}

def generate_stationary_phi(p: int) -> np.ndarray:
    """
    Genera un conjunto de coeficientes phi para un proceso AR(p) que es estacionario por construcción.
    
    El método consiste en generar raíces aleatorias dentro del círculo unitario en el plano complejo
    y luego construir el polinomio característico correspondiente para derivar los coeficientes phi.
    """
    # Genera raíces complejas conjugadas para asegurar que los coeficientes del polinomio sean reales.
    num_complex_pairs = p // 2
    radii = np.sqrt(np.random.uniform(0, 1, size=num_complex_pairs))
    thetas = np.random.uniform(0, 2 * np.pi, size=num_complex_pairs)
    complex_roots = radii * np.exp(1j * thetas)
    roots = np.concatenate([complex_roots, np.conjugate(complex_roots)])

    # Si p es impar, se necesita una raíz real adicional entre -1 y 1.
    if p % 2 != 0:
        real_root = np.random.uniform(-1, 1, size=1)
        roots = np.concatenate([roots, real_root])
        
    # Construye el polinomio a partir de las raíces.
    poly_coeffs = polyfromroots(roots)
    
    # Convierte los coeficientes del polinomio a los coeficientes phi del proceso AR.
    phi = -np.real(poly_coeffs[:-1][::-1])
    return phi.reshape(1, p)

def initialize_params(params_distribution, **kwargs):
    """
    Inicializa los parámetros de un modelo de forma flexible.

    Puede recibir los parámetros directamente, generarlos para asegurar estacionariedad,
    o crearlos a partir de una distribución de probabilidad específica.
    """
    # Si no se especifica distribución, no hay parámetros.
    if params_distribution is None:
        return None

    # Si se pasan los parámetros directamente como una lista o array.
    if isinstance(params_distribution, (list, np.ndarray)):
        return np.array(params_distribution)

    # Si se solicita generar parámetros que garanticen estacionariedad.
    if params_distribution == 'stationary':
        return generate_stationary_phi(kwargs.get('p', 2)).flatten()

    # Busca la distribución solicitada en el diccionario de generadores.
    generator_func = DISTRIBUTION_GENERATORS.get(params_distribution)

    if generator_func:
        try:
            # Llama a la función generadora correspondiente.
            return generator_func(**kwargs)
        except KeyError as e:
            raise TypeError(f"Falta el argumento requerido: {e} para la distribución '{params_distribution}'")
    else:
        raise ValueError(f"'{params_distribution}' no es un valor válido para params_distribution.")
    
