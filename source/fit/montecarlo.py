import functools
import multiprocessing
import os
import time
from tqdm.auto import tqdm

def monte_carlo_mp(n_iterations: int = 1000, n_jobs: int = -1):
    """
    Decorador para ejecutar una simulación de Monte Carlo en paralelo.

    Args:
        n_iterations (int): El número total de iteraciones a ejecutar.
        n_jobs (int): El número de procesos a utilizar. 
                      -1 significa usar todos los núcleos de CPU disponibles.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determinar el número de procesos a usar
            if n_jobs == -1:
                num_processes = os.cpu_count()
            else:
                num_processes = min(n_jobs, os.cpu_count())

            print(f"Iniciando simulación paralela con {n_iterations} iteraciones en {num_processes} procesos...")

            worker_func = functools.partial(func, *args, **kwargs)
            
            # Creamos el pool de procesos
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = list(tqdm(pool.imap(worker_func, range(n_iterations)), 
                                    total=n_iterations, 
                                    desc="Simulación Monte Carlo"))

            return results
        return wrapper
    return decorator

def monte_carlo(n_iterations: int = 1000, pass_iteration: bool = False):
    """
    Decorador para ejecutar una función múltiples veces en una simulación de Monte Carlo.

    Args:
        n_iterations (int): El número de veces que se ejecutará la función.
        pass_iteration (bool): Si es True, pasa el número de la iteración actual
                               a la función decorada como un argumento de palabra
                               clave llamado 'iteration'.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):            
            # Lista para almacenar los resultados de cada iteración
            results = []
            
            # Bucle principal de la simulación con una barra de progreso
            for i in tqdm(range(n_iterations), desc="Simulación Monte Carlo"):
                
                # Prepara los argumentos para la función original
                current_kwargs = kwargs.copy()
                if pass_iteration:
                    current_kwargs['iteration'] = i
                
                # Ejecuta la función original (una iteración de la simulación)
                step_result = func(*args, **current_kwargs)
                
                # Guarda el resultado de la iteración
                results.append(step_result)
            
            return results
        return wrapper
    return decorator