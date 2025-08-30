import functools
import multiprocess
import os
import time
from tqdm.auto import tqdm
import pandas as pd


def monte_carlo_mp(n_iterations: int = 1000, n_jobs: int = -1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if n_jobs == -1:
                num_processes = 1
            else:
                num_processes = min(n_jobs, os.cpu_count())

            print(f"Iniciando simulación paralela con {n_iterations} iteraciones en {num_processes} procesos...")
            
            worker_func = functools.partial(func, *args, **kwargs)
            
            with multiprocess.Pool(processes=num_processes) as pool:
                results = list(tqdm(pool.imap(worker_func, range(n_iterations)), 
                                    total=n_iterations, 
                                    desc="Simulación Monte Carlo"))
            results = pd.DataFrame(results)
            results = results.to_dict(orient='list')
            return results
        return wrapper
    return decorator

def monte_carlo(n_iterations: int = 1000):
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
                step_result = func(*args, **kwargs)
                # Guarda el resultado de la iteración
                results.append(step_result)
            results = pd.DataFrame(results)
            results = results.to_dict(orient='list')
            return results
        return wrapper
    return decorator