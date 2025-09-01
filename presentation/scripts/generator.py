import numpy as np
import pandas as pd


def generate_var_process(T=500, burn_in=100):
    phi_1 = np.array([
        [ 0.5, -0.1,  0.2],
        [ 0.1,  0.3, -0.1],
        [-0.2,  0.1,  0.4]
    ])
    phi_2 = np.array([
        [ 0.15,  0.05, -0.1],
        [-0.05,  0.2,   0.1],
        [ 0.1,  -0.05,  0.1]
    ])
    phi = np.array([phi_1, phi_2])
    c = np.array([0.1, 0.05, 0.2])
    omega = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.2, 0.4],
        [0.3, 0.4, 0.9]
    ])

    k = phi[0].shape[0]  # Número de variables
    p = phi.shape[0]     # Orden del VAR (p=2 en este caso)

    # Inicializar la serie de tiempo con ceros
    data = np.zeros((T + burn_in, k))

    # Generar errores aleatorios multivariados
    errors = np.random.multivariate_normal(np.zeros(k), omega, T + burn_in)

    # Simular el proceso VAR
    for t in range(p, T + burn_in):
        lagged_terms = np.zeros(k)
        for i in range(p):
            lagged_terms += np.dot(phi[i], data[t - (i + 1)])
        data[t] = c + lagged_terms + errors[t]

    simulated_data = data[burn_in:]
    df_simulated = pd.DataFrame(simulated_data, 
        columns=[f'Y{i+1}' for i in range(simulated_data.shape[1])])
    return df_simulated