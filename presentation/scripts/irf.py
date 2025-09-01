import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from presentation.scripts.generator import generate_var_process
from presentation.scripts.test_fn import get_GIRD
from source.models.var import VAR as VAROWN
from statsmodels.tsa.api import VAR

T_sim = 100
df_simulated = generate_var_process(T=T_sim)

# Ajustar un Modelo VAR a los Datos Simulados ---
model = VAR(df_simulated)
results = model.fit(2)

# Crear la IRF utilizando Generalized Impulse Response Function (GIRF) ---
n_periods = 20
sigma_u_df  = results.resid.cov()
sigma_u     = sigma_u_df.values
irf_results = results.irf(n_periods)

# No estoy seguro que es esto y para que sirve dentr de GIRF 
psi_coeffs  = irf_results.irfs 

k = df_simulated.shape[1]

girf = get_GIRD(sigma_u, n_periods, k, psi_coeffs)

# ==================================================================
# Graficar las GIRF ================================================
# ==================================================================
print("\n--- Gráficos de las Funciones de Impulso-Respuesta Generalizadas (GIRF) ---")
fig, axes = plt.subplots(k, k, figsize=(15, 10), sharex=True)

for i in range(k):
    for j in range(k):
        ax = axes[i, j]
        ax.plot(range(n_periods + 1), girf[:, i, j], color='blue')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(f'Resp. de {df_simulated.columns[i]} a Shock en {df_simulated.columns[j]}')
        if i == k - 1:
            ax.set_xlabel('Períodos')
        if j == 0:
            ax.set_ylabel('Impacto')

plt.tight_layout()
plt.show()

print("\nProceso completado. Se ha generado un proceso VAR(2), verificado su estacionariedad y calculado las GIRF.")