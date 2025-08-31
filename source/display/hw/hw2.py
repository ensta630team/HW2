import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Optional, List

# --- (Se asume la misma configuración de estilo que proporcionaste) ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})
FONT_SIZES = {'title': 16, 'label': 14, 'legend': 12, 'tick': 12}


def plot_real_vs_generated(
    real_data: np.ndarray, 
    pre_fit_sample: np.ndarray, 
    post_fit_sample: np.ndarray,
    variable_names: Optional[List[str]] = None,
    fig: Optional[plt.Figure] = None,
    axes: Optional[np.ndarray] = None,
    title: str = r'Comparacion: Serie Real vs. Series Generadas'
) -> tuple[plt.Figure, np.ndarray]:
    """
    Grafica la serie real y las muestras generadas, usando un subplot por cada variable.

    Argumentos:
        real_data (np.ndarray): Array con los datos observados.
        pre_fit_sample (np.ndarray): Array con datos muestreados del modelo no ajustado.
        post_fit_sample (np.ndarray): Array con datos muestreados del modelo ajustado.
        variable_names (List[str], opcional): Nombres para cada variable (columna).
        fig (plt.Figure, opcional): Figura de matplotlib existente.
        axes (np.ndarray, opcional): Array de ejes de matplotlib existentes.
        title (str, opcional): Título general para la figura.

    Retorna:
        tuple[plt.Figure, np.ndarray]: La figura y el array de ejes utilizados.
    """
    n_vars = real_data.shape[1]

    # --- 1. Manejo de Nombres de Variables y Datos ---
    if variable_names is None:
        variable_names = [f'Variable {i+1}' for i in range(n_vars)]
    
    # Convertimos a DataFrame para facilitar el manejo
    real_df = pd.DataFrame(real_data, columns=variable_names)
    pre_fit_df = pd.DataFrame(pre_fit_sample, columns=variable_names)
    post_fit_df = pd.DataFrame(post_fit_sample, columns=variable_names)

    # --- 2. Manejo de Figura y Ejes ---
    if axes is None:
        # Se crean n_vars subplots, compartiendo el eje X
        fig, axes = plt.subplots(
            n_vars, 1, 
            figsize=(14, 5 * n_vars), 
            sharex=True
        )
        # Si solo hay una variable, axes no es un array, lo convertimos para consistencia
        if n_vars == 1:
            axes = np.array([axes])
    else:
        fig = axes.flatten()[0].get_figure()

    # Se establece un título general para toda la figura
    fig.suptitle(title, fontsize=FONT_SIZES['title'])

    # --- 3. Iteración y Trazado en cada Subplot ---
    for i, var_name in enumerate(variable_names):
        ax = axes[i] # Seleccionamos el subplot actual
        
        # Trazar la serie real (línea sólida)
        ax.plot(real_df.index, real_df[var_name], 
                label='Real', color='black', linestyle='-')
        
        # Trazar la muestra antes del ajuste (línea punteada)
        ax.plot(pre_fit_df.index, pre_fit_df[var_name], 
                label='Antes de Ajuste', color='blue', linestyle=':')
        
        # Trazar la muestra después del ajuste (línea discontinua)
        ax.plot(post_fit_df.index, post_fit_df[var_name], 
                label='Despues de Ajuste', color='red', linestyle='--')

        # --- 4. Formato de cada Subplot ---
        ax.set_title(f'{var_name}', fontsize=FONT_SIZES['label'])
        ax.set_ylabel('Valor', fontsize=FONT_SIZES['label'])
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
        ax.legend(fontsize=FONT_SIZES['legend'])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Se añade la etiqueta del eje X solo al último subplot para evitar redundancia
    axes[-1].set_xlabel('Tiempo', fontsize=FONT_SIZES['label'])
    
    return fig, axes
