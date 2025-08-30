import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

from source.statistics import calculate_acf, calculate_pacf

# --- Configuración de Estilo para los Gráficos ---
# Se utiliza el estilo 'seaborn-v0_8-whitegrid' para un aspecto limpio.
plt.style.use('seaborn-v0_8-whitegrid')

# Configura matplotlib para usar LaTeX en el renderizado de texto,
# lo que permite una tipografía de alta calidad, ideal para reportes académicos.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.sans-serif": ["Computer Modern Sans serif"],
    "text.latex.preamble": r"\usepackage{amsmath}"
})

# Define tamaños de fuente estándar para consistencia en todos los gráficos.
FONT_SIZES = {
    'title': 16,
    'label': 14,
    'legend': 12,
    'tick': 12
}

def plot_irf_comparison(irf_exact, irf_simulation, model, ax=None):
    """
    Grafica una comparación entre la Función de Impulso-Respuesta (IRF) exacta y la simulada.

    Argumentos:
        irf_exact (np.ndarray): IRF calculada con el método matricial exacto.
        irf_simulation (np.ndarray): IRF calculada mediante simulación.
        model: El objeto del modelo, que debe tener un atributo 'p' para el orden AR.
        ax (matplotlib.axes.Axes, opcional): Un array preexistente de dos ejes para graficar.

    Retorna:
        matplotlib.axes.Axes: El array de ejes utilizado para el gráfico.
    """
    H = len(irf_exact)
    periods = np.arange(H)
    if ax is None:
        fig, ax_array = plt.subplots(1, 2, figsize=(14, 6))
    else:
        ax_array = ax
        fig = ax_array[0].get_figure()

    # Primer subplot: Comparación de ambas IRF.
    ax_array[0].plot(periods, irf_exact, 'o-', label='Método Exacto (Matricial)', linewidth=2, markersize=6, color='darkblue')
    ax_array[0].plot(periods, irf_simulation, 'x--', label='Método Simulación', color='darkred', linewidth=2, markersize=7)
    ax_array[0].set_title(f'Comparación IRF de un Proceso \n AR({model.p}) con Horizonte H = {H}', fontsize=FONT_SIZES['title'])
    ax_array[0].set_xlabel('Períodos (h)', fontsize=FONT_SIZES['label'])
    ax_array[0].set_ylabel('Respuesta al Impulso', fontsize=FONT_SIZES['label'])
    ax_array[0].axhline(0, color='black', linestyle='--', linewidth=0.8)

    # Segundo subplot: Diferencia (residuos) entre los dos métodos.
    ax_array[1].plot(periods, irf_exact - irf_simulation, color='k')
    ax_array[1].set_title('Residuos entre métodos', fontsize=FONT_SIZES['title'])
    ax_array[1].set_xlabel('Períodos (h)', fontsize=FONT_SIZES['label'])
    ax_array[1].set_ylabel('Diferencia de IRF \n(exacto - simulacion)', fontsize=FONT_SIZES['label'])
    ax_array[1].axhline(0, color='black', linestyle='--', linewidth=0.8)

    for ax_item in ax_array:
        ax_item.grid(True, which='both', linestyle=':', linewidth=0.7)
        ax_item.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])

    # Crea una única leyenda para toda la figura.
    handles, labels = ax_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=FONT_SIZES['legend'])

    return ax_array

def save_figure(figure, path, **kwargs):
    """
    Guarda una figura de matplotlib en la ruta especificada.

    Argumentos:
        figure (matplotlib.figure.Figure): La figura a guardar.
        path (str): La ruta donde se guardará la figura.
    """
    # Crea el directorio si no existe.
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    figure.savefig(path, bbox_inches='tight', dpi=300, **kwargs)
    print(f"✅ Figura guardada exitosamente en {path}")

def plot_serie(serie, model, ax=None, **kwargs):
    """
    Grafica una serie de tiempo.

    Argumentos:
        serie (np.ndarray): Los datos de la serie de tiempo a graficar.
        model: El objeto del modelo, debe tener un atributo 'p'.
        ax (matplotlib.axes.Axes, opcional): Un eje preexistente para graficar.

    Retorna:
        matplotlib.axes.Axes: El eje utilizado para el gráfico.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    x = np.arange(len(serie))
    ax.plot(x, serie, marker='.', color='darkred')
    ax.set_title(f'Serie de Tiempo AR({model.p}) {len(serie)} Periodos', fontsize=FONT_SIZES['title'])
    ax.set_xlabel('Períodos (h)', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Valor', fontsize=FONT_SIZES['label'])
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])

    return ax

def plot_series_p3(dataset, fig=None, axes=None):
    """
    Grafica tres series de tiempo de un dataset en subplots separados.

    Argumentos:
        dataset (pd.DataFrame): DataFrame con las columnas 't', 'pi_t', 'y_t', 'i_t'.
        fig, axes: Figura y ejes preexistentes (opcional).

    Retorna:
        tuple: Una tupla con la figura y el array de ejes.
    """
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    x_labels_tick = dataset['t'][1::50]

    axes[0].plot(dataset['t'][1:], dataset['pi_t'], linestyle='-', color='darkred', label=r'$\pi_t$ $(\Delta$ IPC)')
    axes[1].plot(dataset['t'][1:], dataset['y_t'], linestyle='-', color='darkblue', label=r'$y_t$ ($\Delta$ IMACEC)')
    axes[2].plot(dataset['t'], dataset['i_t'], linestyle='-', color='darkgreen', label=r'$i_t$ (PM)')

    axes[1].set_ylabel('Valor', fontsize=FONT_SIZES['label'])
    axes[2].set_xlabel('Periodo', fontsize=FONT_SIZES['label'])

    for ax in axes:
        ax.set_xticks(x_labels_tick)
        ax.set_xticklabels(x_labels_tick, rotation=0, fontsize=FONT_SIZES['tick'])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='y', which='major', labelsize=FONT_SIZES['tick'])

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, fontsize=FONT_SIZES['legend'])
    return fig, axes

def plot_acf_pacf(data: np.ndarray, lags: int = 40, alpha: float = 0.05, title_suffix: str = "", fig=None, axes=None):
    """
    Grafica las funciones de autocorrelación (ACF) y autocorrelación parcial (PACF).

    Argumentos:
        data (np.ndarray): La serie de tiempo (array 1D).
        lags (int): El número máximo de rezagos a graficar.
        alpha (float): Nivel de significancia para los intervalos de confianza.
        title_suffix (str): Sufijo para añadir a los títulos de los gráficos.
        fig, axes: Figura y ejes preexistentes (opcional).

    Retorna:
        tuple: Una tupla con la figura y el array de ejes.
    """
    if len(data) < 2:
        print("No hay suficientes datos para calcular ACF/PACF.")
        return

    acf_vals = calculate_acf(data, lags)
    pacf_vals = calculate_pacf(data, lags)
    conf_level = 1.96 / np.sqrt(len(data))

    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Graficar ACF
    axes[0].stem(range(lags + 1), acf_vals, markerfmt='k.')
    axes[0].axhspan(-conf_level, conf_level, color='gray', alpha=0.2)
    axes[0].set_title(f'Función de Autocorrelación (ACF) {title_suffix}', fontsize=FONT_SIZES['title'])
    axes[0].set_ylabel('Autocorrelación', fontsize=FONT_SIZES['label'])

    # Graficar PACF
    axes[1].stem(np.arange(1, lags + 1), pacf_vals[1:], markerfmt='k.')
    axes[1].axhspan(-conf_level, conf_level, color='gray', alpha=0.2)
    axes[1].set_title(f'Función de Autocorrelación Parcial (PACF) {title_suffix}', fontsize=FONT_SIZES['title'])
    axes[1].set_xlabel('Rezagos', fontsize=FONT_SIZES['label'])

    for ax in axes:
        ax.axhline(0, color='k', linestyle='--')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
        ax.set_xticks(range(0, lags + 1, max(1, lags // 10)))

    if lags > 0:
        axes[1].set_xlim(0.5, lags + 0.5)

    return fig, axes

def plot_monthly_boxplot(t, y, title: str = "Distribución Mensual de la Serie", xlabel: str = "Mes", ylabel: str = "Valor", fig=None, ax=None):
    """
    Crea un boxplot para visualizar la distribución de una serie de tiempo por cada mes del año.
    Es útil para detectar patrones estacionales.

    Argumentos:
        t (array-like): Componente de tiempo (ej. 'YYYY/MM').
        y (array-like): Componente de valor de la serie.
        title, xlabel, ylabel: Títulos de los ejes y del gráfico.
        fig, ax: Figura y eje preexistentes (opcional).

    Retorna:
        tuple: Una tupla con la figura y el eje.
    """
    data = pd.Series(y, index=pd.to_datetime(t, format='%Y/%m'))

    # Prepara los datos para el gráfico.
    df = pd.DataFrame({'value': data.values, 'month': data.index.month}).sort_values('month')

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    sns.boxplot(x='month', y='value', data=df, ax=ax, color='#F0F0F0', medianprops={'color': '#006400', 'linewidth': 1.5})

    # Personalización del gráfico.
    month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    ax.set_xticklabels(month_names, fontsize=FONT_SIZES['tick'])
    ax.set_title(title, fontsize=FONT_SIZES['title'], weight='bold')
    ax.set_xlabel(xlabel, fontsize=FONT_SIZES['label'])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES['label'])
    ax.tick_params(axis='y', which='major', labelsize=FONT_SIZES['tick'])
    
    # Añade línea de mediana global para referencia.
    global_median = df['value'].median()
    ax.axhline(global_median, color='#004080', linestyle='--', linewidth=1., label=f'Mediana Global ({global_median:.3f})')
    ax.legend(fontsize=FONT_SIZES['legend'])

    return fig, ax

def plot_hannan_rissanen_results(results, fig=None, axes=None):
    """
    Grafica la distribución de los mejores órdenes p y q obtenidos del algoritmo Hannan-Rissanen.

    Argumentos:
        results (dict): Diccionario con los resultados, incluyendo 'freq_p', 'freq_q', 'best_p', 'best_q'.
        fig, axes: Figura y ejes preexistentes (opcional).
    """
    freq_p = results['freq_p']
    freq_q = results['freq_q']
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Gráfico de barras para la frecuencia de p y q.
    sns.barplot(x=np.arange(len(freq_p)), y=freq_p, ax=axes[0], color='darkblue', alpha=0.5)
    sns.barplot(x=np.arange(len(freq_q)), y=freq_q, ax=axes[1], color='darkblue', alpha=0.5)
    
    # Destaca el mejor p y q encontrados.
    axes[0].plot(results['best_p'], freq_p[results['best_p']], "*", markersize=10, color="darkred", label=f'Mejor p: {results["best_p"]}')
    axes[1].plot(results['best_q'], freq_q[results['best_q']], "*", markersize=10, color="darkred", label=f'Mejor q: {results["best_q"]}')

    axes[0].set_xlabel('Valores de p', fontsize=FONT_SIZES['label'])
    axes[1].set_xlabel('Valores de q', fontsize=FONT_SIZES['label'])
    axes[0].set_ylabel('Densidad', fontsize=FONT_SIZES['label'])

    for ax in axes:
        ax.set_xticks(np.arange(0, len(freq_q), 2))
        ax.legend(fontsize=FONT_SIZES['legend'], facecolor='whitesmoke', edgecolor='gray', shadow=True)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])

    return fig, axes

def plot_bootstrap_kde(bootstrap_results: dict, fig=None, axes=None):
    """
    Visualiza la distribución de los órdenes p y q del bootstrap usando gráficos de Densidad de Kernel (KDE).

    Argumentos:
        bootstrap_results (dict): Diccionario devuelto por la función de bootstrap.
        fig, axes: Figura y ejes preexistentes (opcional).
    """
    dist_df = bootstrap_results['param_distribution']
    best_p = bootstrap_results['best_p']
    best_q = bootstrap_results['best_q']
    
    # Reconstruye las listas de todos los p y q encontrados para el KDE.
    all_p = [p for (p, _), freq in dist_df.iterrows() for _ in range(int(freq))]
    all_q = [q for (_, q), freq in dist_df.iterrows() for _ in range(int(freq))]

    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico KDE para p
    sns.kdeplot(x=all_p, ax=axes[0], color='darkblue', linewidth=2.5, fill=True, alpha=0.1)
    axes[0].axvline(x=best_p, color='red', linestyle='--', linewidth=2, label=f'Mejor p: {best_p}')
    axes[0].set_title('Distribución del Mejor p', fontsize=FONT_SIZES['title'])
    axes[0].set_xlabel('Orden AR (p)', fontsize=FONT_SIZES['label'])
    
    # Gráfico KDE para q
    sns.kdeplot(x=all_q, ax=axes[1], color='darkblue', linewidth=2.5, fill=True, alpha=0.1)
    axes[1].axvline(x=best_q, color='red', linestyle='--', linewidth=2, label=f'Mejor q: {best_q}')
    axes[1].set_title('Distribución del Mejor q', fontsize=FONT_SIZES['title'])
    axes[1].set_xlabel('Orden MA (q)', fontsize=FONT_SIZES['label'])

    for ax in axes:
        ax.set_ylabel('Densidad', fontsize=FONT_SIZES['label'])
        ax.legend(fontsize=FONT_SIZES['legend'])
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
        ax.set_xlim(left=-0.5)

    plt.tight_layout()
    plt.show()
    
    return fig, axes

def plot_forecast(train_t, train_y, test_t, test_y, forecasts, lower_ci, upper_ci, model_name="ARMA", fig=None, axes=None):
    """
    Crea un gráfico con pronósticos, valores reales y análisis de residuos.

    Argumentos:
        train_t, train_y: Datos de tiempo y valores de entrenamiento.
        test_t, test_y: Datos de tiempo y valores reales de prueba.
        forecasts, lower_ci, upper_ci: Pronósticos y sus intervalos de confianza.
        model_name (str): Nombre del modelo para el título.
        fig, axes: Figura y ejes preexistentes (opcional).
    """
    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Subplot 1: Pronóstico vs. Valores Reales
    ax1 = axes[0]
    ax1.plot(train_t, train_y, label='Datos de Entrenamiento', color='black', linewidth=1.5)
    ax1.plot(test_t, test_y, marker='o', label='Valores Reales (Prueba)', color='darkred', linestyle='', markersize=3)
    ax1.plot(test_t, forecasts, label='Pronóstico Puntual', color='blue', linestyle='--')
    ax1.fill_between(test_t, lower_ci, upper_ci, color='blue', alpha=0.2, label='Intervalo de Confianza (95%)')
    ax1.set_ylabel('Valor de la Serie', fontsize=FONT_SIZES['label'])
    
    # Calcular métricas de error (RMSE y R²)
    residuals = test_y - forecasts
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((test_y - np.mean(test_y))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    ax1.set_title(f'Pronóstico del Modelo {model_name}\nRMSE: {rmse:.4f} | R²: {r2:.4f}', fontsize=FONT_SIZES['title'], weight='bold')
    
    # Subplot 2: Análisis de Residuos
    ax2 = axes[1]
    ax2.stem(test_t, residuals, basefmt="black", linefmt='grey', markerfmt='D')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_title('Residuos del Pronóstico', fontsize=FONT_SIZES['title']-2)
    ax2.set_xlabel('Fecha', fontsize=FONT_SIZES['label'])
    ax2.set_ylabel('Error', fontsize=FONT_SIZES['label'])
    
    for ax in [ax1, ax2]:
        ax.legend(fontsize=FONT_SIZES['legend'])
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)
        ax.tick_params(axis='y', which='major', labelsize=FONT_SIZES['tick'])

    fig.autofmt_xdate()
    plt.tight_layout(pad=2.0)
    plt.show()

    return fig, axes

def plot_irf(irf_values: np.ndarray, model=None, model_name: str = "ARMA", fig=None, ax=None):
    """
    Grafica la Función de Impulso-Respuesta (IRF) para un modelo de series de tiempo.

    Argumentos:
        irf_values (np.ndarray): Array 1D con los valores de la IRF.
        model_name (str, opcional): Nombre del modelo para el título.
        fig, ax: Figura y eje preexistentes (opcional).
    """
    H = len(irf_values)
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(np.arange(H), irf_values, 'o-', label='Respuesta al Impulso', linewidth=2, markersize=4, color='darkblue')
    ax.axhline(0, color='black', linestyle='--', linewidth=1.0)
    
    ax.set_title(f'Función de Impulso-Respuesta (IRF) \n Para Modelo {model_name}({model.q}, {model.p})', fontsize=FONT_SIZES['title'], weight='bold')
    ax.set_xlabel(r'Períodos (Horizonte $H$)', fontsize=FONT_SIZES['label'])
    ax.set_ylabel(r'Respuesta de $y_t$ a un shock de $u_t$', fontsize=FONT_SIZES['label'])
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
    ax.set_xlim(left=-1, right=H)
    
    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_empirical_distribution(mean_samples: np.ndarray, var_samples: np.ndarray, model_name: str = "", fig=None, axes=None):
    """
    Grafica las distribuciones empíricas de medias y varianzas muestrales.

    Argumentos:
        mean_samples (np.ndarray): Array 1D de medias muestrales de simulaciones.
        var_samples (np.ndarray): Array 1D de varianzas muestrales de simulaciones.
        model_name (str, opcional): Nombre del modelo para los títulos.
        fig, axes: Figura y ejes preexistentes (opcional).
    """
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Distribución empírica de la media
    axes[0].hist(mean_samples, bins=50, density=True, alpha=0.7, color='darkblue', label='Densidad Empírica')
    axes[0].set_xlabel('Valor de la Media Muestral', fontsize=FONT_SIZES['label'])
    axes[0].set_title(f'Distribución Empírica de la Media\n{model_name}', fontsize=FONT_SIZES['title'], weight='bold')

    # Subplot 2: Distribución empírica de la varianza
    axes[1].hist(var_samples, bins=50, density=True, alpha=0.7, color='darkgreen', label='Densidad Empírica')
    axes[1].set_xlabel('Valor de la Varianza Muestral', fontsize=FONT_SIZES['label'])
    axes[1].set_title(f'Distribución Empírica de la Varianza\n{model_name}', fontsize=FONT_SIZES['title'], weight='bold')

    for ax in axes:
        ax.set_ylabel('Densidad', fontsize=FONT_SIZES['label'])
        ax.legend(fontsize=FONT_SIZES['legend'])
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])

    fig.tight_layout(pad=2.0)
    plt.show()

    return fig, axes

def plot_simulation_paths(simulation_data: np.ndarray, model_name: str = "", num_paths_to_plot: int = 3, fig=None, axes=None):
    """
    Grafica un número específico de trayectorias de simulación.

    Argumentos:
        simulation_data (np.ndarray): Array 2D donde cada columna es una trayectoria de simulación.
        model_name (str, opcional): Nombre del modelo para el título principal.
        num_paths_to_plot (int, opcional): Número de trayectorias a graficar.
    """
    num_paths_to_plot = min(num_paths_to_plot, simulation_data.shape[1])

    if fig is None or axes is None:
        fig, axes = plt.subplots(1, num_paths_to_plot, figsize=(5 * num_paths_to_plot, 4), sharey=True)
    
    if num_paths_to_plot == 1:
        axes = [axes] # Asegura que `axes` sea siempre iterable.

    for i, ax in enumerate(axes):
        ax.plot(simulation_data[:, i], color='darkblue', linewidth=1.5)
        ax.set_title(f'Serie {i + 1}', fontsize=FONT_SIZES['title'] - 2)
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['tick'])
        ax.set_xlabel('Período', fontsize=FONT_SIZES['label'])
        if i == 0:
            ax.set_ylabel('Valor', fontsize=FONT_SIZES['label'])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes