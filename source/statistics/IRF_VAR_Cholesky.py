import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm

# Determinar los Ψ_i_s por recurrencia
def compute_psi_sequence(Phi, p, n_terms):
    """
    Calcula la sucesión Ψ_i para i = 0, 1, 2, ..., n_terms-1
    
    Argumentos:
        Phi     : lista de p matrices (Φ₁, Φ₂, ..., Φₚ)
        p       : número natural (cantidad de matrices Φ)
        n_terms : número de términos Ψ a calcular
    
    Retorna:
        Lista de matrices Ψ₀, Ψ₁, Ψ₂, ..., Ψ_{n_terms-1}
    """
    
    # Verificar que tenemos exactamente p matrices Φ
    if len(Phi) != p:
        raise ValueError(f"Se esperaban {p} matrices Φ, pero se recibieron {len(Phi)}")
    
    # Obtener las dimensiones de las matrices
    matrix_shape = Phi[0].shape
    n = matrix_shape[0]  # tamaño de la matriz
    
    # Inicializar la lista de Ψ
    Psi = []
    
    # Ψ₀ = Id (matriz identidad)
    Psi0 = np.eye(n)
    Psi.append(Psi0)
    
    # Calcular los términos restantes
    for s in range(1, n_terms):
        # Inicializar la suma
        suma = np.zeros((n, n))
        
        # Calcular: Ψ_s = sum_{i=1}^p Φ_i * Ψ_{s-i}
        for i in range(1, p+1):
            if s - i >= 0: 
                suma += np.dot(Phi[i-1], Psi[s-i])
        Psi.append(suma)
    
    return Psi

################################################################################
################################################################################

# Cálculo del IRF mediante descomposición de Cholesky
def compute_IRF_VAR(Psi, omega, s):
    """
    Cálculo corregido de IRF usando descomposición de Cholesky
    """
    # Descomposición de Cholesky (triangular inferior)
    L = cholesky(omega, lower=True)
    
    # Cálculo del IRF
    IRF_s = np.dot(Psi[s], L)
    
    return IRF_s

################################################################################
################################################################################

# Intervalos de confianza
def bootstrap_IRF_intervals(X, Phi, omega, p, n_terms, n_boot=1000, alpha=0.05):
    """
    Calcula intervalos de confianza para IRF usando bootstrap
    
    Argumentos:
        X        : Matriz de datos original (T × n)
        Phi      : Lista de matrices de coeficientes estimados
        omega    : Matriz de covarianza estimada
        p        : Número de retardos
        n_terms  : Número de horizontes
        n_boot   : Número de réplicas bootstrap
        alpha    : Nivel de significancia
    
    Retorna:
        IRF_lower, IRF_upper: Límites inferior y superior para cada horizonte
    """
    T, n = X.shape
    IRF_boot = []
    
    # Bootstrap
    for i in range(n_boot):
        # Generar datos bootstrap
        residuals_boot = np.random.multivariate_normal(np.zeros(n), omega, T)
        X_boot = np.zeros_like(X)
        
        # Inicializar con datos reales
        X_boot[:p] = X[:p]
        
        # Generar serie bootstrap
        for t in range(p, T):
            X_boot[t] = np.sum([np.dot(Phi[j], X_boot[t-j-1]) for j in range(p)], axis=0)
            X_boot[t] += residuals_boot[t]
        
        # Re-estimar VAR y calcular IRF (simplificado)
        # En la práctica aquí se re-estimarían los coeficientes Phi_boot
        # Para este ejemplo, usamos los mismos Phi pero con omega bootstrap
        omega_boot = np.cov(residuals_boot.T)
        Psi_boot = compute_psi_sequence(Phi, p, n_terms)
        IRF_boot.append([compute_IRF_VAR(Psi_boot, omega_boot, s) for s in range(n_terms)])
    
    # Calcular intervalos de confianza
    IRF_boot = np.array(IRF_boot)
    lower_percentile = 100 * alpha/2
    upper_percentile = 100 * (1 - alpha/2)
    
    IRF_lower = np.percentile(IRF_boot, lower_percentile, axis=0)
    IRF_upper = np.percentile(IRF_boot, upper_percentile, axis=0)
    
    return IRF_lower, IRF_upper
