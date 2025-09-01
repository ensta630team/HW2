import numpy as np
from scipy.linalg import lu

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

# Cálculo del IRF mediante triangularización
def compute_IRF_VAR(Psi, omega, s):
    """"
    Cálculo de IRF dado el vector de matrices Psi asociadas al VAR

    Argumentos:
        Phi     : Vector formado por las matrices Φ_i
        omega   : Matriz de Covarianzas
        s       : Horizonte que se quiere

    Retornará:
        Psi_s*a_j (a_j columna j de la matriz triangular A)    
    """

    # Descomponer triangularmente la matriz omega
    P, L, U = lu(omega) # L es la matriz triangular inferior

    # Cálculo del IRF
    IRF_s = np.dot(Psi[s], L)

    return IRF_s

    # Ejemplo de uso
if __name__ == "__main__":
    
    p = 2
    n_terms = 12
    
    # Definir matrices Φ₁ y Φ₂
    Phi1 = np.array([[0.5, 0.2], [0.3, 0.4]])
    Phi2 = np.array([[0.1, 0.0], [0.0, 0.1]])
    Phi = [Phi1, Phi2]
    
    # Calcular la sucesión Ψ
    Psi_sequence = compute_psi_sequence(Phi, p, n_terms)

    omega = np.array([[2, 1], [1, 3]])
    s = 2

    # Calcular IRF
    IRF_compute = compute_IRF_VAR(Psi_sequence,omega,s)
    print("Sucesión Ψ_i:")
    for i, psi in enumerate(Psi_sequence):
        print(f"Ψ_{i} =")
        print(psi)
        print()
    print(IRF_compute)
    