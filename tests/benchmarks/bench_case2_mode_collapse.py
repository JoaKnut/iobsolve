r"""
Módulo de Benchmark II: Anticipación Predictiva de Colapso Modal.

Simula el entrenamiento de una arquitectura latente profunda (IA) para constatar 
la hipersensibilidad del invariante topológico (\sigma_{\mathcal{Q}}^2) frente a 
la métrica de varianza estándar (\sigma^2).

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto". 
       Sección 5.2: Caso de Estudio II - Colapso Modal.
"""

import torch
import numpy as np
import scipy.sparse as sparse # type: ignore
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph

from iobsolve.core.space import DiscreteTopology
from iobsolve.discrete.hinge import DiscreteIntegrityOperator

def run_benchmark():
    r"""
    Monitorea la degeneración isométrica iterativa del espacio latente.

    Notes
    -----
    El ensayo demuestra que \sigma_{\mathcal{Q}}^2 actúa como un sensor 
    predictivo de alerta temprana, superando a los indicadores retrasados 
    (\textit{lagging indicators}) habituales en Machine Learning.
    """
    epochs = 100
    N_batch = 256
    dim = 64
    
    global_variance = []
    diob_sigma_q = []
    
    operator = DiscreteIntegrityOperator()
    X = np.random.normal(0, 1, (N_batch, dim))
    
    for epoch in range(epochs):
        if epoch > 40:
            atractor = np.ones(dim) * 2.0
            X = X * (1 - 0.05) + atractor * 0.05  # Contracción hiperespacial
            
        X_noisy = X + np.random.normal(0, 0.1, X.shape)
        
        # --- Varianza Estándar (\sigma^2) ---
        global_variance.append(float(np.var(X_noisy)))
        
        # --- Varianza Topológica del D-IOB (\sigma_{\mathcal{Q}}^2) ---
        A_scipy = kneighbors_graph(X_noisy, 5, mode='connectivity', include_self=False)
        A_coo = sparse.coo_matrix(A_scipy)
        
        indices = torch.tensor(np.vstack((A_coo.row, A_coo.col)), dtype=torch.long)
        values = torch.tensor(A_coo.data, dtype=torch.float32)
        A_torch = torch.sparse_coo_tensor(indices, values, torch.Size(A_coo.shape)).coalesce() # type: ignore
        
        topology = DiscreteTopology(adjacency=A_torch)
        X_torch = torch.tensor(X_noisy, dtype=torch.float32)

        stress_tensor = operator.compute_stress(
            state_tensor=X_torch,
            topology=topology,
            laplacian_type='normalized',
            normalize_output=False
        )
        
        sigma_q = torch.var(stress_tensor).item()
        diob_sigma_q.append(sigma_q)

    # Renderizado Fenomenológico
    arr_global = np.array(global_variance) / np.max(global_variance)
    arr_diob = np.array(diob_sigma_q) / np.max(diob_sigma_q)

    plt.figure(figsize=(10, 5))
    plt.plot(arr_global, '--', color='gray', label=r'Varianza Estándar del Espacio Latente ($\sigma^2$)')
    plt.plot(arr_diob, '-', color='darkorange', linewidth=2.5, label=r'D-IOB: $\sigma_{\mathcal{Q}}^2$ (Varianza Topológica)')
    plt.axvline(x=40, color='red', linestyle=':', label='Inicio Colapso Modal')
    
    plt.xlabel('Iteraciones (Epochs)')
    plt.ylabel('Invariante Estructural (Normalizado)')
    plt.title('Benchmark II: Alerta Temprana de Colapso Dimensional')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('benchmark_case2.png', dpi=300)
    print("✓ Benchmark 2: benchmark_case2.png generado.")

if __name__ == "__main__":
    run_benchmark()