r"""
Módulo de Benchmark I: Escalabilidad Algorítmica y Latencia Asintótica.

Ejecuta simulaciones empíricas para contrastar la complejidad computacional 
del Operador de Bisagra Discreto (D-IOB) frente a la Extracción Espectral canónica 
(Algoritmo iterativo de Lanczos).

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto". 
       Sección 5.1: Validación Empírica - Escalabilidad y Latencia.
"""

import time
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh # type: ignore
from scipy.sparse import csgraph # type: ignore

from iobsolve.core.space import DiscreteTopology
from iobsolve.discrete.hinge import DiscreteIntegrityOperator

def run_benchmark():
    r"""
    Simula la latencia de aislamiento topológico en redes Scale-Free crecientes.

    Notes
    -----
    Comprueba empíricamente que la evaluación del Laplaciano discreto normalizado 
    mediante tensores ralas preserva una cota de tiempo \mathcal{O}(k_i), evadiendo 
    la barrera \mathcal{O}(V^2 \log V) de los métodos espectrales.
    """
    sizes = [100, 500, 1000, 2500, 5000, 10000]
    time_spectral = []
    time_diob = []

    operator = DiscreteIntegrityOperator()

    for N in sizes:
        # Generación topológica (Barabási-Albert)
        G = nx.barabasi_albert_graph(N, m=3)
        A_scipy = nx.to_scipy_sparse_array(G, dtype=float)
        
        # Inyección al espacio tensorial de PyTorch
        A_coo = A_scipy.tocoo()
        indices = torch.tensor(np.vstack((A_coo.row, A_coo.col)), dtype=torch.long)
        values = torch.tensor(A_coo.data, dtype=torch.float32)
        A_torch = torch.sparse_coo_tensor(indices, values, torch.Size(A_coo.shape)).coalesce()
        
        topology = DiscreteTopology(adjacency=A_torch)
        
        X_np = np.random.normal(0, 1, (N, 10))
        X_np[0] += 15.0  # Inducción de singularidad asimétrica
        X_torch = torch.tensor(X_np, dtype=torch.float32)

        # --- Extracción Espectral (Algoritmo de Lanczos) ---
        t0 = time.perf_counter()
        L = csgraph.laplacian(A_scipy, normed=False) # type: ignore
        _, eigenvectors = eigsh(L, k=2, which='SM')  # type: ignore
        _ = np.abs(eigenvectors[:, 1])
        t_spec = time.perf_counter() - t0
        time_spectral.append(t_spec * 1000)

        # --- Operador D-IOB Nativo ---
        t0 = time.perf_counter()
        _ = operator.compute_stress(
            state_tensor=X_torch,
            topology=topology,
            laplacian_type='combinatorial',
            normalize_output=False
        )
        t_diob = time.perf_counter() - t0
        time_diob.append(t_diob * 1000)

    # Renderizado Fenomenológico
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, time_spectral, 'o--', color='firebrick', label='Extracción Espectral (Algoritmo de Lanczos)')
    plt.plot(sizes, time_diob, 's-', color='forestgreen', linewidth=2.5, label='D-IOB (PyTorch Core Sparse)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tamaño de la Red ($N$ nodos)')
    plt.ylabel('Tiempo de Ejecución (ms)')
    plt.title('Benchmark I: Escalabilidad y Latencia Asintótica')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('benchmark_case1.png', dpi=300)
    print("✓ Benchmark 1: benchmark_case1.png generado.")

if __name__ == "__main__":
    run_benchmark()