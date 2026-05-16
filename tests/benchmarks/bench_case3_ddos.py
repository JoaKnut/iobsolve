r"""
Módulo de Benchmark III: Inmunidad al Efecto de Enmascaramiento.

Contrasta el límite de ruptura (\textit{Breakdown Point}) probabilístico 
de las métricas de ciberseguridad estándar frente a la estimación asimétrica 
recursiva del D-IOB ante inyecciones volumétricas distribuidas (DDoS).

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto". 
       Sección 5.3: Tolerancia al Masking Effect.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from iobsolve.discrete.estimators import RecursiveTopologicalZScore

def run_benchmark():
    r"""
    Simula la inyección de tráfico anómalo masivo para forzar la dilución analítica.

    Notes
    -----
    Garantiza empíricamente que el uso de cuantiles estadísticos (Mediana y MAD) 
    inmuniza al motor contra el sesgo inducido por tensores de estado hostiles,
    preservando el F1-Score en regímenes de alta contaminación.
    """
    N = 1000
    contamination_rates = np.linspace(0.01, 0.45, 20)
    
    f1_standard = []
    f1_diob = []
    tau = 3.0
    
    for rate in contamination_rates:
        n_attackers = int(N * rate)
        n_legit = N - n_attackers
        
        Q_legit = np.random.exponential(scale=1.0, size=n_legit)
        Q_attackers = np.random.normal(loc=15.0, scale=2.0, size=n_attackers)
        Q_total = np.concatenate([Q_legit, Q_attackers])
        y_true = np.concatenate([np.zeros(n_legit), np.ones(n_attackers)])
        
        # --- Z-Score Poblacional (Sujeto a Masking Effect) ---
        mean_Q = np.mean(Q_total)
        std_Q = np.std(Q_total) + 1e-8
        z_std = (Q_total - mean_Q) / std_Q
        y_pred_std = (z_std > tau).astype(int)
        f1_standard.append(f1_score(y_true, y_pred_std))
        
        # --- Z-Score Robusto del D-IOB ---
        z_estimator = RecursiveTopologicalZScore(num_nodes=N)
        stress_tensor = torch.tensor(Q_total, dtype=torch.float32)
        
        z_rob_tensor = z_estimator.update_and_compute(stress_tensor)
        y_pred_rob = (z_rob_tensor.numpy() > tau).astype(int)
        f1_diob.append(f1_score(y_true, y_pred_rob))

    # Renderizado Fenomenológico
    plt.figure(figsize=(8, 6))
    plt.plot(contamination_rates * 100, f1_standard, 'o--', color='firebrick', label='Z-Score Poblacional (Media/Std)')
    plt.plot(contamination_rates * 100, f1_diob, 's-', color='forestgreen', linewidth=2.5, label=r'D-IOB: Z-Score Topológico ($\mathcal{M}_i$)')
    
    plt.xlabel('Ratio de Contaminación DDoS (%)')
    plt.ylabel('F1-Score de Detección')
    plt.title('Benchmark III: Resistencia al Efecto de Enmascaramiento')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig('benchmark_case3.png', dpi=300)
    print("✓ Benchmark 3: benchmark_case3.png generado.")

if __name__ == "__main__":
    run_benchmark()