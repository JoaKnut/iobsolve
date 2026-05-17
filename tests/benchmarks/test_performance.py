r"""
Suite de Benchmarks: Escalabilidad Asintótica del Framework IOB-Solve.

Estos tests validan que las cotas de complejidad algorítmica documentadas
se mantengan en la práctica bajo cargas representativas. Están marcados con
``pytest.mark.benchmark`` y pueden excluirse de CI rápido con:

    pytest -m "not benchmark"

Cotas evaluadas
---------------
- D-IOB sobre grafos de 100 k nodos:     O(k_i) mediante Sparse COO.
- IOB-FFT sobre malla 256×256:           O(N log N) de la FFT nD.
- Lorenz-96 vectorizado (10k × 1k):      O(N) gracias a torch.roll.
- Auditoría latente LLM (B=2048, D=4096): O(B × D) Laplaciano denso.

Notes
-----
Los umbrales de tiempo contemplan una CPU de un solo núcleo sin aceleración
GPU (entorno de referencia: sandbox con ~2 GHz). En hardware moderno los
tiempos serán sustancialmente menores.
"""

import time
import warnings
import pytest
import torch

from iobsolve.core.space import DiscreteTopology
from iobsolve.plugins.discrete.network_shield import DDoSShield
from iobsolve.plugins.discrete.mode_collapse import ModeCollapseDetector
from iobsolve.continuous.flow_theorem import FlowTheoremLocator
from iobsolve.plugins.continuous.dynamics import Lorenz96System

warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")

pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# Utilidad compartida
# ---------------------------------------------------------------------------

def _warmup_torch() -> None:
    """Fuerza la compilación JIT de PyTorch antes de medir."""
    _ = torch.ones(10) @ torch.ones(10)


# ---------------------------------------------------------------------------
# BENCHMARK 1 — D-IOB en grafo de 100 k nodos (Sparse COO)
# ---------------------------------------------------------------------------

def test_benchmark_discrete_scale_100k():
    """
    Evalúa la complejidad :math:`\\mathcal{O}(k_i)` del D-IOB en grafos masivos.

    Instancia una red aleatoria de 100 000 nodos con grado promedio 5 y verifica
    que la cirugía (cómputo del Z-Score + extirpación) se complete en menos de
    3 segundos, confirmando que el motor no degenera a complejidad densa
    :math:`\\mathcal{O}(N^2)`.

    Umbral
    ------
    < 3.0 s en CPU de referencia (2 GHz, un núcleo).
    """
    _warmup_torch()
    N = 100_000
    aristas = N * 5

    indices = torch.randint(0, N, (2, aristas))
    valores = torch.ones(aristas, dtype=torch.float64)
    adj_sparse = torch.sparse_coo_tensor(indices, valores, (N, N)).coalesce()

    topology = DiscreteTopology(adjacency=adj_sparse)
    shield = DDoSShield(topology=topology, critical_threshold=3.0)

    traffic = torch.abs(torch.randn(N, dtype=torch.float64))
    traffic[0] = 50_000.0  # anomalía para forzar la cirugía

    start = time.perf_counter()
    _, alertas = shield.process_telemetry(traffic)
    elapsed = time.perf_counter() - start

    assert elapsed < 3.0, (
        f"Violación de escalabilidad O(k_i): {elapsed:.4f}s en 100 k nodos "
        f"(umbral: 3.0s)."
    )
    assert alertas[0].item() is True, "La anomalía no fue detectada bajo carga."


# ---------------------------------------------------------------------------
# BENCHMARK 2 — IOB-FFT en malla 256×256
# ---------------------------------------------------------------------------

def test_benchmark_continuous_fft_scaling():
    """
    Evalúa la cota :math:`\\mathcal{O}(N \\log N)` del IOB-FFT.

    Aumenta la resolución de la malla a 256×256 (65 536 puntos) y verifica
    que la Transformada de Fourier multidimensional no colapse el procesador.
    La profundidad del árbol es baja (3) para aislar el coste de la FFT del
    coste de la bisección recursiva.

    Umbral
    ------
    < 2.0 s en CPU de referencia.
    """
    _warmup_torch()

    class EcuacionMasiva:
        def __call__(self, t, x):
            return x ** 2 - 4.0

    locator = FlowTheoremLocator(
        system_equation=EcuacionMasiva(),
        grid_resolution=256,
    )
    dominio = ((-10.0, 10.0), (-10.0, 10.0))

    start = time.perf_counter()
    locator.locate_root_centroids(dominio, max_depth=3)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, (
        f"Cuello de botella en IOB-FFT: {elapsed:.4f}s para malla 256×256 "
        f"(umbral: 2.0s)."
    )


# ---------------------------------------------------------------------------
# BENCHMARK 3 — Lorenz-96 vectorizado (10 k × 1 k)
# ---------------------------------------------------------------------------

def test_benchmark_lorenz96_vectorization():
    """
    Evalúa la cota :math:`\\mathcal{O}(N)` del atractor de Lorenz-96.

    La implementación usa ``torch.roll`` en lugar de bucles Python, por lo que
    debería procesar un batch de 10 000 trayectorias en 1 000 dimensiones
    (10 millones de flotantes) sin degradación cuadrática.

    Umbral
    ------
    < 3.0 s en CPU de referencia (incluye primera llamada con overhead JIT).
    """
    _warmup_torch()
    lorenz = Lorenz96System(forcing_constant=8.0)

    estado_masivo = torch.randn(10_000, 1_000, dtype=torch.float64)

    start = time.perf_counter()
    _ = lorenz(0.0, estado_masivo)
    elapsed = time.perf_counter() - start

    assert elapsed < 3.0, (
        f"Cuello de botella en Lorenz-96: {elapsed:.4f}s para batch 10k×1k "
        f"(umbral: 3.0s)."
    )


# ---------------------------------------------------------------------------
# BENCHMARK 4 — Auditoría latente escala LLM (B=2048, D=4096)
# ---------------------------------------------------------------------------

def test_benchmark_audit_llm_scale():
    """
    Evalúa el Laplaciano Combinatorio en escala de Modelos de Lenguaje Grande.

    Enfrenta al ``ModeCollapseDetector`` con un batch de 2 048 embeddings de
    4 096 dimensiones (representativo de un LLM o ViT grande). La multiplicación
    Laplaciano–embeddings es :math:`\\mathcal{O}(B^2 \\cdot D)` en la topología
    completamente conectada usada aquí.

    Umbral
    ------
    < 3.0 s en CPU de referencia.
    """
    _warmup_torch()
    B, Dim = 2_048, 4_096
    embeddings_pesados = torch.randn(B, Dim, dtype=torch.float64)

    adj = torch.ones((B, B), dtype=torch.float64) - torch.eye(B, dtype=torch.float64)
    topology = DiscreteTopology(adjacency=adj)
    detector = ModeCollapseDetector(topology=topology, collapse_threshold=0.9)

    start = time.perf_counter()
    _ = detector.scan_activations(embeddings_pesados)
    elapsed = time.perf_counter() - start

    assert elapsed < 3.0, (
        f"Escalabilidad comprometida en auditoría LLM: {elapsed:.4f}s "
        f"para B=2048 D=4096 (umbral: 3.0s)."
    )
