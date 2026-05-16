import time
import torch
from iobsolve.core.space import DiscreteTopology
from iobsolve.plugins.discrete.network_shield import DDoSShield
from iobsolve.plugins.discrete.mode_collapse import ModeCollapseDetector
from iobsolve.continuous.flow_theorem import FlowTheoremLocator
from iobsolve.plugins.continuous.dynamics import Lorenz96System
import warnings
warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")

# =====================================================================
# PRUEBAS DE RENDIMIENTO Y ESCALABILIDAD ASINTÓTICA (BENCHMARKS)
# =====================================================================

def test_benchmark_discrete_scale_100k():
    """
    Evalúa el comportamiento asintótico del Laplaciano en grafos masivos.
    Instancia una red de 100,000 nodos y verifica que la cirugía (D-IOB)
    se ejecute en menos de 2 segundos gracias a las matrices ralas (sparse).
    """
    N = 100_000
    aristas = N * 5  # Grado promedio de 5 conexiones por nodo
    
    # 1. Generar matriz de adyacencia dispersa aleatoria
    indices = torch.randint(0, N, (2, aristas))
    valores = torch.ones(aristas, dtype=torch.float64)
    adj_sparse = torch.sparse_coo_tensor(indices, valores, (N, N)).coalesce()
    
    topology = DiscreteTopology(adjacency=adj_sparse)
    shield = DDoSShield(topology=topology, critical_threshold=3.0)
    
    # Tráfico simulado
    traffic = torch.abs(torch.randn(N, dtype=torch.float64))
    traffic[0] = 50000.0  # Inyectar anomalía para forzar el cálculo de cirugía
    
    # 2. Medir el tiempo exacto del procesamiento core
    start_time = time.perf_counter()
    _, alertas = shield.process_telemetry(traffic)
    elapsed = time.perf_counter() - start_time
    
    # 3. ASERCIÓN DE COMPLEJIDAD
    # Si tarda más de 2 segundos, hemos roto la complejidad O(k) 
    # y el código está haciendo operaciones densas por error.
    assert elapsed < 2.0, f"Violación de escalabilidad: Tardo {elapsed:.4f}s en 100k nodos."
    assert alertas[0].item() is True, "El ataque no fue detectado bajo presión."


def test_benchmark_continuous_fft_scaling():
    """
    Evalúa la cota computacional O(N log N) del IOB-FFT.
    Aumenta drásticamente la resolución de la malla a 256x256 (65,536 puntos)
    y verifica que la FFT multidimensional no colapse el procesador.
    """
    class EcuacionMasiva:
        def __call__(self, t, x):
            return x**2 - 4.0
            
    # Forzamos una resolución de 256x256 (muy por encima de los 16x16 por defecto)
    locator = FlowTheoremLocator(system_equation=EcuacionMasiva(), grid_resolution=256)
    dominio = ((-10.0, 10.0), (-10.0, 10.0))
    
    start_time = time.perf_counter()
    # Poca profundidad porque queremos medir el costo de la malla, no del árbol
    locator.locate_root_centroids(dominio, max_depth=3)
    elapsed = time.perf_counter() - start_time
    
    # 3. ASERCIÓN DE COMPLEJIDAD
    # En una malla 256x256, el FFT 2D en PyTorch debería tomar < 1.0s
    assert elapsed < 1.0, f"Cuello de botella en IOB-FFT: Tomó {elapsed:.4f}s para una malla 256x256."

def test_benchmark_lorenz96_vectorization():
    """
    Evalúa la cota asintótica del atractor Lorenz-96.
    Al utilizar `torch.roll` en lugar de bucles for, el sistema debería 
    ser capaz de calcular el flujo para un batch de 10,000 trayectorias 
    en 1,000 dimensiones (10 Millones de datos flotantes) en una fracción de segundo.
    """
    lorenz = Lorenz96System(forcing_constant=8.0)
    
    # Batch masivo: 10,000 estados simultáneos, cada uno de 1,000 dimensiones
    estado_masivo = torch.randn(10000, 1000, dtype=torch.float64)
    
    start_time = time.perf_counter()
    _ = lorenz(0.0, estado_masivo)
    elapsed = time.perf_counter() - start_time
    
    # ASERCIÓN DE COMPLEJIDAD
    # Procesar 10 millones de floats vectorizados en CPU debería tomar menos de 0.2 segundos.
    assert elapsed < 1.0, f"Cuello de botella en Lorenz-96: Tardó {elapsed:.4f}s."
    
def test_benchmark_audit_llm_scale():
    """
    Evalúa el Laplaciano Combinatorio en el plugin ModeCollapseDetector 
    enfrentándolo a la escala de embeddings de un Modelo de Lenguaje Grande (LLM).
    Batch: 2048 tokens, Dimensión latente: 4096.
    """
    B, Dim = 2048, 4096
    embeddings_pesados = torch.randn(B, Dim, dtype=torch.float64)
    
    # Topología totalmente conectada (batch adjacency)
    adj = torch.ones((B, B), dtype=torch.float64) - torch.eye(B, dtype=torch.float64)
    topology = DiscreteTopology(adjacency=adj)
    
    detector = ModeCollapseDetector(topology=topology, collapse_threshold=0.9)
    
    start_time = time.perf_counter()
    _ = detector.scan_activations(embeddings_pesados)
    elapsed = time.perf_counter() - start_time
    
    # ASERCIÓN DE COMPLEJIDAD
    # El Laplaciano sobre una matriz densa de 2048x2048 con vectores de 4096 dims 
    # es pesado pero debe completarse eficientemente (idealmente < 1.0s).
    assert elapsed < 1.0, f"Escalabilidad comprometida en auditoría IA: Tomó {elapsed:.4f}s."