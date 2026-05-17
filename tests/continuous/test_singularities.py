import math
import torch
from iobsolve.continuous.flow_theorem import FlowTheoremLocator

EPSILON = 1e-2

# =====================================================================
# FUNCIÓN DE CLUSTERING (Fusión de Fronteras)
# =====================================================================
def cluster_centroids(centroids, tolerance=EPSILON):
    """
    Agrupa centroides que están extremadamente cerca (solapamiento de frontera)
    y devuelve el centroide promedio de cada grupo.
    """
    if not centroids:
        return []
        
    clusters = []
    for punto in centroids:
        encontrado = False
        for cluster in clusters:
            # Distancia euclidiana entre el punto y el centroide del cluster
            dist = math.sqrt(sum((a - b)**2 for a, b in zip(punto, cluster[0])))
            if dist < tolerance:
                cluster.append(punto)
                encontrado = True
                break
        if not encontrado:
            clusters.append([punto])
            
    # Promediar cada cluster para obtener el punto focal exacto
    centroides_fusionados = []
    for cluster in clusters:
        avg_punto = tuple(sum(coords)/len(coords) for coords in zip(*cluster))
        centroides_fusionados.append(avg_punto)
        
    return centroides_fusionados

# =====================================================================
# VARIEDADES DE PRUEBA 
# =====================================================================
class SistemaParabola1D:
    def __call__(self, t, x): return x**2 - 4.0

class SistemaSinRaices1D:
    def __call__(self, t, x): return x**2 + 1.0

class SumideroLineal2D:
    def __call__(self, t, x):
        u = -x[..., 0]
        v = -x[..., 1]
        return torch.stack([u, v], dim=-1)

class SistemaTrascendental2D:
    def __call__(self, t, x):
        u = torch.sin(x[..., 0])
        v = torch.cos(x[..., 1])
        return torch.stack([u, v], dim=-1)

# =====================================================================
# PRUEBAS DE ASSERTION
# =====================================================================
def test_precision_parabola_1d():
    locator = FlowTheoremLocator(system_equation=SistemaParabola1D(), require_sign_change=True)
    raices = locator.locate_root_centroids(((-5.0, 5.0),), max_depth=10)
    raices = cluster_centroids(raices) # <-- Aplicamos clustering
    
    assert len(raices) == 2
    raices_x = sorted([r[0] for r in raices])
    assert math.isclose(raices_x[0], -2.0, abs_tol=EPSILON)
    assert math.isclose(raices_x[1], 2.0, abs_tol=EPSILON)

def test_ausencia_raices_1d():
    locator = FlowTheoremLocator(system_equation=SistemaSinRaices1D(), require_sign_change=True)
    raices = locator.locate_root_centroids(((-5.0, 5.0),), max_depth=8)
    raices = cluster_centroids(raices)
    assert len(raices) == 0

def test_precision_sumidero_lineal_2d():
    locator = FlowTheoremLocator(system_equation=SumideroLineal2D(), require_sign_change=True)
    raices = locator.locate_root_centroids(((-5.0, 5.0), (-5.0, 5.0)), max_depth=12)
    raices = cluster_centroids(raices) # <-- Fusión de los 4 cuadrantes
    
    assert len(raices) == 1, f"Debería encontrar 1 atractor, encontró {len(raices)}"
    rx, ry = raices[0]
    assert math.isclose(rx, 0.0, abs_tol=EPSILON)
    assert math.isclose(ry, 0.0, abs_tol=EPSILON)

def test_precision_trascendental_2d():
    locator = FlowTheoremLocator(system_equation=SistemaTrascendental2D(), require_sign_change=True)
    raices = locator.locate_root_centroids(((-2.0, 2.0), (0.0, 3.0)), max_depth=14)
    raices = cluster_centroids(raices) # <-- Fusión de los 2 hemisferios
    
    assert len(raices) == 1
    rx, ry = raices[0]
    assert math.isclose(rx, 0.0, abs_tol=EPSILON)
    assert math.isclose(ry, math.pi / 2, abs_tol=EPSILON)