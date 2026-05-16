import torch
import pytest
from iobsolve.core.space import DiscreteTopology
from iobsolve.plugins.discrete.network_shield import DDoSShield
from iobsolve.plugins.discrete.mode_collapse import ModeCollapseDetector

# =====================================================================
# PRUEBAS DEL MOTOR DISCRETO: VERDAD BASE TOPOLÓGICA
# =====================================================================

def test_shield_red_homogenea():
    """
    Verdad Base: En un grafo anillo circular (todos los nodos tienen grado 2) 
    y con un flujo de tráfico idéntico, el estrés topológico debe ser NULO. 
    Ningún nodo debe ser extirpado.
    """
    N = 10
    # Creamos un grafo anillo
    adj = torch.zeros((N, N), dtype=torch.float64)
    for i in range(N):
        adj[i, (i+1)%N] = 1.0
        adj[i, (i-1)%N] = 1.0
        
    topology = DiscreteTopology(adjacency=adj.to_sparse_coo())
    shield = DDoSShield(topology=topology, critical_threshold=3.0)
    
    # Tráfico perfectamente homogéneo
    flujo_nominal = torch.ones(N, dtype=torch.float64)
    
    _, alertas = shield.process_telemetry(flujo_nominal)
    
    # ASERCIÓN: No debe haber NINGUNA alerta
    assert not alertas.any(), "Falso positivo: El motor detectó anomalías en una topología en perfecto equilibrio."


def test_shield_ataque_estrella():
    """
    Verdad Base: En una topología estrella, el nodo central (índice 0) 
    es un cuello de botella geométrico. Si le inyectamos un flujo masivo,
    el IOB debe marcar EXACTAMENTE y ÚNICAMENTE al nodo 0 para cirugía.
    """
    N = 50
    # Grafo Estrella: El nodo 0 está conectado a todos. Los demás solo al 0.
    adj = torch.zeros((N, N), dtype=torch.float64)
    adj[0, 1:] = 1.0
    adj[1:, 0] = 1.0
    
    topology = DiscreteTopology(adjacency=adj.to_sparse_coo())
    
    # Umbral sensible para detectar la asimetría
    shield = DDoSShield(topology=topology, critical_threshold=2.5)
    
    # Tráfico normal para los bordes (ruido gaussiano suave)
    flujo_ataque = torch.abs(torch.randn(N, dtype=torch.float64))
    
    # Inyección asimétrica masiva en el nodo central (ataque DDoS)
    flujo_ataque[0] = 10000.0 
    
    _, alertas = shield.process_telemetry(flujo_ataque)
    
    # ASERCIÓN: Debe haber exactamente 1 nodo comprometido
    assert alertas.sum().item() == 1, f"Debería extirpar 1 nodo, extirpó {alertas.sum().item()}"
    
    # ASERCIÓN: Ese único nodo DEBE ser el nodo 0
    nodo_extirpado = torch.where(alertas)[0].item()
    assert nodo_extirpado == 0, f"Error topológico: Extirpó el nodo equivocado (nodo {nodo_extirpado})"


def test_audit_isometria_perfecta():
    """
    Verdad Base: Si pasamos una matriz identidad a los embeddings (vectores 
    completamente ortogonales entre sí), la varianza es máxima.
    NO debe haber colapso modal.
    """
    N = 64
    # Vectores ortogonales puros (Base canónica)
    embeddings_ortogonales = torch.eye(N, dtype=torch.float64)
    
    # Grafo totalmente conectado (Batch adjacency)
    adj = torch.ones((N, N), dtype=torch.float64) - torch.eye(N, dtype=torch.float64)
    topology = DiscreteTopology(adjacency=adj)
    
    detector = ModeCollapseDetector(topology=topology, collapse_threshold=0.9)
    colapso = detector.scan_activations(embeddings_ortogonales)
    
    assert colapso is False, "Fallo métrico: Detectó colapso en un espacio perfectamente ortogonal e isométrico."


def test_audit_colapso_total():
    """
    Verdad Base: Si todos los embeddings convergen al mismo punto exacto en el espacio 
    latente (degeneración de la red neuronal), la varianza topológica colapsa.
    El motor DEBE detectarlo.
    """
    N, Dim = 64, 128
    # Todos los vectores son idénticos al vector base
    vector_atractor = torch.randn(Dim, dtype=torch.float64)
    embeddings_colapsados = vector_atractor.repeat(N, 1)
    
    adj = torch.ones((N, N), dtype=torch.float64) - torch.eye(N, dtype=torch.float64)
    topology = DiscreteTopology(adjacency=adj)
    
    # Umbral estricto
    detector = ModeCollapseDetector(topology=topology, collapse_threshold=0.8)
    colapso = detector.scan_activations(embeddings_colapsados)
    
    assert colapso is True, "Peligro: El IOB no detectó un atractor puntual (Colapso Modal Severo) en el hiperespacio."