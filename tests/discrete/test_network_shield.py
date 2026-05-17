import torch
from iobsolve.core.space import DiscreteTopology
from iobsolve.plugins.discrete.network_shield import DDoSShield

def test_ddos_shield_normal_traffic():
    """Prueba el régimen de homogeneidad (Blindaje del MAD)."""
    # Grafo denso de 10 nodos
    adj = torch.ones((10, 10))
    topo = DiscreteTopology(adj)
    shield = DDoSShield(topo, critical_threshold=3.0)
    
    # Tráfico perfectamente idéntico (Esto haría que el MAD sea 0)
    payload = torch.ones(10) * 150.0 
    
    _, alerts = shield.process_telemetry(payload)
    assert not alerts.any(), "Falso positivo: El blindaje del MAD falló y disparó alertas espurias."

def test_ddos_shield_attack():
    """Prueba la detección de una anomalía volumétrica."""
    adj = torch.ones((10, 10))
    topo = DiscreteTopology(adj)
    shield = DDoSShield(topo, critical_threshold=3.0)
    
    # Tráfico basal
    payload = torch.ones(10) * 100.0
    # Inyección volumétrica hostil en el nodo 3
    payload[3] = 5000.0 
    
    _, alerts = shield.process_telemetry(payload)
    assert alerts[3] == True, "Falso negativo: El escudo no detectó la anomalía volumétrica."

def test_watchdog_disconnected_graph():
    """Prueba la Ambigüedad del Vacío Topológico."""
    # Grafo completamente vacío (0 conexiones)
    adj = torch.zeros((10, 10)) 
    topo = DiscreteTopology(adj)
    shield = DDoSShield(topo)
    
    # Incluso con un ataque obvio...
    payload = torch.ones(10)
    payload[5] = 9999.0 
    
    # El watchdog debería interceptar la ejecución
    _, alerts = shield.process_telemetry(payload)
    
    # No debe haber alertas porque el análisis topológico se aborta
    assert not alerts.any(), "Peligro: El motor operó a pesar de que la red estaba desconectada."