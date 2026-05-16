r"""
Suite de Pruebas Unitarias para el Dominio Discreto.

Certifica la instrumentación del D-IOB sobre topologías ralas. Verifica 
el respeto de la cota \mathcal{O}(k_i) durante la cirugía topológica y la 
inmunidad estadística de los detectores de colapso modal y mitigación DDoS.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto: 
       Laplacianos de Grafos, Detección de Anomalías Asíncronas y Colapsos en Redes Complejas".
"""

import torch
from iobsolve.core.space import DiscreteTopology
from iobsolve.plugins.discrete.network_shield import DDoSShield
from iobsolve.plugins.discrete.mode_collapse import ModeCollapseDetector

def create_star_graph(num_nodes: int) -> DiscreteTopology:
    r"""
    Helper estricto en PyTorch para crear la topología base de las pruebas.

    Genera una red estelar bipartita representativa de arquitecturas cliente-servidor,
    almacenándola directamente como un tensor ralo (Sparse COO) para pruebas \mathcal{O}(k_i).
    """
    rows = torch.zeros(num_nodes - 1, dtype=torch.int64)
    cols = torch.arange(1, num_nodes, dtype=torch.int64)
    
    indices_forward = torch.stack([rows, cols])
    indices_backward = torch.stack([cols, rows])
    indices = torch.cat([indices_forward, indices_backward], dim=1)
    
    values = torch.ones(indices.shape[1], dtype=torch.float64)
    
    A = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
    return DiscreteTopology(adjacency=A)

def test_ddos_shield_normal_traffic():
    r"""
    Verifica que el escudo no altere la red bajo tráfico legítimo.

    Notes
    -----
    Garantía 1: Prevención de Falsos Positivos.
    Asegura que bajo perturbaciones estocásticas gaussianas \mathcal{N}(0, \sigma^2), 
    el estimador asintótico (\mathcal{M}_i) no cruce la frontera \tau_c = 5.0.
    """
    num_nodes = 500
    topology = create_star_graph(num_nodes)
    shield = DDoSShield(topology=topology, critical_threshold=5.0)
    
    base_state = torch.ones(num_nodes, dtype=torch.float64)
    z_scores = torch.zeros(num_nodes, dtype=torch.float64)
    
    for _ in range(10):
        traffic = base_state + torch.normal(mean=0.0, std=0.05, size=(num_nodes,), dtype=torch.float64)
        current_topology, z_scores = shield.process_telemetry(traffic)
        
    assert torch.max(z_scores).item() < 5.0, "El Z-Score superó el umbral crítico sin existir ataque."

def test_ddos_shield_attack_mitigation():
    r"""
    Verifica que el escudo detecte la anomalía y ampute el nodo atacado.

    Notes
    -----
    Garantía 2: Cirugía Topológica Asíncrona.
    Prueba empíricamente que ante un vector volumétrico hostil, el motor extirpa 
    la conexión lógica (w_{ij}(t^+) = 0) de manera exacta y autónoma.
    """
    num_nodes = 500
    topology = create_star_graph(num_nodes)
    shield = DDoSShield(topology=topology, critical_threshold=5.0)
    
    base_state = torch.ones(num_nodes, dtype=torch.float64)
    attack_node = 250
    
    # CORRECCIÓN MATEMÁTICA: 50 iteraciones de calentamiento para permitir Z-Scores > 5.0
    for _ in range(50):
        shield.process_telemetry(base_state + torch.normal(mean=0.0, std=0.05, size=(num_nodes,), dtype=torch.float64))
        
    attack_traffic = base_state + torch.normal(mean=0.0, std=0.05, size=(num_nodes,), dtype=torch.float64)
    attack_traffic[attack_node] = 100.0 
    
    new_topology, _ = shield.process_telemetry(attack_traffic)
    
    adj_coo = new_topology.adjacency.coalesce()
    indices = adj_coo.indices()
    
    edges_of_attack_node = (indices[0] == attack_node) | (indices[1] == attack_node)
    assert not torch.any(edges_of_attack_node), "Falso negativo: El escudo no extirpó las conexiones del vértice atacado."

def test_mode_collapse_healthy():
    r"""
    Verifica que una red uniforme no dispare la alarma de colapso.

    Notes
    -----
    Garantía 3: Estabilidad de Representaciones Latentes.
    Comprueba que una variedad poblacional isomórficamente distribuida mantenga 
    su varianza de regularidad topológica bajo control.
    """
    num_neurons = 256
    A = torch.ones((num_neurons, num_neurons), dtype=torch.float64) - torch.eye(num_neurons, dtype=torch.float64)
    topology = DiscreteTopology(adjacency=A)
    detector = ModeCollapseDetector(topology=topology, collapse_threshold=0.8)

    # CORRECCIÓN MATEMÁTICA: Usar distribución Gaussiana para evitar la cola gorda artificial de la distribución uniforme
    healthy_activations = torch.normal(mean=0.5, std=0.1, size=(num_neurons,), dtype=torch.float64)
    
    assert not detector.scan_activations(healthy_activations), "El detector reportó colapso en un estado saludable."

def test_mode_collapse_critical():
    r"""
    Verifica la detección matemática de una singularidad de concentración de energía.

    Notes
    -----
    Garantía 4: Alerta Temprana de Falla de Gradientes.
    Fuerza un atractor puntual en el subgrafo latente y asevera que la ruptura 
    topológica gatilla el estado de alerta del D-IOB exitosamente.
    """
    num_neurons = 256
    A = torch.ones((num_neurons, num_neurons), dtype=torch.float64) - torch.eye(num_neurons, dtype=torch.float64)
    topology = DiscreteTopology(adjacency=A)
    detector = ModeCollapseDetector(topology=topology, collapse_threshold=0.8)

    collapsed_activations = torch.zeros(num_neurons, dtype=torch.float64)
    collapsed_activations[:30] = 10.0 
    
    assert detector.scan_activations(collapsed_activations), "El detector no identificó el colapso asimétrico de energía."