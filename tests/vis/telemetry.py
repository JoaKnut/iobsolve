r"""
Módulo de Visualización Telemétrica (IOB-Vis).

Genera gráficas científicas fenomenológicas (Diagramas de Hovmöller, Series Temporales
de Invariantes, y Funciones de Densidad de Probabilidad) para ilustrar los 
casos de estudio presentados en los manuscritos del framework.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra". (IOB Continuo).
.. [2] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto". (D-IOB).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración estética para papers científicos
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_continuous_lorenz(steps: int = 200, num_variables: int = 40, dt: float = 0.01):
    r"""
    Caso de Estudio I (Paper I): Atractor de Lorenz-96.

    Genera un Diagrama de Hovmöller doble comparando el estado del sistema (\phi(x,t))
    y el campo de estrés topológico (\mathcal{H}(x,t)) detectado por el IOB Continuo.
    Visualiza la propagación espacio-temporal de perturbaciones en regímenes caóticos.
    """
    from iobsolve.core.space import EuclideanManifold
    from iobsolve.continuous.hinge import ContinuousIntegrityOperator
    from iobsolve.plugins.continuous.dynamics import Lorenz96System

    manifold = EuclideanManifold(shape=(num_variables,), grid_spacing=1.0)
    lorenz = Lorenz96System(forcing_constant=8.0)
    operator = ContinuousIntegrityOperator()

    state = torch.ones(num_variables, dtype=torch.float64) * 8.0
    state[19] += 0.01  # Perturbación inicial

    history_state = []
    history_stress = []

    for t in range(steps):
        vector_field = lorenz(t * dt, state)
        stress = operator.compute_stress(
            state_tensor=vector_field, 
            manifold=manifold, 
            normalize=True
        )
        history_state.append(state.clone().cpu().numpy())
        history_stress.append(stress.cpu().numpy())
        state = state + vector_field * dt

    history_state = np.array(history_state)
    history_stress = np.array(history_stress)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Hovmöller del Estado Físico
    sns.heatmap(history_state, ax=axes[0], cmap="viridis", cbar=True)
    axes[0].set_title("Diagrama de Hovmöller (Estado Dinámico)")
    axes[0].set_xlabel("Índice Espacial (Variedad Euclidian)")
    axes[0].set_ylabel("Tiempo (Iteraciones)")

    # Hovmöller del Estrés Topológico
    sns.heatmap(history_stress, ax=axes[1], cmap="magma", cbar=True)
    axes[1].set_title("Estrés Topológico Normalizado (IOB)")
    axes[1].set_xlabel("Índice Espacial")
    axes[1].set_ylabel("Tiempo (Iteraciones)")

    plt.tight_layout()
    plt.savefig("tests/vis/lorenz_telemetry.png", dpi=300)
    print("Gráfica guardada: lorenz_telemetry.png")
    plt.show()


def plot_discrete_ddos(num_nodes: int = 500, total_steps: int = 120, attack_step: int = 80):
    r"""
    Caso de Estudio II (Paper II): Escudo DDoS.

    Muestra la evolución del Z-Score topológico máximo (\mathcal{M}_{\max}) y 
    la cardinalidad de conexiones activas (|E|). Ilustra fenomenológicamente 
    cómo la cirugía algorítmica mitiga el ataque en tiempo real (\mathcal{O}(k_i))
    restaurando la topología de la malla \mathcal{G}.
    """
    from iobsolve.core.space import DiscreteTopology
    from iobsolve.plugins.discrete.network_shield import DDoSShield

    # Crear topología en estrella (PyTorch Nativo)
    rows = torch.zeros(num_nodes - 1, dtype=torch.int64)
    cols = torch.arange(1, num_nodes, dtype=torch.int64)
    indices = torch.cat([torch.stack([rows, cols]), torch.stack([cols, rows])], dim=1)
    values = torch.ones(indices.shape[1], dtype=torch.float64)
    A = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
    topology = DiscreteTopology(adjacency=A)

    shield = DDoSShield(topology=topology, critical_threshold=5.0)
    base_state = torch.ones(num_nodes, dtype=torch.float64)

    max_z_scores = []
    cardinality = []
    attack_node = 250

    for t in range(total_steps):
        traffic = base_state + torch.normal(mean=0.0, std=0.05, size=(num_nodes,), dtype=torch.float64)
        
        # Inyección de ataque DDoS durante 3 iteraciones
        if t >= attack_step and t < attack_step + 3:
            traffic[attack_node] = 100.0 

        current_topology, z_scores = shield.process_telemetry(traffic)
        
        max_z_scores.append(torch.max(z_scores).item())
        # Calculamos la cantidad de aristas activas (medida de la cirugía)
        active_edges = current_topology.adjacency.coalesce().indices().shape[1]
        cardinality.append(active_edges)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'crimson'
    ax1.set_xlabel('Tiempo (Iteraciones)')
    ax1.set_ylabel(r'Z-Score Topológico Máximo ($\mathcal{M}_{\max}$)', color=color)
    ax1.plot(range(total_steps), max_z_scores, color=color, linewidth=2, label="Máx Z-Score")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=5.0, color='black', linestyle='--', alpha=0.7, label=r"Umbral $\tau_c$")
    ax1.axvline(x=attack_step, color='gray', linestyle=':', label="Inyección DDoS")

    # Eje gemelo para la cantidad de aristas
    ax2 = ax1.twinx()  
    color = 'steelblue'
    ax2.set_ylabel(r'Conexiones Activas ($|E|$)', color=color)  
    ax2.plot(range(total_steps), cardinality, color=color, linestyle='-.', alpha=0.8, label="Aristas |E|")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Mitigación Asíncrona (Cirugía de Welford)')
    fig.tight_layout()  
    plt.savefig("tests/vis/ddos_shield_telemetry.png", dpi=300)
    print("Gráfica guardada: ddos_shield_telemetry.png")
    plt.show()


def plot_discrete_mode_collapse():
    r"""
    Caso de Estudio III (Paper II): Colapso Modal.

    Compara la Función de Densidad de Probabilidad (PDF) del estrés topológico 
    \mathcal{H}_i entre un espacio latente métricamente saludable y una variedad 
    degenerada (concentración hiperespacial asimétrica).
    """
    from iobsolve.core.space import DiscreteTopology
    from iobsolve.discrete.hinge import DiscreteIntegrityOperator

    num_neurons = 256
    A = torch.ones((num_neurons, num_neurons), dtype=torch.float64) - torch.eye(num_neurons, dtype=torch.float64)
    topology = DiscreteTopology(adjacency=A)
    operator = DiscreteIntegrityOperator()

    # Red Saludable (Gaussiana)
    healthy_activations = torch.normal(mean=0.5, std=0.1, size=(num_neurons,), dtype=torch.float64)
    stress_healthy = operator.compute_stress(
        state_tensor=healthy_activations,
        topology=topology,
        laplacian_type='combinatorial',
        normalize_output=True
    ).cpu().numpy()

    # Red Colapsada (Concentración asimétrica de energía)
    collapsed_activations = torch.zeros(num_neurons, dtype=torch.float64)
    collapsed_activations[:30] = 10.0 
    stress_collapsed = operator.compute_stress(
        state_tensor=collapsed_activations,
        topology=topology,
        laplacian_type='combinatorial',
        normalize_output=True
    ).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    sns.histplot(stress_healthy, ax=axes[0], bins=30, color="mediumseagreen", kde=True, stat="density")
    axes[0].set_title("Distribución de Estrés: Red Saludable")
    axes[0].set_xlabel(r"Estrés Topológico $\mathcal{H}_i$")
    axes[0].set_ylabel("Densidad Probabilística")
    axes[0].axvline(x=0.8, color='red', linestyle='--', label="Umbral Colapso (0.8)")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].legend()

    sns.histplot(stress_collapsed, ax=axes[1], bins=30, color="darkorange", kde=True, stat="density")
    axes[1].set_title("Distribución de Estrés: Colapso Modal")
    axes[1].set_xlabel(r"Estrés Topológico $\mathcal{H}_i$")
    axes[1].axvline(x=0.8, color='red', linestyle='--', label="Umbral Colapso (0.8)")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("tests/vis/mode_collapse_telemetry.png", dpi=300)
    print("Gráfica guardada: mode_collapse_telemetry.png")
    plt.show()

if __name__ == "__main__":
    print("Generando Gráfica I: Lorenz-96 (Paper I)...")
    plot_continuous_lorenz()
    
    print("\nGenerando Gráfica II: Escudo DDoS (Paper II)...")
    plot_discrete_ddos()
    
    print("\nGenerando Gráfica III: Colapso Modal (Paper II)...")
    plot_discrete_mode_collapse()