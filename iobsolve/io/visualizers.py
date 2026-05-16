"""
Motor de Visualización Topológica para IOB-Solve.

Este módulo instrumenta las proyecciones gráficas utilizando Lazy Imports
para respetar el tipado estricto y no penalizar el tiempo de carga del CLI.
"""

import warnings
import importlib
import importlib.util
from typing import List, Any

import torch

def _check_vis_dependencies() -> None:
    """Verifica la disponibilidad del entorno gráfico dinámicamente."""
    required_packages = ["matplotlib", "networkx", "seaborn", "scipy"]
    missing = [pkg for pkg in required_packages if importlib.util.find_spec(pkg) is None]
    
    if missing:
        raise ImportError(
            f"Faltan dependencias de renderizado: {missing}. "
            "Instale el entorno gráfico ejecutando: pip install -e '.[vis]'"
        )


def plot_roots_spectrum(roots: List[Any], radius: float, filepath: str) -> None:
    """
    Renderiza el espectro de singularidades (atractores o subdominios) 
    en el plano complejo calculando sus centroides dinámicamente.
    """
    _check_vis_dependencies()
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 8))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    real_parts: List[float] = []
    imag_parts: List[float] = []
    
    for r in roots:
        if isinstance(r, complex):
            real_parts.append(r.real)
            imag_parts.append(r.imag)
        elif isinstance(r, (tuple, list)) and len(r) == 2:
            # [!] Inferencia del centroide para subdominios QuadTree
            try:
                x_domain, y_domain = r
                x_center = (x_domain[0] + x_domain[1]) / 2.0
                y_center = (y_domain[0] + y_domain[1]) / 2.0
                real_parts.append(x_center)
                imag_parts.append(y_center)
            except (IndexError, TypeError):
                continue

    if real_parts:
        plt.scatter(real_parts, imag_parts, c='#e74c3c', marker='x', s=40, alpha=0.5, label='Zonas Singulares')
    
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    
    plt.xlim(-radius, radius)
    plt.ylim(-radius, radius)
    plt.title('IOB-QuadTree: Espectro Topológico del Espacio', fontsize=14, pad=15)
    plt.xlabel('Eje X / Re(z)', fontsize=12)
    plt.ylabel('Eje Y / Im(z)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_shield_surgery(adjacency_matrix: torch.Tensor, alerts: torch.Tensor, filepath: str) -> None:
    """
    Renderiza el 1-esqueleto de la red. Resalta topológicamente los nodos extirpados
    por el Operador D-IOB.
    """
    _check_vis_dependencies()
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    
    # Proyección a matriz densa en memoria CPU
    if adjacency_matrix.is_sparse:
        adj_dense = adjacency_matrix.to_dense().cpu().numpy()
    else:
        adj_dense = adjacency_matrix.cpu().numpy()
        
    G = nx.from_numpy_array(adj_dense)
    num_nodes = len(G.nodes)
    
    anomalous_indices = set(torch.where(alerts)[0].cpu().tolist())
    node_colors = ['#e74c3c' if i in anomalous_indices else '#3498db' for i in range(num_nodes)]
    node_sizes = [150 if i in anomalous_indices else 35 for i in range(num_nodes)]
    
    plt.figure(figsize=(12, 12))
    plt.title('Cirugía Topológica D-IOB (Anti-DDoS)', fontsize=16, pad=20)
    
    # Para grafos densos, spring_layout es asintóticamente costoso. 
    # Optimizamos el renderizado basándonos en la cardinalidad de los nodos.
    if num_nodes <= 500:
        pos = nx.spring_layout(G, k=0.15, iterations=20)
    else:
        pos = nx.random_layout(G)
        warnings.warn("Topología altamente densa: Utilizando layout estocástico para visualización.")
        
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.15, edge_color='#7f8c8d')
    
    red_patch = mpatches.Patch(color='#e74c3c', label='Extirpado (Ataque Volumétrico)')
    blue_patch = mpatches.Patch(color='#3498db', label='Baricentro Estable (Tráfico Nominal)')
    plt.legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=11)
    
    plt.axis('off')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_audit_variance(embeddings: torch.Tensor, is_collapsing: bool, filepath: str) -> None:
    """
    Estima y renderiza la Distribución de Densidad de las distancias euclidianas
    en el espacio latente para evidenciar el colapso modal.
    """
    _check_vis_dependencies()
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.spatial.distance import pdist
    
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    X = embeddings.cpu().numpy()
    
    # Distancia iter-partícula (métrica de cohesión)
    dists = pdist(X, metric='euclidean')
    
    color_map = '#9b59b6' if is_collapsing else '#2ecc71'
    sns.histplot(dists, kde=True, color=color_map, stat='density', fill=True, alpha=0.5)
    
    estado_topologico = "DEGENERACIÓN ISOMÉTRICA" if is_collapsing else "ISOMETRÍA PRESERVADA"
    plt.title(f'Auditoría Latente: Varianza Topológica [{estado_topologico}]', fontsize=14, pad=15)
    plt.xlabel(r'Distancia Euclidiana Inter-nodo ($\mathcal{L}_2$)', fontsize=12)
    plt.ylabel('Densidad Estocástica', fontsize=12)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()