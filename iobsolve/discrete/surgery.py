r"""
Módulo de Cirugía Topológica.

Implementa el motor de cicatrización de redes complejas (Paper II).
Ejecuta podas asimétricas y aislamientos lógicos (\textit{Topological Pruning}) de manera 
vectorizada. Manipula el grafo computacional de PyTorch directamente sobre sus 
índices dispersos (COO) para confinar el costo asintótico al subgrafo afectado.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto". 
       Sección 3.2: Cirugía Topológica y el Costo de Cicatrización.
"""

from typing import Any
import torch
from iobsolve.core.space import DiscreteTopology

_sparse_coo: Any = getattr(torch, "sparse_coo_tensor")


class TopologicalSurgeon:
    r"""
    Motor de Cirugía Algorítmica Local.

    Orquesta las modificaciones estructurales de la matriz de adyacencia dinámica \mathcal{W}(t)
    en respuesta a colapsos detectados por el D-IOB.
    """

    def __init__(self, topology: DiscreteTopology):
        r"""
        Parameters
        ----------
        topology : DiscreteTopology
            La variedad discreta \mathcal{G}(V, E) base que será sometida a intervención.
        """
        self.topology = DiscreteTopology(adjacency=topology.adjacency.clone())

    def isolate_vertices(self, singular_indices: torch.Tensor) -> DiscreteTopology:
        r"""
        Ejecuta la extirpación topológica de los vértices anómalos.

        Fuerza analíticamente w_{ij}(t^+) = 0 para todo j \in \mathcal{N}(i), 
        desconectando completamente a la singularidad del resto de la malla.

        Parameters
        ----------
        singular_indices : torch.Tensor
            Vector de índices de los vértices (nodos) catalogados como anómalos.

        Returns
        -------
        DiscreteTopology
            La nueva variedad topológica con la cicatrización efectuada.
            
        Complexity
        ----------
        Asintóticamente \mathcal{O}(k_i) en matrices dispersas (Sparse COO), ya que 
        aplica máscaras booleanas exclusivamente sobre las coordenadas activas, 
        evitando una iteración densa sobre V^2.
        """
        if singular_indices.numel() == 0:
            return self.topology

        adjacency = self.topology.adjacency
        indices_to_isolate = singular_indices.flatten()

        if not adjacency.is_sparse:
            new_adjacency = adjacency.clone()
            new_adjacency[indices_to_isolate, :] = 0
            new_adjacency[:, indices_to_isolate] = 0
            
            return DiscreteTopology(adjacency=new_adjacency)
        else:
            adj_coo = adjacency.to_sparse_coo() if adjacency.layout != torch.sparse_coo else adjacency
            edge_coords = adj_coo.indices()
            edge_weights = adj_coo.values()
            
            mask_source = ~torch.isin(edge_coords[0], indices_to_isolate)
            mask_target = ~torch.isin(edge_coords[1], indices_to_isolate)
            valid_edges_mask = mask_source & mask_target
            
            new_indices = edge_coords[:, valid_edges_mask]
            new_values = edge_weights[valid_edges_mask]
            
            # Constructor blindado por el cortafuegos 'Any'
            new_csr_adj = _sparse_coo(
                new_indices, 
                new_values, 
                size=adj_coo.size(),
                device=adj_coo.device
            ).coalesce()
            
            return DiscreteTopology(adjacency=new_csr_adj)

    def prune_asymmetric_edges(self, source_index: int, target_indices: torch.Tensor) -> DiscreteTopology:
        r"""
        Poda aristas direccionales específicas sin aislar el nodo pivote por completo.

        Útil para mitigar vectores de estrés asimétricos (e.g., ataques DDoS selectivos) 
        manteniendo intacto el servicio legítimo periférico.

        Parameters
        ----------
        source_index : int
            Índice del nodo central o pivote (\text{server}).
        target_indices : torch.Tensor
            Índices de los nodos periféricos (\text{clients}) cuya conexión será podada.

        Returns
        -------
        DiscreteTopology
            La variedad resultante de la poda selectiva.
        """
        adjacency = self.topology.adjacency

        if not adjacency.is_sparse:
            new_adjacency = adjacency.clone()
            new_adjacency[source_index, target_indices] = 0
            new_adjacency[target_indices, source_index] = 0
                
            return DiscreteTopology(adjacency=new_adjacency)
        else:
            adj_coo = adjacency.to_sparse_coo() if adjacency.layout != torch.sparse_coo else adjacency
            edge_coords = adj_coo.indices()
            edge_weights = adj_coo.values()
            
            is_forward_cut = (edge_coords[0] == source_index) & torch.isin(edge_coords[1], target_indices)
            is_backward_cut = (edge_coords[1] == source_index) & torch.isin(edge_coords[0], target_indices)
            
            valid_edges_mask = ~(is_forward_cut | is_backward_cut)
            
            new_indices = edge_coords[:, valid_edges_mask]
            new_values = edge_weights[valid_edges_mask]
            
            new_csr_adj = _sparse_coo(
                new_indices, 
                new_values, 
                size=adj_coo.size(),
                device=adj_coo.device
            ).coalesce()
            
            return DiscreteTopology(adjacency=new_csr_adj)