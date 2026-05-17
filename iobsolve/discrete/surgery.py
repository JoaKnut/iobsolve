r"""
Módulo de Cirugía Topológica.

Implementa el motor de cicatrización de redes complejas (Paper II).
Ejecuta podas asimétricas y aislamientos lógicos (*Topological Pruning*) de
manera vectorizada, manipulando el grafo computacional de PyTorch directamente
sobre sus índices dispersos (COO) para confinar el coste asintótico al subgrafo
afectado.

Las operaciones expuestas son **no destructivas**: siempre retornan una nueva
:class:`~iobsolve.core.space.DiscreteTopology` sin modificar la instancia
original. El motor utiliza clonado lazy en matrices densas y operaciones de
máscara booleana en matrices dispersas (COO), ambos en
:math:`\mathcal{O}(k_i)` para topologías ralas.

Operaciones disponibles
-----------------------
- :meth:`TopologicalSurgeon.isolate_vertices` — Extirpa un conjunto de vértices
  anulando todas sus aristas entrantes y salientes.
- :meth:`TopologicalSurgeon.prune_asymmetric_edges` — Poda aristas
  direccionales específicas desde un nodo pivote hacia un subconjunto de nodos
  periféricos, manteniendo el resto de la conectividad intacto.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto".
       Sección 3.2: Cirugía Topológica y el Costo de Cicatrización.
"""

from typing import Any

import torch

from iobsolve.core.space import DiscreteTopology

# Cortafuegos para el linter: constructor opaco de sparse COO
_sparse_coo: Any = getattr(torch, "sparse_coo_tensor")


class TopologicalSurgeon:
    r"""
    Motor de Cirugía Algorítmica Local.

    Orquesta las modificaciones estructurales de la matriz de adyacencia
    dinámica :math:`\mathcal{W}(t)` en respuesta a colapsos detectados por el
    D-IOB. Todas las operaciones son **no destructivas**: retornan siempre una
    nueva instancia de :class:`~iobsolve.core.space.DiscreteTopology`.

    Parameters
    ----------
    topology : DiscreteTopology
        La variedad discreta :math:`\mathcal{G}(V, E)` base que será sometida
        a intervención. Se realiza un ``clone()`` interno para evitar efectos
        laterales sobre la instancia original.

    Attributes
    ----------
    topology : DiscreteTopology
        Copia de trabajo interna de la topología. Inmutable desde fuera de la
        clase; cada operación de cirugía retorna una nueva instancia.

    Examples
    --------
    Aislar el nodo central de una topología estrella ante un ataque DDoS:

    >>> import torch
    >>> from iobsolve.core.space import DiscreteTopology
    >>> from iobsolve.discrete.surgery import TopologicalSurgeon
    >>> N = 10
    >>> adj = torch.zeros((N, N), dtype=torch.float64)
    >>> adj[0, 1:] = 1.0; adj[1:, 0] = 1.0
    >>> topo = DiscreteTopology(adjacency=adj)
    >>> surgeon = TopologicalSurgeon(topology=topo)
    >>> new_topo = surgeon.isolate_vertices(torch.tensor([0]))
    >>> float(new_topo.adjacency[0].sum())
    0.0
    """

    def __init__(self, topology: DiscreteTopology) -> None:
        self.topology = DiscreteTopology(adjacency=topology.adjacency.clone())

    def isolate_vertices(
        self, singular_indices: torch.Tensor
    ) -> DiscreteTopology:
        r"""
        Ejecuta la extirpación topológica de los vértices anómalos.

        Fuerza analíticamente :math:`w_{ij}(t^+) = 0` para todo
        :math:`j \in \mathcal{N}(i)`, desconectando completamente a la
        singularidad del resto de la malla (aislamiento completo de fila y
        columna en la matriz de adyacencia).

        Parameters
        ----------
        singular_indices : torch.Tensor
            Vector de índices (1-D, ``dtype=torch.int64`` o similar) de los
            vértices catalogados como anómalos. Si está vacío, la topología
            se retorna sin modificaciones.

        Returns
        -------
        DiscreteTopology
            Nueva variedad topológica con la cicatrización efectuada.

        Complexity
        ----------
        :math:`\mathcal{O}(k_i)` en matrices dispersas (Sparse COO), ya que
        aplica máscaras booleanas exclusivamente sobre las coordenadas activas,
        evitando una iteración densa sobre :math:`V^2`.
        :math:`\mathcal{O}(|V|^2)` en matrices densas (coste del slicing
        ``adj[indices, :] = 0``).

        Notes
        -----
        En formato denso, la operación es un slice-assign en fila y columna.
        En formato Sparse COO, se aplica una máscara booleana sobre
        ``adj.indices()`` filtrando todas las aristas donde aparezca algún
        índice singular como fuente o destino.
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

        # --- Formato Sparse COO ---
        adj_coo = (
            adjacency.to_sparse_coo()
            if adjacency.layout != torch.sparse_coo
            else adjacency
        )
        edge_coords = adj_coo.indices()
        edge_weights = adj_coo.values()

        mask_source = ~torch.isin(edge_coords[0], indices_to_isolate)
        mask_target = ~torch.isin(edge_coords[1], indices_to_isolate)
        valid_edges_mask = mask_source & mask_target

        new_indices = edge_coords[:, valid_edges_mask]
        new_values = edge_weights[valid_edges_mask]

        new_sparse_adj = _sparse_coo(
            new_indices,
            new_values,
            size=adj_coo.size(),
            device=adj_coo.device,
        ).coalesce()

        return DiscreteTopology(adjacency=new_sparse_adj)

    def prune_asymmetric_edges(
        self,
        source_index: int,
        target_indices: torch.Tensor,
    ) -> DiscreteTopology:
        r"""
        Poda aristas direccionales específicas sin aislar el nodo pivote.

        Útil para mitigar vectores de estrés asimétricos (p.ej., ataques DDoS
        selectivos) manteniendo intacto el servicio legítimo periférico del
        nodo comprometido.

        La operación anula las aristas :math:`(s, t_j)` y :math:`(t_j, s)`
        para todo :math:`t_j \in \texttt{target\_indices}`, pero preserva
        las conexiones de :math:`s` con el resto de la vecindad.

        Parameters
        ----------
        source_index : int
            Índice del nodo central o pivote (*server*).
        target_indices : torch.Tensor
            Índices de los nodos periféricos (*clients*) cuya conexión será
            podada (corte bidireccional).

        Returns
        -------
        DiscreteTopology
            La variedad resultante de la poda selectiva.

        Complexity
        ----------
        :math:`\mathcal{O}(k_s)` en matrices dispersas, donde :math:`k_s` es
        el grado del nodo fuente. :math:`\mathcal{O}(|E|)` en el peor caso
        densamente conectado.
        """
        adjacency = self.topology.adjacency

        if not adjacency.is_sparse:
            new_adjacency = adjacency.clone()
            new_adjacency[source_index, target_indices] = 0
            new_adjacency[target_indices, source_index] = 0
            return DiscreteTopology(adjacency=new_adjacency)

        # --- Formato Sparse COO ---
        adj_coo = (
            adjacency.to_sparse_coo()
            if adjacency.layout != torch.sparse_coo
            else adjacency
        )
        edge_coords = adj_coo.indices()
        edge_weights = adj_coo.values()

        is_forward_cut = (edge_coords[0] == source_index) & torch.isin(
            edge_coords[1], target_indices
        )
        is_backward_cut = (edge_coords[1] == source_index) & torch.isin(
            edge_coords[0], target_indices
        )
        valid_edges_mask = ~(is_forward_cut | is_backward_cut)

        new_indices = edge_coords[:, valid_edges_mask]
        new_values = edge_weights[valid_edges_mask]

        new_sparse_adj = _sparse_coo(
            new_indices,
            new_values,
            size=adj_coo.size(),
            device=adj_coo.device,
        ).coalesce()

        return DiscreteTopology(adjacency=new_sparse_adj)
