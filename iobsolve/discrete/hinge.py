r"""
Módulo del Operador de Bisagra Discreto (D-IOB).

Este módulo implementa la axiomatización geométrica para redes complejas 
y topologías ralas, permitiendo cuantificar la divergencia de un nodo respecto 
al baricentro de su vecindad. El operador se soporta íntegramente en el motor 
de diferenciación automática de PyTorch (Autograd) y operaciones tensoriales 
dispersas para garantizar una latencia asintótica de \mathcal{O}(k_i).

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto: 
       Laplacianos de Grafos, Detección de Anomalías Asíncronas y Colapsos en Redes Complejas". 
       IOB-Solve Research Paper Archive.
"""

from typing import Tuple, Literal, Any, cast
import torch

from iobsolve.core.base_operator import BaseIntegrityOperator
from iobsolve.core.laplacian import DiscreteLaplacian
from iobsolve.core.space import DiscreteTopology
from iobsolve.core.types import NodalStateVector, StressTensor

_vector_norm: Any = getattr(torch.linalg, "vector_norm")
_sparse_mm: Any = getattr(torch.sparse, "mm")


class DiscreteIntegrityOperator(BaseIntegrityOperator):
    r"""
    Operador de Bisagra para el Dominio Discreto (D-IOB).

    Evalúa el Índice de Estrés Nodal de una malla topológica proyectando el campo 
    vectorial de estado sobre el Laplaciano-Beltrami discreto.

    Parameters
    ----------
    epsilon_tolerance : float, default=1e-8
        Regularizador estrictamente vinculado al límite de precisión computacional 
        (\varepsilon_{mach}) para prevenir divisiones por cero en el límite de vacío topológico.

    Notes
    -----
    El operador calcula el Índice de Estrés Nodal \mathcal{Q}_i(t), definido teóricamente como:
    
    .. math:: \mathcal{Q}_i(t) = \frac{\|\mathbf{R}_i(t)\|_2^2}{m \cdot \Big( \sum w_{ij}(t) + \epsilon \Big)^2}

    donde \mathbf{R}_i(t) es el residuo baricéntrico equivalente a la acción negativa 
    del Laplaciano no normalizado sobre el campo vectorial.
    """

    def __init__(self, epsilon_tolerance: float = 1e-8):
        super().__init__(epsilon_tolerance)

    def compute_stress(self, 
                       state_tensor: NodalStateVector, 
                       *,
                       topology: DiscreteTopology, 
                       laplacian_type: Literal['combinatorial', 'normalized'] = 'normalized',
                       normalize_output: bool = True,
                       **kwargs: Any) -> StressTensor:
        r"""
        Calcula el vector de estrés nodal macroscópico para todo el grafo.

        Parameters
        ----------
        state_tensor : NodalStateVector
            Tensor de estado \mathbf{x}_i(t) \in \mathbb{R}^{N \times m} asociado a la red.
        topology : DiscreteTopology
            La variedad topológica \mathcal{G}(V, E, W) que rige la conectividad.
        laplacian_type : {'combinatorial', 'normalized'}, default='normalized'
            Determina si la divergencia se evalúa frente al Laplaciano clásico (\mathbf{D} - \mathbf{W}) 
            o a su variante normalizada isométricamente.
        normalize_output : bool, default=True
            Si es True, aplica un escalado Min-Max al tensor resultante, confinando 
            el estrés al intervalo probabilístico [0, 1].

        Returns
        -------
        StressTensor
            Tensor unidimensional de tamaño N conteniendo el valor escalar \mathcal{Q}_i 
            para cada vértice de la topología.
            
        Complexity
        ----------
        \mathcal{O}(\langle k \rangle) por vértice evaluado localmente, evadiendo la 
        propagación tensorial densa \mathcal{O}(N^3) gracias a la instrumentación sparse_coo.
        """
        topology.validate_nodal_state(state_tensor)

        if laplacian_type == 'normalized':
            laplacian = DiscreteLaplacian.compute_normalized(topology.adjacency)
        else:
            laplacian = DiscreteLaplacian.compute_combinatorial(topology.adjacency)

        # 3. Proyección topológica (\mathbf{R} = -\mathbf{L} \mathbf{X})
        if state_tensor.dim() == 1:
            if laplacian.is_sparse:
                divergence = torch.mv(laplacian, state_tensor)
            else:
                divergence = laplacian @ state_tensor
        else:
            if laplacian.is_sparse:
                divergence = cast(torch.Tensor, _sparse_mm(laplacian, state_tensor))
            else:
                divergence = laplacian @ state_tensor

        # 4. Cuantificación de la magnitud del estrés
        if divergence.dim() > 1:
            stress_vector = cast(torch.Tensor, _vector_norm(divergence, dim=-1))
        else:
            stress_vector = torch.abs(divergence)

        # 5. Mapeo relativo \in [0, 1]
        if normalize_output:
            max_stress = torch.max(stress_vector)
            if max_stress > self.epsilon_tolerance:
                stress_vector = stress_vector / max_stress
            else:
                stress_vector = torch.zeros_like(stress_vector)

        return stress_vector

    def locate_singularities(self, 
                             stress_tensor: StressTensor, 
                             threshold: float) -> Tuple[torch.Tensor, ...]:
        r"""
        Aísla los índices de los vértices que superan el umbral de tolerancia crítica.

        Parameters
        ----------
        stress_tensor : StressTensor
            El vector de estrés o Z-Score topológico \mathcal{M}_i(t) computado.
        threshold : float
            El límite crítico de tolerancia \tau (típicamente \tau \geq 3.0 por la 
            desigualdad de Chebyshev).

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Tensores de índices que apuntan a los nodos topológicamente anómalos.
        """
        singular_nodes = torch.where(stress_tensor > threshold)
        return singular_nodes