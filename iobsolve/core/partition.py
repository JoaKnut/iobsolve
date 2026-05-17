r"""
Módulo de Partición Espacial (IOB-QuadTree).

Implementa el motor de subdivisión de hipercubos para la localización 
geometrométrica de singularidades. Este módulo materializa el concepto 
de aislamiento espacial en el Dominio Continuo, ramificando el espacio 
de fases de manera recursiva mediante estructuras de datos jerárquicas 
(QuadTree/Octree N-dimensionales).

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra: Aislamiento 
       Topológico de Singularidades y Control de Bifurcaciones en Variedades Continuas".
"""

import itertools
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

#: Definición topológica de un subdominio euclidiano \Omega \subset \mathbb{R}^n.
#: Representado como el producto cartesiano de n intervalos: ((x_min, x_max), (y_min, y_max), ...)
Hypercube = Tuple[Tuple[float, float], ...]

#: Criterio de Integridad. Función evaluadora inyectable (Closure).
IntegrityCriterion = Callable[[Hypercube], bool]

@dataclass
class PartitionNode:
    r"""
    Estructura de datos recursiva que representa un subdominio \Omega_i.
    
    Actúa como un nodo en el árbol de partición espacial, preservando 
    la topología de su región y la existencia de singularidades internas.
    """
    domain: Hypercube
    depth: int
    contains_singularity: bool
    children: Optional[List['PartitionNode']] = None

    @property
    def is_leaf(self) -> bool:
        r"""Determina si el nodo es una hoja terminal del árbol."""
        return self.children is None or len(self.children) == 0


class SpatialPartitionEngine:
    r"""
    Motor IOB-QuadTree (N-dimensional).
    
    Orquesta la discretización topológica del espacio continuo aislando 
    regiones que superan la tolerancia crítica de estrés estructural.

    Parameters
    ----------
    max_depth : int, default=10
        Límite máximo de recursión para prevenir el colapso asintótico 
        frente a singularidades puntuales irreducibles.
    min_volume_tolerance : float, default=1e-12
        Mínima medida de Lebesgue \mu(\Omega) permitida antes de detener 
        forzosamente la partición, mitigando errores de punto flotante.
    """

    def __init__(self, max_depth: int = 10, min_volume_tolerance: float = 1e-12):
        self.max_depth = max_depth
        self.min_volume_tolerance = min_volume_tolerance

    @staticmethod
    def _compute_volume(domain: Hypercube) -> float:
        r"""
        Computa la medida de Lebesgue (volumen) del hipercubo \mu(\Omega).
        
        Notes
        -----
        .. math:: \mu(\Omega) = \prod_{k=1}^{n} (x_{k,\max} - x_{k,\min})
        """
        volume = 1.0
        for axis_min, axis_max in domain:
            volume *= (axis_max - axis_min)
        return volume

    @staticmethod
    def subdivide_hypercube(domain: Hypercube) -> List[Hypercube]:
        r"""
        Divide un hipercubo \Omega \subset \mathbb{R}^n en 2^n sub-hipercubos disjuntos.
        
        Parameters
        ----------
        domain : Hypercube
            Hipercubo espacial a biseccionar.
            
        Returns
        -------
        List[Hypercube]
            Lista de 2^n subdominios con igual volumen métrico.
        """
        midpoints = [(axis_min + axis_max) / 2.0 for axis_min, axis_max in domain]
        
        intervals_per_dim = [
            ((float(d[0]), float(m)), (float(m), float(d[1]))) 
            for d, m in zip(domain, midpoints)
        ]
        
        # cast estructural por diseño del itertools.product
        subdomains: List[Hypercube] = list(itertools.product(*intervals_per_dim))  # type: ignore
        return subdomains

    def isolate_singularities(self, 
                              current_domain: Hypercube, 
                              criterion: IntegrityCriterion, 
                              current_depth: int = 0) -> PartitionNode:
        r"""
        Construye el árbol recursivamente aislando regiones con estrés topológico.
        
        Parameters
        ----------
        current_domain : Hypercube
            Región actual evaluada por el operador.
        criterion : IntegrityCriterion
            Función booleana que retorna True si la región alberga una singularidad.
        current_depth : int, default=0
            Profundidad actual en la pila de recursión.
            
        Returns
        -------
        PartitionNode
            El árbol jerárquico encapsulando las singularidades.
        """
        # Evaluación de la integral/estrés de contorno
        has_singularity = criterion(current_domain)
        
        # Criterios de parada algorítmica
        if (not has_singularity) or \
           (current_depth >= self.max_depth) or \
           (self._compute_volume(current_domain) < self.min_volume_tolerance):
            
            return PartitionNode(
                domain=current_domain,
                depth=current_depth,
                contains_singularity=has_singularity,
                children=None
            )

        # Ramificación
        subdomains = self.subdivide_hypercube(current_domain)
        
        children = [
            self.isolate_singularities(sub, criterion, current_depth + 1)
            for sub in subdomains
        ]

        return PartitionNode(
            domain=current_domain,
            depth=current_depth,
            contains_singularity=True,
            children=children
        )

    def extract_singular_manifolds(self, root_node: PartitionNode) -> List[Hypercube]:
        r"""
        Extrae únicamente los hipercubos terminales con una singularidad topológica.
        
        Parameters
        ----------
        root_node : PartitionNode
            Nodo raíz resultante del proceso de bisección espacial.
            
        Returns
        -------
        List[Hypercube]
            Lista plana de los subdominios catalogados como críticos.
        """
        singularities: List[Hypercube] = []

        def traverse(node: PartitionNode) -> None:
            if node.is_leaf:
                if node.contains_singularity:
                    singularities.append(node.domain)
            elif node.children:
                for child in node.children:
                    traverse(child)

        traverse(root_node)
        return singularities