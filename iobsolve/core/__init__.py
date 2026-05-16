r"""
Núcleo Matemático del Framework IOB-Solve (Core).

Este submódulo agrupa las primitivas topológicas, los operadores diferenciales 
(Laplacianos continuos y discretos) y las abstracciones geométricas necesarias 
para fundamentar el Operador de Integridad de Bisagra.

Todo el ecosistema tensorial está diseñado sobre el motor de PyTorch para 
garantizar la preservación de gradientes (Autograd) y la ejecución nativa en GPU.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra: Aislamiento 
       Topológico de Singularidades y Control de Bifurcaciones en Variedades Continuas".
.. [2] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto: 
       Laplacianos de Grafos, Detección de Anomalías Asíncronas y Colapsos en Redes Complejas".
"""

from .base_operator import BaseIntegrityOperator
from .laplacian import ContinuousLaplacian, DiscreteLaplacian
from .space import EuclideanManifold, DiscreteTopology
from .types import ManifoldField, StressTensor, AdjacencyMatrix, NodalStateVector

__all__ = [
    "BaseIntegrityOperator",
    "ContinuousLaplacian",
    "DiscreteLaplacian",
    "EuclideanManifold",
    "DiscreteTopology",
    "ManifoldField",
    "StressTensor",
    "AdjacencyMatrix",
    "NodalStateVector",
]