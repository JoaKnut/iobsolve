r"""
Submódulo del Dominio Discreto (Paper II).

Expone los operadores, estimadores y herramientas de cirugía topológica del
motor discreto (D-IOB) del framework IOB-Solve.

El motor discreto opera sobre 1-esqueletos :math:`\mathcal{G}(V, E, W)` —
grafos ponderados con métrica baricéntrica — cuantificando el Índice de
Estrés Nodal :math:`\mathcal{Q}_i(t)` mediante el Laplaciano-Beltrami
discreto y ejecutando cirugía algorítmica en :math:`\mathcal{O}(k_i)` sobre
matrices dispersas (Sparse COO).

Exports
-------
DiscreteIntegrityOperator
    Operador de Bisagra Discreto. Proyecta el campo de estados sobre
    :math:`\mathbf{L}` para cuantificar el residuo baricéntrico.
RecursiveTopologicalZScore
    Estimador robusto de tensiones basado en MAD con factor de olvido
    exponencial. Breakdown point del 50 %.
TopologicalSurgeon
    Motor de cirugía algorítmica. Extirpa vértices anómalos en
    :math:`\mathcal{O}(k_i)` preservando la conectividad legítima.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto:
       Laplacianos de Grafos, Detección de Anomalías Asíncronas y Colapsos
       en Redes Complejas".
"""

from .hinge import DiscreteIntegrityOperator
from .estimators import RecursiveTopologicalZScore
from .surgery import TopologicalSurgeon

__all__ = [
    "DiscreteIntegrityOperator",
    "RecursiveTopologicalZScore",
    "TopologicalSurgeon",
]
