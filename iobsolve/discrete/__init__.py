r"""
Dominio Discreto del Framework IOB-Solve (Discrete).

Este submódulo instrumenta la axiomatización geométrica para topologías ralas, 
redes complejas y series temporales asíncronas. Incluye el Operador de Bisagra 
Discreto (D-IOB), el motor de Cirugía Topológica y los estimadores estadísticos 
robustos diseñados para eludir el "Masking Effect".

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto". 
       IOB-Solve Research Paper Archive.
"""

from .hinge import DiscreteIntegrityOperator
from .surgery import TopologicalSurgeon
from .estimators import RecursiveTopologicalZScore

__all__ = [
    "DiscreteIntegrityOperator",
    "TopologicalSurgeon",
    "RecursiveTopologicalZScore",
]