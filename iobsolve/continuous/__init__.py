r"""
Dominio Continuo del Framework IOB-Solve (Continuous).

Este submódulo instrumenta la axiomatización geométrica para variedades euclidianas 
diferenciables. Incluye el Operador de Bisagra (IOB) primigenio para aislar 
singularidades topológicas y el Teorema del Flujo de Integridad, el cual acopla 
el mapeo espectral con particiones espaciales para evadir el cálculo del Jacobiano 
en sistemas dinámicos.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra: Aislamiento 
       Topológico de Singularidades y Control de Bifurcaciones en Variedades Continuas".
"""

from .hinge import ContinuousIntegrityOperator
from .flow_theorem import FlowTheoremLocator

__all__ = [
    "ContinuousIntegrityOperator",
    "FlowTheoremLocator",
]