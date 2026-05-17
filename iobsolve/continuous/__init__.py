r"""
Submódulo del Dominio Continuo (Paper I).

Expone el :class:`~iobsolve.continuous.hinge.ContinuousIntegrityOperator`
y el :class:`~iobsolve.continuous.flow_theorem.FlowTheoremLocator` como
la API pública del motor continuo del framework IOB-Solve.

El motor continuo opera sobre variedades euclidianas diferenciables
:math:`\Omega \subset \mathbb{R}^n`, cuantificando el estrés geométrico local
mediante el Laplaciano continuo y localizando singularidades mediante la
bisección recursiva del espacio de fases (IOB-QuadTree) combinada con el
análisis espectral de alta frecuencia (IOB-FFT).

Exports
-------
ContinuousIntegrityOperator
    Operador de Bisagra para el dominio continuo. Evalúa
    :math:`\mathcal{H}(x) = |\nabla^2 \phi(x)|`.
FlowTheoremLocator
    Localizador de raíces y singularidades basado en el Teorema del Flujo
    de Integridad (TVI + IOB-FFT + IOB-QuadTree).

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra: Aislamiento
       Topológico de Singularidades y Control de Bifurcaciones en Variedades
       Continuas".
"""

from .hinge import ContinuousIntegrityOperator
from .flow_theorem import FlowTheoremLocator

__all__ = [
    "ContinuousIntegrityOperator",
    "FlowTheoremLocator",
]
