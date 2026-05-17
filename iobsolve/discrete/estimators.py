r"""
Módulo de Estimadores Estadísticos Topológicos.

Implementa los mecanismos de normalización asimétrica (*Robust Z-Score*)
requeridos por el D-IOB. Reemplaza la vulnerabilidad de las métricas de media y
varianza poblacional por estimadores recursivos basados en cuantiles (Mediana y MAD),
garantizando la inmunidad del motor analítico frente al Efecto de Enmascaramiento.

El estimador implementa adicionalmente un factor de olvido exponencial
(``decay_factor``) que suprime la inercia de observaciones antiguas mediante
una mediana ponderada exponencialmente, previniendo el *concept drift* en
flujos de tráfico no estacionarios.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto".
       Sección 3.1: El Z-Score Topológico Robusto y Prevención del Colapso Estocástico.
.. [2] Leys, C. et al. (2013). "Detecting outliers: Do not use standard deviation
       around the mean, use absolute deviation around the median".
       *Journal of Experimental Social Psychology*, 49(4), 764–766.
"""

import torch
from iobsolve.core.types import StressTensor


class RecursiveTopologicalZScore:
    r"""
    Estimador Robusto de Tensiones Topológicas con factor de olvido exponencial.

    Evade la dilución estadística (*Masking Effect*) causada por ataques
    asimétricos masivos, empujando el límite de ruptura teórica (*Breakdown
    Point*) hacia el 50 % mediante el uso de la Mediana y la Desviación
    Absoluta de la Mediana (MAD) como estimadores de localización y escala.

    El factor de olvido exponencial ``decay_factor`` :math:`\lambda \in (0, 1)`
    pondera el estrés histórico de forma que observaciones antiguas contribuyan
    menos al estimado actual, siguiendo la recurrencia:

    .. math::

        \tilde{Q}^*(t) = \lambda \cdot \tilde{Q}(t-1) + (1 - \lambda) \cdot Q_i(t)

    Donde :math:`\tilde{Q}(t)` es la mediana del estrés actual. En esta
    implementación la ponderación se aplica antes de calcular la mediana,
    mezclando el historial previo con la observación corriente.

    Parameters
    ----------
    num_nodes : int
        Cardinalidad del conjunto de vértices dinámico :math:`|V|`.
    epsilon : float, default=1e-8
        Tolerancia al ruido basal (:math:`\delta_{\min}`) para prevenir
        hiper-divergencias espurias en regímenes de extrema homogeneidad
        (:math:`\text{MAD} \approx 0`, e.g., todos los nodos con el mismo tráfico).
    decay_factor : float, default=0.99
        Factor de olvido exponencial :math:`\lambda \in (0, 1)`. Valores
        cercanos a 1 dan mayor peso al historial; valores bajos (~0.9) hacen
        al estimador más reactivo a cambios abruptos.

    Attributes
    ----------
    _prev_median : torch.Tensor or None
        Mediana poblacional del instante anterior. ``None`` en el primer ciclo.

    Examples
    --------
    Detección de anomalía en un vector de estrés con un outlier extremo:

    >>> import torch
    >>> from iobsolve.discrete.estimators import RecursiveTopologicalZScore
    >>> estimator = RecursiveTopologicalZScore(num_nodes=10)
    >>> stress = torch.ones(10, dtype=torch.float64)
    >>> stress[7] = 1000.0  # nodo anómalo
    >>> z_scores = estimator.update_and_compute(stress)
    >>> z_scores[7] > 3.0
    tensor(True)
    """

    def __init__(
        self,
        num_nodes: int,
        epsilon: float = 1e-8,
        decay_factor: float = 0.99,
    ):
        self.num_nodes = num_nodes
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self._prev_median: torch.Tensor | None = None

    def update_and_compute(self, current_stress: StressTensor) -> StressTensor:
        r"""
        Procesa el estrés instantáneo y computa la señal topológica de aislamiento.

        Aplica el factor de olvido exponencial sobre la mediana histórica antes
        de calcular el Z-Score, suavizando la señal frente a ráfagas transitorias
        y previniendo el *concept drift* en flujos no estacionarios.

        Notes
        -----
        El Z-Score Topológico Modificado :math:`\mathcal{M}_i(t)` se define como:

        .. math::

            \mathcal{M}_i(t) = \frac{0.6745 \cdot \Big( Q_i(t) - \tilde{Q}^*(t) \Big)}
                                     {\max\!\big(\text{MAD}(t),\, \varepsilon\big)}

        donde :math:`\tilde{Q}^*(t)` es la mediana ponderada exponencialmente
        y el escalar :math:`0.6745` garantiza consistencia asintótica con la
        desviación estándar bajo normalidad estocástica.

        Parameters
        ----------
        current_stress : StressTensor
            Distribución instantánea de tensiones baricéntricas :math:`Q_i(t)`
            arrojada por el D-IOB. Tensor 1-D de longitud :math:`|V|`.

        Returns
        -------
        StressTensor
            Vector normalizado :math:`\mathcal{M}_i(t)` listo para someterse a
            la cota crítica :math:`\tau` en el módulo de Cirugía Topológica.

        Complexity
        ----------
        :math:`\mathcal{O}(|V| \log |V|)` por el cómputo de la mediana (sort interno).
        Dominado asintóticamente por la clasificación; todos los demás pasos son
        :math:`\mathcal{O}(|V|)`.
        """
        # 1. Extracción Baricéntrica con factor de olvido exponencial
        raw_median = torch.median(current_stress)

        if self._prev_median is None:
            # Primera iteración: sin historial previo
            blended_median = raw_median
        else:
            # Mezcla exponencial: lambda * historia + (1-lambda) * actual
            blended_median = (
                self.decay_factor * self._prev_median
                + (1.0 - self.decay_factor) * raw_median
            )

        self._prev_median = blended_median.detach()

        # 2. Desviación Absoluta de la Mediana (MAD)
        abs_deviation = torch.abs(current_stress - blended_median)
        mad_stress = torch.median(abs_deviation)

        # 3. Prevención del Límite Degenerado (MAD* = max(MAD, epsilon))
        safe_mad = torch.clamp(mad_stress, min=self.epsilon)

        # 4. Cálculo Vectorizado de M_i(t)
        robust_z_score = (0.6745 * (current_stress - blended_median)) / safe_mad

        return robust_z_score
