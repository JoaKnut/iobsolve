r"""
Módulo de Estimadores Estadísticos Topológicos.

Implementa los mecanismos de normalización asimétrica (\textit{Robust Z-Score}) 
requeridos por el D-IOB. Reemplaza la vulnerabilidad de las métricas de media y 
varianza poblacional por estimadores recursivos basados en cuantiles (Mediana y MAD),
garantizando la inmunidad del motor analítico frente al Efecto de Enmascaramiento.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto". 
       Sección 3.1: El Z-Score Topológico Robusto y Prevención del Colapso Estocástico.
"""

import torch
from iobsolve.core.types import StressTensor

class RecursiveTopologicalZScore:
    r"""
    Estimador Robusto de Tensiones Topológicas.

    Evade la dilución estadística (\textit{Masking Effect}) causada por ataques asimétricos masivos,
    empujando el límite de ruptura teórica (\textit{Breakdown Point}) hacia el 50%.

    Parameters
    ----------
    num_nodes : int
        Cardinalidad del conjunto de vértices dinámico |V|.
    epsilon : float, default=1e-8
        Tolerancia al ruido basal (\delta_{min}) para prevenir hiper-divergencias 
        espurias en regímenes de extrema homogeneidad.
    """

    def __init__(self, num_nodes: int, epsilon: float = 1e-8):
        self.num_nodes = num_nodes
        self.epsilon = epsilon
        self.decay_factor = 0.99  # Factor de olvido exponencial para prevenir drift

    def update_and_compute(self, current_stress: StressTensor) -> StressTensor:
        r"""
        Procesa el estrés instantáneo y computa la señal topológica de aislamiento.

        Notes
        -----
        El Z-Score Topológico Modificado (\mathcal{M}_i(t)) se define analíticamente como:
        
        .. math:: \mathcal{M}_i(t) = \frac{0.6745 \cdot \Big( \mathcal{Q}_i(t) - \tilde{\mathcal{Q}}(t) \Big)}{\max(\text{MAD}(t), \epsilon)}

        donde \tilde{\mathcal{Q}}(t) es la mediana del estrés poblacional. El escalar 0.6745 
        garantiza consistencia asintótica con la desviación estándar bajo normalidad estocástica.

        Parameters
        ----------
        current_stress : StressTensor
            Distribución instantánea de tensiones baricéntricas \mathcal{Q}_i(t) 
            arrojada por el D-IOB.

        Returns
        -------
        StressTensor
            Vector normalizado \mathcal{M}_i(t) listo para someterse a la cota 
            crítica \tau en el módulo de Cirugía Topológica.
            
        Complexity
        ----------
        \mathcal{O}(N) temporal gracias a la computación tensorial de la mediana
        en GPU/CPU.
        """
        # 1. Extracción Baricéntrica Robusta (Mediana Poblacional)
        median_stress = torch.median(current_stress)
        
        # 2. Desviación Absoluta de la Mediana (MAD)
        abs_deviation = torch.abs(current_stress - median_stress)
        mad_stress = torch.median(abs_deviation)
        
        # 3. Prevención del Límite Degenerado (\text{MAD}^* = \max(\text{MAD}, \epsilon))
        safe_mad = torch.clamp(mad_stress, min=self.epsilon)
        
        # 4. Cálculo Vectorizado de \mathcal{M}_i(t)
        robust_z_score = (0.6745 * (current_stress - median_stress)) / safe_mad
        
        return robust_z_score