import numpy as np
from collections import deque
from typing import Tuple, Optional

from .types import IOBSystem
from .space import SphereSampler, MonteCarloSampler

class HingeIntegrityOperator:
    """
    Operador de Integridad de Bisagra (IOB) para la evaluación de suavidad métrica.

    Cuantifica la deformación topológica de un sistema dinámico mediante 
    un muestreo bilateral simétrico en el espacio de fases.

    Parameters
    ----------
    L_metric : float
        Radio métrico de exploración (resolución óptica del operador).
    epsilon : float, optional
        Término de regularización para evitar divisiones por cero (default: 1e-9).
    """

    def __init__(self, L_metric: float, epsilon: float = 1e-9):
        if L_metric <= 0:
            raise ValueError("L_metric debe ser estrictamente positivo.")
            
        self.L = L_metric
        self.epsilon = epsilon

    def evaluate_directional_stress(self, system: IOBSystem, pivot: np.ndarray, u: np.ndarray) -> float:
        """
        Evalúa el estrés geométrico en una dirección unitaria específica.
        """
        # Desplazamiento bilateral
        state_plus = pivot + self.L * u
        state_minus = pivot - self.L * u
        
        # Evaluación del campo
        val_pivot = system.evaluate(pivot)
        val_plus = system.evaluate(state_plus)
        val_minus = system.evaluate(state_minus)
        
        # Cancelación de curvatura impar (Aproximación del Laplaciano 1D)
        residual = val_plus + val_minus - 2 * val_pivot
        norm_residual_sq = np.sum(residual ** 2)
        
        df_u = system.evaluate_jacobian(pivot, u)
        
        if df_u is not None:
            norm_df_sq = np.sum(df_u ** 2)
            stress = norm_residual_sq / (2 * (self.L ** 2) * (norm_df_sq + self.epsilon))
        else:
            # Fallback para sistemas de caja negra
            stress = norm_residual_sq / (2 * (self.L ** 4) + self.epsilon)
            
        return float(stress)

    def evaluate_isotropic_stress(self,
                                  system: IOBSystem,
                                  pivot: np.ndarray,
                                  sampler: Optional[SphereSampler] = None,
                                  samples: int = 10) -> float:
        """
        Calcula el estrés isotrópico integrando sobre la hiperesfera S^(n-1).
        
        TODO: Implementar multiprocesamiento o vectorización pura en Numpy 
        para el bucle de direcciones si `samples` es muy elevado.
        """
        sampler = sampler or MonteCarloSampler()
        directions = sampler.sample(system.dimensionality, samples)
        
        # TODO: Refactorizar a list comprehension + sum() para mayor velocidad en C
        stress_sum = sum(
            self.evaluate_directional_stress(system, pivot, u) 
            for u in directions
        )
        
        return stress_sum / len(directions)


class TopologicalCrisisPredictor:
    """
    Sensor Dinámico Euleriano-Lagrangiano.
    
    Procesa un flujo temporal de métricas de estrés (Q_d) para calcular la 
    aceleración de la deformación (Psi_c) e identificar rupturas sistémicas.
    """

    def __init__(self, alert_threshold: float, history_size: int = 20):
        self.alert_threshold = alert_threshold
        self.max_history = history_size
        # El uso de deque con maxlen asegura un O(1) en inserción y memoria acotada
        self.stress_history = deque(maxlen=history_size)

    def push_stress_state(self, q_d: float) -> Tuple[bool, float]:
        """
        Ingresa la densidad de estrés instantánea y evalúa el riesgo de crisis.

        Returns
        -------
        Tuple[bool, float]
            (is_crisis, psi_c_value)
        """
        s_t = np.log(q_d + 1e-9)
        self.stress_history.append(s_t)
        
        if len(self.stress_history) < 3:
            return False, 0.0
            
        # Filtro laplaciano discreto 1D (Diferencias finitas centrales)
        st = len(self.stress_history) - 1
        psi_c = abs(
            self.stress_history[st] 
            - 2 * self.stress_history[st-1] 
            + self.stress_history[st-2]
        )
        
        return bool(psi_c > self.alert_threshold), float(psi_c)