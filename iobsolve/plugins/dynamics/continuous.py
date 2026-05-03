import numpy as np
import logging
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal
from typing import Callable, Tuple, List, Any, cast

from iobsolve.core.types import IOBSystem
from iobsolve.core.operator import HingeIntegrityOperator, TopologicalCrisisPredictor
from iobsolve.core.space import SphereSampler

logger = logging.getLogger(__name__)

class ContinuousDynamicSystem(IOBSystem):
    """
    Abstracción matemática para un sistema de EDOs: dx/dt = F(t, x).
    
    Parameters
    ----------
    vector_field : Callable
        Función que recibe (t, x) y retorna el vector velocidad.
    dim : int
        Dimensionalidad del espacio de fases (N).
    """
    def __init__(self, vector_field: Callable[[float, np.ndarray], np.ndarray], dim: int):
        self.F = vector_field
        self._dim = dim

    @property
    def dimensionality(self) -> int:
        return self._dim

    def evaluate(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        return self.F(t, state)


class GeometricImportanceSampler(SphereSampler):
    """
    Muestreo de Importancia Geométrico (GIS).
    
    Reduce la varianza estocástica extrayendo vectores direccionales (u) 
    sesgados hacia la matriz de covarianza de la trayectoria reciente. 
    Crucial para vencer la dimensionalidad O(N^3) del cálculo de Jacobianas.
    """
    def __init__(self, history_size: int = 10):
        self.history_size = history_size
        self.trajectory_increments: List[np.ndarray] = []

    def update_trajectory(self, delta_x: np.ndarray) -> None:
        self.trajectory_increments.append(delta_x)
        if len(self.trajectory_increments) > self.history_size:
            self.trajectory_increments.pop(0)

    def sample(self, dimensionality: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna vectores unitarios y sus probabilidades de densidad.
        """
        if len(self.trajectory_increments) < 2:
            # Fallback a muestreo isótropo estándar (Ruido Gaussiano Normalizado)
            vecs = np.random.randn(n_samples, dimensionality)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / (norms + 1e-15), np.ones(n_samples)

        increments_matrix = np.vstack(self.trajectory_increments)
        
        # TODO: np.cov es O(N^2 * history_size). Para N > 1000, considerar aproximaciones
        # dispersas o decaimiento exponencial estocástico.
        C = np.cov(increments_matrix, rowvar=False)
        
        # Regularización de Tikhonov suave para garantizar que C es definida positiva
        C += np.eye(dimensionality) * 1e-6
        C_safe = cast(Any, C)
        
        dist = multivariate_normal(mean=np.zeros(dimensionality), cov=C_safe, allow_singular=True)
        raw_samples = np.atleast_2d(dist.rvs(size=n_samples))
        
        norms = np.linalg.norm(raw_samples, axis=1, keepdims=True)
        u_vectors = raw_samples / (norms + 1e-15)
        
        p_u = np.atleast_1d(dist.pdf(raw_samples))
        
        return u_vectors, p_u


class DynamicsCrisisMonitor:
    """
    Monitor Integrador Lagrangiano.
    
    Integra el sistema dinámico continuo paso a paso, inyectando el operador 
    IOB en cada chunk temporal para medir el estrés deformacional Psi_c(t).
    
    Parameters
    ----------
    system : ContinuousDynamicSystem
        El sistema a integrar.
    alert_threshold : float, optional
        Límite de ruptura sistémica (default: 10.0).
    l_metric : float, optional
        Longitud de la sonda del laplaciano (default: 1e-3).
    use_gis : bool, optional
        Si True, utiliza muestreo geométrico sesgado (default: True).
    """
    def __init__(self, 
                 system: ContinuousDynamicSystem, 
                 alert_threshold: float = 10.0,
                 l_metric: float = 1e-3,
                 use_gis: bool = True):
                 
        self.system = system
        self.operator = HingeIntegrityOperator(L_metric=l_metric)
        self.predictor = TopologicalCrisisPredictor(alert_threshold=alert_threshold)
        
        self.use_gis = use_gis
        self.sampler = GeometricImportanceSampler() if use_gis else None

    def _evaluate_dynamic_stress(self, state: np.ndarray, t: float, n_samples: int = 10) -> float:
        """Pondera el tensor direccional basándose en el pdf muestral."""
        if not (self.use_gis and self.sampler):
            return self.operator.evaluate_isotropic_stress(self.system, state, samples=n_samples)

        u_vectors, p_u = self.sampler.sample(self.system.dimensionality, n_samples)
        stress_sum = 0.0
        
        # FIXME: Bucle lento en Python plano. Si `evaluate` soporta broadcasting (vectorización),
        # reescribir este paso algebraicamente usando np.einsum o Numba.
        for i, u in enumerate(u_vectors):
            state_plus = state + self.operator.L * u
            state_minus = state - self.operator.L * u
            
            val_pivot = self.system.evaluate(state, t)
            val_plus = self.system.evaluate(state_plus, t)
            val_minus = self.system.evaluate(state_minus, t)
            
            res = val_plus + val_minus - 2 * val_pivot
            q_dir = np.sum(res**2) / (2 * (self.operator.L ** 4) + self.operator.epsilon)
            
            weight = 1.0 / (p_u[i] + 1e-10) 
            stress_sum += q_dir * weight
            
        return float(stress_sum / n_samples)

    def simulate_and_monitor(self, 
                             initial_state: np.ndarray, 
                             t_span: Tuple[float, float], 
                             dt: float = 0.01,
                             samples_per_step: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integra y rastrea la variedad métrica.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (Tiempos evaluados, Matriz de trayectoria, Serie temporal de Psi_c)
        """
        t_eval = np.arange(t_span[0], t_span[1], dt)
        
        # RK45 es estándar, pero puede sufrir stiffness. Considerar 'Radau' o 'BDF'
        # si el vector_field es altamente no-lineal cerca del atractor.
        solution = solve_ivp(self.system.F, t_span, initial_state, t_eval=t_eval, method='RK45')
        
        trajectory = solution.y.T
        times = solution.t
        psi_c_telemetry = np.zeros_like(times)
        
        prev_state = trajectory[0]
        crisis_logged = False 
        
        for idx, current_state in enumerate(trajectory):
            if self.use_gis and self.sampler and idx > 0:
                self.sampler.update_trajectory(current_state - prev_state)
                
            q_d = self._evaluate_dynamic_stress(current_state, times[idx], n_samples=samples_per_step)
            is_crisis, psi_c = self.predictor.push_stress_state(q_d)
            
            if is_crisis and not crisis_logged:
                logger.warning(f"Crisis topológica inminente detectada en t={times[idx]:.2f}")
                crisis_logged = True
                
            psi_c_telemetry[idx] = psi_c
            prev_state = current_state
            
        return times, trajectory, psi_c_telemetry