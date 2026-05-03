import numpy as np
import logging
from scipy.integrate import solve_ivp
from typing import Tuple, Union

from iobsolve.plugins.dynamics.continuous import ContinuousDynamicSystem, GeometricImportanceSampler
from iobsolve.core.operator import HingeIntegrityOperator, TopologicalCrisisPredictor

logger = logging.getLogger(__name__)

class TopologicalSurgeon:
    """
    Controlador No-Lineal que implementa Cirugía Topológica.
    
    Evalúa el tensor geométrico del sistema y, en caso de crisis inminente, 
    inyecta un vector de control estabilizador proporcional al exceso de estrés.
    
    Parameters
    ----------
    system : ContinuousDynamicSystem
        El sistema dinámico base sin perturbar.
    alert_threshold : float, optional
        Umbral crítico de estrés Psi_c (default: 5.0).
    control_gain : float, optional
        Ganancia del controlador (kappa) para la fricción inducida (default: 0.5).
    l_metric : Union[float, str], optional
        Radio métrico. Soporta modo "auto" para calibración online (default: "auto").
    use_gis : bool, optional
        Habilita Muestreo Geométrico Sesgado para optimizar O(N) (default: True).
    """
    def __init__(self, 
                 system: ContinuousDynamicSystem, 
                 alert_threshold: float = 5.0,
                 control_gain: float = 0.5,
                 l_metric: Union[float, str] = "auto",
                 use_gis: bool = True):
                 
        self.base_system = system
        self.alert_threshold = alert_threshold
        self.kappa = control_gain
        
        self.auto_tune = (l_metric == "auto")
        initial_l = 0.1 if self.auto_tune else float(l_metric)
        
        self.operator = HingeIntegrityOperator(L_metric=initial_l)
        self.predictor = TopologicalCrisisPredictor(alert_threshold=alert_threshold)
        
        if self.auto_tune:
            from iobsolve.core.calibration import MetricCalibrator
            # Calibración prolongada para absorber transitorios iniciales (50 pasos)
            self.calibrator = MetricCalibrator(target_baseline_qd=1.0, calibration_steps=50)
            
        self.use_gis = use_gis
        self.sampler = GeometricImportanceSampler() if use_gis else None

    def _evaluate_stress(self, state: np.ndarray, t: float, n_samples: int = 6) -> float:
        """
        Computa el estrés métrico instantáneo Q_d(t).
        Si auto_tune está activo, asiste al calibrador y retorna 0 temporalmente.
        """
        if not (self.use_gis and self.sampler):
            return self.operator.evaluate_isotropic_stress(self.base_system, state, samples=n_samples)

        u_vectors, p_u = self.sampler.sample(self.base_system.dimensionality, n_samples)
        stress_sum = 0.0
        raw_residual_sum = 0.0 
        
        # FIXME: Vectorizar usando broadcasting si la función base soporta arrays N-D.
        # Evaluar múltiples u's de forma iterativa impacta en el dt máximo permitido.
        for i, u in enumerate(u_vectors):
            s_plus = state + self.operator.L * u
            s_minus = state - self.operator.L * u
            
            v_pivot = self.base_system.evaluate(state, t)
            v_plus = self.base_system.evaluate(s_plus, t)
            v_minus = self.base_system.evaluate(s_minus, t)
            
            res = v_plus + v_minus - 2 * v_pivot
            norm_res_sq = np.sum(res**2)
            raw_residual_sum += norm_res_sq
            
            q_dir = norm_res_sq / (2 * (self.operator.L ** 4) + self.operator.epsilon)
            weight = 1.0 / (p_u[i] + 1e-10) 
            stress_sum += q_dir * weight
            
        if self.auto_tune:
            if not self.calibrator.is_calibrated:
                self.calibrator.push_residual(raw_residual_sum / n_samples)
                return 0.0 
            self.operator.L = self.calibrator.optimal_L
            
        return float(stress_sum / n_samples)

    def simulate_and_control(self, 
                             initial_state: np.ndarray, 
                             t_span: Tuple[float, float], 
                             dt: float = 0.05,
                             samples_per_step: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Integra el sistema cerrado en chunks temporales, inyectando 
        el esfuerzo estabilizador causal de forma dinámica.
        """
        t_eval = np.arange(t_span[0], t_span[1], dt)
        n_steps = len(t_eval)
        dim = self.base_system.dimensionality
        
        # Pre-alocación estática para mayor rendimiento en RAM
        trajectory = np.zeros((n_steps, dim))
        psi_c_telemetry = np.zeros(n_steps)
        control_telemetry = np.zeros(n_steps)
        
        trajectory[0] = initial_state
        current_state = initial_state
        
        logger.info("Iniciando Cirugía Topológica Dinámica...")
        
        for idx in range(n_steps - 1):
            t_current, t_next = t_eval[idx], t_eval[idx + 1]
            
            # --- EVALUACIÓN Y PREDICCIÓN EULERIANA ---
            if self.use_gis and self.sampler and idx > 0:
                self.sampler.update_trajectory(current_state - trajectory[idx-1])
                
            q_d = self._evaluate_stress(current_state, t_current, n_samples=samples_per_step)
            is_crisis, current_psi_c = self.predictor.push_stress_state(q_d)
            psi_c_telemetry[idx] = current_psi_c
            
            # --- LEY DE CONTROL (Amortiguación Geométrica) ---
            # Penaliza al sistema proporcionalmente al gradiente topológico asimétrico
            if is_crisis:
                excess_stress = current_psi_c - self.alert_threshold
                control_factor = self.kappa * excess_stress
            else:
                control_factor = 0.0
                
            control_telemetry[idx] = control_factor

            # Inyección de Control: U(x) = -\kappa * F_base(x)
            def controlled_vector_field(t: float, x: np.ndarray) -> np.ndarray:
                base_velocity = self.base_system.evaluate(x, t)
                return base_velocity - (control_factor * base_velocity)

            # Integración de paso corto (Chunk)
            sol = solve_ivp(
                controlled_vector_field, 
                (t_current, t_next), 
                current_state, 
                method='RK45'
            )
            
            current_state = sol.y[:, -1]
            trajectory[idx + 1] = current_state
            
        # Replicamos el último frame temporal para cerrar los arrays dimensionalmente
        psi_c_telemetry[-1] = psi_c_telemetry[-2]
        control_telemetry[-1] = control_telemetry[-2]
            
        return t_eval, trajectory, psi_c_telemetry, control_telemetry