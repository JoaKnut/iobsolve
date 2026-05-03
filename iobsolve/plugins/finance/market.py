import numpy as np
import logging
from typing import Tuple, List, Union

from iobsolve.core.operator import TopologicalCrisisPredictor

logger = logging.getLogger(__name__)

class TopologicalMarketMonitor:
    """
    Monitor de Estabilidad Financiera basado en IOB.
    
    Analiza la matriz estocástica de retornos de un portafolio para detectar 
    colapsos de liquidez (Flash Crashes) midiendo la deformación de la variedad.
    """

    def __init__(self, 
                 n_assets: int, 
                 alert_threshold: float = 8.0, 
                 history_size: int = 20,
                 l_metric: Union[float, str] = "auto"):
        
        if n_assets <= 0:
            raise ValueError("El número de activos debe ser mayor a cero.")
            
        self.n_assets = n_assets
        self.auto_tune = (l_metric == "auto")
        self.l_metric = 1.0 if self.auto_tune else float(l_metric)
        
        if self.auto_tune:
            from iobsolve.core.calibration import MetricCalibrator
            # Calibración inicial empírica (40 ticks es estándar para HFT)
            self.calibrator = MetricCalibrator(target_baseline_qd=1.0, calibration_steps=40)
            
        self.predictor = TopologicalCrisisPredictor(alert_threshold, history_size)
        
        # Buffers internos
        self._price_history: List[np.ndarray] = []

    def push_market_state(self, current_prices: np.ndarray) -> Tuple[bool, float, float]:
        """
        Procesa un nuevo vector de precios del mercado.
        
        Parameters
        ----------
        current_prices : np.ndarray
            Vector 1D de tamaño `n_assets`.
        """
        if current_prices.shape[0] != self.n_assets:
            raise ValueError(f"Dimensión incorrecta. Esperada: {self.n_assets}, Recibida: {current_prices.shape[0]}")
            
        # Clipping defensivo para evitar log(0)
        safe_prices = np.clip(current_prices, 1e-8, None)
        self._price_history.append(safe_prices)
        
        if len(self._price_history) < 3:
            return False, 0.0, 0.0
            
        if len(self._price_history) > 3:
            self._price_history.pop(0)
            
        # Retornos logarítmicos (Velocidad en el espacio de fases)
        r_t = np.log(self._price_history[-1] / self._price_history[-2])
        r_t_minus_1 = np.log(self._price_history[-2] / self._price_history[-3])
        
        # Aceleración discreta
        deformation = r_t - r_t_minus_1
        norm_def_sq = np.sum(deformation ** 2)
        
        # Lógica de auto-tuning
        if self.auto_tune:
            if not self.calibrator.is_calibrated:
                self.calibrator.push_residual(norm_def_sq)
                return False, 0.0, 0.0
            self.l_metric = self.calibrator.optimal_L
            
        q_d = float(norm_def_sq / (2 * (self.l_metric ** 4) + 1e-12))
        is_crisis, psi_c = self.predictor.push_stress_state(q_d)
        
        if is_crisis:
            logger.warning(f"RUPTURA DE VARIEDAD DETECTADA: Psi_c = {psi_c:.2f} | Flash Crash inminente.")
            
        return is_crisis, psi_c, q_d

    def reset(self):
        """Limpia los buffers y reinicia el predictor."""
        self._price_history.clear()
        self.predictor = TopologicalCrisisPredictor(self.predictor.alert_threshold, self.predictor.max_history)