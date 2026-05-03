import logging
from typing import Optional

logger = logging.getLogger(__name__)

class MetricCalibrator:
    """
    Módulo de auto-calibración online para la escala métrica del IOB (L).

    Estima la resolución espacial óptima muestreando la varianza basal 
    (ruido del sistema) durante una fase inicial de estabilización, 
    fijando el estrés geométrico en el objetivo Q_d.

    Parameters
    ----------
    target_baseline_qd : float, optional
        Valor de base deseado para el estrés geométrico bajo operación normal (default: 1.0).
    calibration_steps : int, optional
        Número de pasos de integración para acumular estadísticas basales (default: 50).
    """
    def __init__(self, target_baseline_qd: float = 1.0, calibration_steps: int = 50):
        if target_baseline_qd <= 0 or calibration_steps <= 0:
            raise ValueError("target_baseline_qd y calibration_steps deben ser positivos.")
            
        self.target_qd = target_baseline_qd
        self.calibration_steps = calibration_steps
        
        self._raw_residuals: list[float] = []
        self.is_calibrated: bool = False
        self.optimal_L: float = 1.0  # Fallback conservador

    def push_residual(self, raw_residual_sq: float) -> Optional[float]:
        """
        Ingresa residuos crudos no normalizados para calcular la escala empírica L.

        Parameters
        ----------
        raw_residual_sq : float
            Norma L2 al cuadrado del residuo espacial bilateral.

        Returns
        -------
        float or None
            La métrica L calibrada si la fase de 'warm-up' ha finalizado, de lo contrario None.
        """
        if self.is_calibrated:
            return self.optimal_L
            
        self._raw_residuals.append(raw_residual_sq)
        
        if len(self._raw_residuals) >= self.calibration_steps:
            # Filtro de ceros estrictos para evitar escalados singulares
            valid_residuals = [r for r in self._raw_residuals if r > 1e-12]
            
            mean_res = sum(valid_residuals) / len(valid_residuals) if valid_residuals else 1e-8
            
            # Aislamiento algebraico de L a partir de Q_d = ||Res||^2 / (2 * L^4)
            calibrated_l = (mean_res / (2 * self.target_qd)) ** 0.25
            
            # Límites empíricos para prevenir pérdida de precisión o overflow en integración
            self.optimal_L = max(min(calibrated_l, 10.0), 1e-4)
            self.is_calibrated = True
            
            logger.info(f"[IOB-AUTO] Calibración completada. L_metric fijado en: {self.optimal_L:.5f}")
            return self.optimal_L
            
        return None