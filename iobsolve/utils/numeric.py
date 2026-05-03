import numpy as np

def atomic_mapping_exp(g_real: np.ndarray, eta: float = 700.0) -> np.ndarray:
    """
    Mapeo Atómico para la estabilización de funciones asintóticas del tipo f(z) = exp(g(z)) - 1.
    
    Proyecta el infinito analítico hacia un observable acotado para evadir 
    el desbordamiento de coma flotante (IEEE 754 Float64 Overflow).
    
    Parameters
    ----------
    g_real : np.ndarray
        La parte real del argumento exponencial (Re(g(z))).
    eta : float, optional
        Límite empírico de seguridad para `np.exp` en float64 (default: 700.0).
        Valores mayores a 709.78 causarán OverflowError.
        
    Returns
    -------
    np.ndarray
        Array evaluado de forma segura y acotado en el régimen asintótico.
    """
    # Usamos una máscara booleana vectorizada para evitar bucles lentos en Python
    safe_mask = g_real < eta
    
    # Pre-alocación de memoria
    phi_asym = np.zeros_like(g_real)
    
    # Régimen asintótico acotado (Re(g(z)) >> eta) -> Crecimiento lineal en lugar de exponencial
    # FIXME: Evaluar si un decaimiento logarítmico es más estable topológicamente que la negación directa.
    phi_asym[~safe_mask] = -g_real[~safe_mask]
    
    # Régimen estándar evaluable 
    # np.expm1 es numéricamente más estable que np.exp(x) - 1 para x pequeños.
    phi_asym[safe_mask] = -np.log(np.abs(np.expm1(g_real[safe_mask])) + 1e-15)
    
    return phi_asym

def curvature_saturation(phi_asym: np.ndarray, alpha_percentile: float = 75.0) -> np.ndarray:
    """
    Saturación de Curvatura mediante la transformación monotónica estricta tanh().
    
    Preserva la resolución topológica de los picos de densidad mientras evita 
    que gradientes explosivos generen "ringing" o ruido de cuantización digital 
    en la FFT posterior.
    
    Parameters
    ----------
    phi_asym : np.ndarray
        El potencial de fase a saturar.
    alpha_percentile : float, optional
        El percentil (0-100) utilizado para normalizar la escala antes del paso 
        por la tangente hiperbólica (default: 75.0, típicamente el Q3).
    """
    # Validaciones defensivas rápidas
    if not (0.0 <= alpha_percentile <= 100.0):
        raise ValueError("El percentil alpha debe estar en el intervalo [0, 100].")
        
    alpha = np.percentile(np.abs(phi_asym), alpha_percentile)
    
    # Prevención de división por cero si el campo es uniformemente plano
    if alpha < 1e-12:
        alpha = 1e-9 
        
    return np.tanh(phi_asym / alpha)