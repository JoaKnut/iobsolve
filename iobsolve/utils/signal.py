import numpy as np
from scipy.signal import windows

def causal_moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Aplica una convolución causal continua (ventana deslizante asimétrica 
    con soporte estricto en [t-window, t]).
    
    CRÍTICO: A diferencia de los filtros estándar (mode='same'), este método 
    garantiza que no hay "Look-ahead" temporal. El funcional predictivo jamás 
    rompe la flecha causal del tiempo.

    Parameters
    ----------
    data : np.ndarray
        Serie temporal 1D a suavizar.
    window_size : int
        Tamaño de la memoria inercial (en steps discretos).
    """
    if window_size <= 0:
        raise ValueError("El tamaño de la ventana debe ser > 0.")
        
    if len(data) < window_size:
        # TODO: Implementar un logger de warning aquí si se considera pertinente.
        return data  
        
    # Kernel causal uniforme (pesos uniformes solo hacia el pasado)
    kernel = np.ones(window_size) / window_size
    
    # 'full' calcula desde el primer instante; luego truncamos al tamaño original 
    # para evitar el solapamiento futuro.
    smoothed = np.convolve(data, kernel, mode='full')[:len(data)]
    
    # Periodo de Warm-up (Burn-in transitorio): 
    # Durante los primeros 'window_size' pasos, la convolución no tiene 
    # suficientes datos históricos. Ajustamos promediando iterativamente.
    for i in range(window_size - 1):
        smoothed[i] = np.mean(data[:i+1])
        
    return smoothed

def tukey_window_2d(shape: tuple[int, int], alpha: float = 0.95) -> np.ndarray:
    """
    Ventana de apodización espacial bidimensional (Tukey 2D).
    
    Atenúa el error de truncamiento numérico (Spectral Leakage) forzando 
    las fronteras del dominio a cero antes de aplicar el Laplaciano en el 
    dominio de Fourier (IOB-FFT).
    
    Parameters
    ----------
    shape : tuple[int, int]
        Dimensiones (Ny, Nx) de la malla espacial a ventanear.
    alpha : float, optional
        Factor de forma. Un alpha=0.95 (default) genera un aplanamiento central 
        del 95%, dejando solo el 5% de decaimiento cosenoidal en los bordes.
    """
    ny, nx = shape
    
    # Scipy define alpha como la fracción de la ventana que decae (cosine taper).
    # Nuestro alpha define el 'flat top', por lo que invertimos la semántica:
    taper_fraction = max(1.0 - alpha, 0.0)
    
    win_y = windows.tukey(ny, alpha=taper_fraction)
    win_x = windows.tukey(nx, alpha=taper_fraction)
    
    # El producto exterior (outer) de dos ventanas 1D genera la máscara 2D
    return np.outer(win_y, win_x)