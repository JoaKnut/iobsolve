import numpy as np
import scipy.fft as sfft
from typing import Callable, List, Tuple, Optional, cast

from iobsolve.core.types import IOBSystem
from iobsolve.utils.signal import tukey_window_2d
from iobsolve.utils.numeric import curvature_saturation

class ComplexAnalyticSystem(IOBSystem):
    """
    Adaptador universal para funciones de variable compleja f(z).
    
    Convierte una función holomorfa estándar en un sistema evaluable 
    por el framework IOB.

    Parameters
    ----------
    func : Callable[[np.ndarray], np.ndarray]
        Función vectorizada que mapea el plano C -> C.
    """
    def __init__(self, func: Callable[[np.ndarray], np.ndarray]):
        self.func = func

    @property
    def dimensionality(self) -> int:
        return 2  # Plano Euclidiano (Re, Im)

    def evaluate(self, state: np.ndarray) -> np.ndarray:
        return self.func(state)
        
    def evaluate_jacobian(self, state: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
        # Para funciones analíticas, el Jacobiano puede sobreescribirse con f'(z) * dz
        return None

class IOBSpectralMapper:
    """
    Mapeo Espectral Global (IOB-FFT).
    
    Extrae la curvatura del potencial de estrés de fase utilizando 
    el Laplaciano en el dominio de la frecuencia espacial.
    
    Parameters
    ----------
    epsilon : float, optional
        Cota inferior para evitar divergencias logarítmicas (default: 1e-9).
    """
    def __init__(self, epsilon: float = 1e-9):
        self.epsilon = epsilon

    def map_singularities(self, 
                          system: ComplexAnalyticSystem, 
                          Z_grid: np.ndarray, 
                          apply_window: bool = True) -> np.ndarray:
        """
        Evalúa el Funcional de Estrés Espectral Psi(z) sobre una malla compleja.
        
        Note
        ----
        Es computacionalmente intensivo O(N log N). Se recomienda usar mallas 
        con dimensiones potencia de 2 para optimizar la FFT de SciPy.
        """
        f_z = system.evaluate(Z_grid)
        
        # Potencial de estrés logarítmico (transformada de escala)
        phi_z = -np.log(np.abs(f_z) + self.epsilon)
        phi_z = curvature_saturation(phi_z)

        # Apodización para mitigar el Spectral Leakage en los bordes
        if apply_window:
            window = tukey_window_2d(phi_z.shape, alpha=0.95)
            phi_z = phi_z * window

        F_phi = cast(np.ndarray, np.asarray(sfft.fft2(phi_z)))
        
        # Generación de vectores de onda bidimensionales (k_x^2 + k_y^2)
        ny, nx = Z_grid.shape
        kx = sfft.fftfreq(nx)
        ky = sfft.fftfreq(ny)
        KX, KY = np.meshgrid(kx, ky)
        K_sq = KX**2 + KY**2

        # Laplaciano espectral (Teorema de la Derivada de Fourier)
        laplacian_freq = F_phi * K_sq
        psi_z_complex = cast(np.ndarray, np.asarray(sfft.ifft2(laplacian_freq)))
        
        return np.abs(psi_z_complex)

class IOBQuadTreeLocator:
    """
    Algoritmo de Exclusión Topológica mediante Integración de Contorno.
    
    Aísla singularidades/raíces evaluando la carga topológica (N_IOB) 
    encerrada por un perímetro, usando el Principio del Argumento (Cauchy).
    """
    def __init__(self, L_metric: float, max_depth: int = 5):
        self.L = L_metric
        self.max_depth = max_depth

    def _generate_boundary(self, x_min: float, x_max: float, y_min: float, y_max: float, points_per_side: int = 250) -> Tuple[np.ndarray, np.ndarray]:
        """Genera el mallado perimetral y sus vectores normales salientes."""
        x_b = np.linspace(x_min, x_max, points_per_side, endpoint=False)
        y_b = np.full_like(x_b, y_min)
        n_b = np.full(x_b.shape, -1j, dtype=complex)
        
        y_r = np.linspace(y_min, y_max, points_per_side, endpoint=False)
        x_r = np.full_like(y_r, x_max)
        n_r = np.full(y_r.shape, 1 + 0j, dtype=complex)
        
        x_t = np.linspace(x_max, x_min, points_per_side, endpoint=False)
        y_t = np.full_like(x_t, y_max)
        n_t = np.full(x_t.shape, 1j, dtype=complex)
        
        y_l = np.linspace(y_max, y_min, points_per_side, endpoint=False)
        x_l = np.full_like(y_l, x_min)
        n_l = np.full(y_l.shape, -1 + 0j, dtype=complex)
        
        perimeter_z = np.concatenate([x_b + 1j*y_b, x_r + 1j*y_r, x_t + 1j*y_t, x_l + 1j*y_l])
        normals = np.concatenate([n_b, n_r, n_t, n_l])
        
        return perimeter_z, normals

    def _evaluate_perimeter_flow(self, system: ComplexAnalyticSystem, perimeter_z: np.ndarray, normals: np.ndarray) -> float:
        """Computa la integral de línea del estrés geométrico (carga N_IOB)."""
        z_plus = perimeter_z + self.L * normals
        z_minus = perimeter_z - self.L * normals
        
        val_plus = system.evaluate(z_plus)
        val_minus = system.evaluate(z_minus)
        
        integrand = np.log(np.abs(val_plus) + 1e-15) - np.log(np.abs(val_minus) + 1e-15)
        
        # Diferencial de arco discreto (ds)
        ds = np.sum(np.abs(np.diff(np.append(perimeter_z, perimeter_z[0])))) / len(perimeter_z)
        
        N_iob = (1 / (4 * np.pi * self.L)) * np.sum(integrand) * ds
        return float(N_iob)

    def locate(self, system: ComplexAnalyticSystem, domain_bounds: Tuple[float, float, float, float], current_depth: int = 0) -> List[complex]:
        """
        Búsqueda iterativa (Divide and Conquer). Si la carga topológica N_IOB < 1, 
        el dominio está vacío y se poda esa rama del QuadTree.
        """
        x_min, x_max, y_min, y_max = domain_bounds
        perimeter_z, normals = self._generate_boundary(x_min, x_max, y_min, y_max)
        N_iob = self._evaluate_perimeter_flow(system, perimeter_z, normals)
        
        # Poda topológica: Si el flujo es menor a 0.5 (teóricamente entero), no hay singularidad
        if N_iob < 0.5:
            return []
            
        if current_depth >= self.max_depth:
            # Límite de saturación: Devolvemos el baricentro geométrico como semilla
            return [complex((x_min + x_max) / 2, (y_min + y_max) / 2)]
            
        x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
        
        roots = []
        roots.extend(self.locate(system, (x_min, x_mid, y_min, y_mid), current_depth + 1))
        roots.extend(self.locate(system, (x_mid, x_max, y_min, y_mid), current_depth + 1))
        roots.extend(self.locate(system, (x_min, x_mid, y_mid, y_max), current_depth + 1))
        roots.extend(self.locate(system, (x_mid, x_max, y_mid, y_max), current_depth + 1))
        
        return roots

class NewtonRefiner:
    """
    Refinador Damped Newton para semillas topológicas.
    
    Dado que IOB garantiza que la semilla inicial está dentro de la 
    cuenca de atracción (Basin of Attraction), la convergencia es cuadrática 
    y extremadamente robusta frente a funciones rígidas.
    """
    @staticmethod
    def refine(system: ComplexAnalyticSystem, 
               approximate_roots: List[complex], 
               tol: float = 1e-14, 
               max_iter: int = 50) -> List[complex]:
               
        refined_roots = []
        
        for root in approximate_roots:
            z = root
            for _ in range(max_iter):
                f_z = system.evaluate(np.array([z]))[0]
                
                if np.abs(f_z) < tol:
                    break
                    
                df_z = system.evaluate_jacobian(np.array([z]), np.array([1.0 + 0j]))
                
                # Diferencias finitas centrales si el Jacobiano no es analítico
                if df_z is None:
                    h = 1e-8
                    f_plus = system.evaluate(np.array([z + h]))[0]
                    f_minus = system.evaluate(np.array([z - h]))[0]
                    f_prime = (f_plus - f_minus) / (2 * h)
                else:
                    f_prime = df_z[0]
                    
                if np.abs(f_prime) < 1e-15:
                    break # Estancamiento en un valle plano
                    
                dz = f_z / f_prime
                
                # Damped Newton: Cliping del paso diferencial para evitar Overflow
                if np.abs(dz) > 0.5:
                    dz = dz * (0.5 / np.abs(dz))
                    
                z = z - dz
            
            refined_roots.append(z)
            
        return refined_roots