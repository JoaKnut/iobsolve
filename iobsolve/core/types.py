import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

class IOBSystem(ABC):
    """
    Clase base abstracta para sistemas dinámicos o algebraicos evaluables por el framework IOB.

    Define la interfaz obligatoria para computar el estado y sus derivadas 
    direccionales (proyecciones Jacobianas) dentro del espacio de fases.
    """
    
    @property
    @abstractmethod
    def dimensionality(self) -> int:
        """
        Returns
        -------
        int
            La dimensión (N) del espacio de fases.
        """
        pass

    @abstractmethod
    def evaluate(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Evalúa las ecuaciones de estado del sistema.

        Parameters
        ----------
        state : np.ndarray
            Vector de estado actual del sistema.
        t : float, optional
            Variable de tiempo para sistemas no autónomos (default: 0.0).

        Returns
        -------
        np.ndarray
            La salida evaluada (e.g., campo de velocidades dX/dt o función f(z)).
        """
        pass
        
    def evaluate_jacobian(self, state: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
        """
        Computa la derivada direccional del sistema.

        Parameters
        ----------
        state : np.ndarray
            Vector de estado actual.
        direction : np.ndarray
            Vector unitario que representa la dirección de la perturbación.

        Returns
        -------
        np.ndarray or None
            La proyección del Jacobiano exacto. Si retorna None, el operador recurre 
            a diferencias finitas o a la normalización L^4 (modo de caja negra).
        """
        return None