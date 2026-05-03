import numpy as np
from abc import ABC, abstractmethod

class SphereSampler(ABC):
    """
    Estrategia abstracta para la discretización y muestreo de la hiperesfera S^{n-1}.
    """
    
    @abstractmethod
    def sample(self, dimensionality: int, n_samples: int) -> np.ndarray:
        """
        Parameters
        ----------
        dimensionality : int
            La dimensión del espacio de inmersión.
        n_samples : int
            Cantidad de vectores direccionales a generar.

        Returns
        -------
        np.ndarray
            Matriz de forma (n_samples, dimensionality) que contiene vectores unitarios.
        """
        pass

class MonteCarloSampler(SphereSampler):
    """
    Muestreo estocástico isótropo vía distribuciones gaussianas normalizadas.
    
    Adecuado para sistemas de alta dimensionalidad (N >> 3) para mitigar la 
    maldición de la dimensionalidad en la integración numérica.
    """
    def sample(self, dimensionality: int, n_samples: int) -> np.ndarray:
        if dimensionality <= 0 or n_samples <= 0:
            raise ValueError("Dimensionality y n_samples deben ser estrictamente positivos.")
            
        vecs = np.random.randn(n_samples, dimensionality)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        
        # 1e-12 regulariza posibles divisiones por cero en muestras que caen exactamente en el origen
        return vecs / (norms + 1e-12)

class OrthogonalCrossSampler(SphereSampler):
    """
    Muestreo cardinal determinista (Cruz Ortogonal).
    
    Eficiente y suficiente para espacios de baja dimensionalidad y alta simetría.
    """
    def sample(self, dimensionality: int, n_samples: int = 0) -> np.ndarray:
        if dimensionality <= 0:
            raise ValueError("Dimensionality debe ser estrictamente positivo.")
            
        I = np.eye(dimensionality)
        # Retorna 2N vectores: la base canónica estándar y sus contrapartes negativas
        return np.vstack([I, -I])