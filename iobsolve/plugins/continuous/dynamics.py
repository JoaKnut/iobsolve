r"""
Módulo de Dinámica Continua (Plugins).

Implementa sistemas dinámicos caóticos estándar, como el Atractor de Lorenz-96,
para la validación empírica del Operador de Integridad Continuo. 
Permite simular bifurcaciones estocásticas y evaluar la respuesta del tensor 
de estrés \mathcal{H} ante colapsos de la isometría de fase.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra: Aislamiento 
       Topológico de Singularidades y Control de Bifurcaciones en Variedades Continuas".
"""

import torch
from iobsolve.core.types import ManifoldField

class Lorenz96System:
    r"""
    Atractor de Lorenz-96 (Spatially Extended System).
    
    Modelo meteorológico simplificado utilizado en la literatura para estudiar 
    bifurcaciones caóticas y asimilación de datos. Exhibe un comportamiento 
    altamente dependiente del término de forzamiento externo (F).

    Parameters
    ----------
    forcing_constant : float, default=8.0
        Término de forzamiento (F). Para F = 8.0, el sistema exhibe caos altamente 
        desarrollado. Para F < 4.0, el sistema decae hacia un atractor periódico o puntual.
    """

    def __init__(self, forcing_constant: float = 8.0):
        self.forcing = forcing_constant

    def __call__(self, t: float, state_tensor: ManifoldField) -> ManifoldField:
        r"""
        Evalúa el campo vectorial de fase d\mathbf{x}/dt garantizando la 
        invariancia traslacional mediante diferencias finitas periódicas.
        
        Notes
        -----
        La ecuación gobernante del sistema acoplado es:
        .. math:: \frac{dx_i}{dt} = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
        
        Implementado íntegramente con tensores de PyTorch para preservar 
        el grafo de diferenciación automática (Autograd) y permitir aceleración 
        nativa por GPU.

        Parameters
        ----------
        t : float
            Instante de evaluación dinámico.
        state_tensor : ManifoldField
            Tensor conteniendo el estado espacial iterado \mathbf{x}(t).

        Returns
        -------
        ManifoldField
            Campo vectorial representando el flujo derivativo.
            
        Complexity
        ----------
        \mathcal{O}(N) escalar, eludiendo bucles en Python gracias a la vectorización 
        cíclica (\texttt{torch.roll}).
        """
        # torch.roll opera nativamente sobre el device del state_tensor
        x_plus_1 = torch.roll(state_tensor, shifts=-1, dims=-1)
        x_minus_1 = torch.roll(state_tensor, shifts=1, dims=-1)
        x_minus_2 = torch.roll(state_tensor, shifts=2, dims=-1)
        
        advection = (x_plus_1 - x_minus_2) * x_minus_1
        dissipation = state_tensor
        
        return advection - dissipation + self.forcing