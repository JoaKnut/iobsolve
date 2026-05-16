r"""
Módulo de Singularidades Analíticas (Plugins).

Provee variedades de prueba (funciones hiper-trascendentales y no lineales)
para instrumentar el Teorema del Flujo de Integridad (IOB-QuadTree + FFT).
Diseñado como un benchmark riguroso para desafiar la localización espacial 
del motor topológico.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra: Aislamiento 
       Topológico de Singularidades y Control de Bifurcaciones en Variedades Continuas".
"""

import torch
from iobsolve.core.types import ManifoldField

class TranscendentalManifold:
    r"""
    Variedad bidimensional con singularidades hiper-trascendentales.
    
    Diseñada para inducir discontinuidades de fase severas en el dominio 
    espectral, poniendo a prueba el filtro pasaalto del IOB-FFT.
    """
    
    def __call__(self, _: float, state_tensor: ManifoldField) -> ManifoldField:
        r"""
        Mapea el tensor coordenado al espacio de la función preservando el 
        dispositivo y el historial computacional de PyTorch.

        Notes
        -----
        El sistema vectorial subyacente se define analíticamente como:
        .. math:: F(x, y) = \begin{bmatrix} \sin(x) \cosh(y) \\ \cos(x) \sinh(y) \end{bmatrix}
        
        cuyas raíces analíticas se ubican estrictamente en los puntos de 
        la forma (n\pi, 0).

        Parameters
        ----------
        _ : float
            Parámetro de tiempo (ignorado, ya que el campo es autónomo).
        state_tensor : ManifoldField
            Malla coordenativa \Omega evaluada localmente.

        Returns
        -------
        ManifoldField
            El campo proyectado con idéntica dimensionalidad.
        """
        x = state_tensor[..., 0]
        y = state_tensor[..., 1]
        
        u = torch.sin(x) * torch.cosh(y)
        v = torch.cos(x) * torch.sinh(y)
        
        # Apilamiento en la última dimensión preservando el dtype y device
        return torch.stack([u, v], dim=-1)