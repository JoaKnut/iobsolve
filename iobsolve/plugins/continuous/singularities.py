r"""
Módulo de Singularidades Analíticas (Plugins).

Provee variedades de prueba con singularidades analíticamente conocidas para
instrumentar el Teorema del Flujo de Integridad
(:class:`~iobsolve.continuous.flow_theorem.FlowTheoremLocator`). Diseñado
como banco de referencia (*benchmark*) riguroso para desafiar la localización
espacial del motor topológico continuo.

Cada variedad expuesta tiene raíces exactas documentadas, permitiendo validar
tanto la precisión (distancia del centroide estimado a la raíz real) como la
exhaustividad (número de raíces detectadas).

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra: Aislamiento
       Topológico de Singularidades y Control de Bifurcaciones en Variedades
       Continuas".
"""

import torch
from iobsolve.core.types import ManifoldField


class TranscendentalManifold:
    r"""
    Variedad bidimensional con singularidades hiper-trascendentales.

    El campo vectorial subyacente es la parte real e imaginaria de la función
    holomorfa :math:`f(z) = \sin(z)` evaluada sobre :math:`\mathbb{R}^2`:

    .. math::

        F(x, y) =
        \begin{bmatrix}
            \sin(x)\cosh(y) \\
            \cos(x)\sinh(y)
        \end{bmatrix}

    Sus raíces analíticas (puntos donde :math:`F(x,y) = \mathbf{0}`) se ubican
    estrictamente en los puntos de la forma :math:`(n\pi, 0)` para
    :math:`n \in \mathbb{Z}`.

    Esta variedad es la variedad de prueba predeterminada de la CLI
    (``iobsolve roots``). Con radio de búsqueda :math:`R = 10` y profundidad 8
    deben detectarse las raíces en :math:`\{-3\pi, -2\pi, -\pi, 0, \pi, 2\pi, 3\pi\}`.

    Notes
    -----
    La función :math:`F` tiene discontinuidades de fase severas en el dominio
    espectral (ceros del campo coseno y senoidal entrelazados), lo que pone a
    prueba el filtro de alta frecuencia del IOB-FFT con mayor exigencia que una
    función polinómica.

    Examples
    --------
    >>> import torch
    >>> from iobsolve.plugins.continuous.singularities import TranscendentalManifold
    >>> manifold = TranscendentalManifold()
    >>> # Evaluar en la raíz analítica (0, 0)
    >>> x = torch.zeros(1, 1, 2, dtype=torch.float64)
    >>> result = manifold(0.0, x)
    >>> torch.allclose(result, torch.zeros_like(result), atol=1e-12)
    True
    """

    def __call__(self, _: float, state_tensor: ManifoldField) -> ManifoldField:
        r"""
        Mapea el tensor coordenado al espacio del campo :math:`F(x,y)`.

        Parameters
        ----------
        _ : float
            Parámetro de tiempo (ignorado: el campo es autónomo y no depende
            de :math:`t`).
        state_tensor : ManifoldField
            Malla coordenativa de forma :math:`(*\Omega, 2)` evaluada
            localmente por el motor IOB-QuadTree.

        Returns
        -------
        ManifoldField
            El campo vectorial proyectado con idéntica forma que la entrada:
            :math:`(*\Omega, 2)`.
        """
        x = state_tensor[..., 0]
        y = state_tensor[..., 1]

        u = torch.sin(x) * torch.cosh(y)
        v = torch.cos(x) * torch.sinh(y)

        return torch.stack([u, v], dim=-1)
