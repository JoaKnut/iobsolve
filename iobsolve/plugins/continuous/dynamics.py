r"""
Módulo de Dinámica Continua (Plugins).

Implementa sistemas dinámicos caóticos estándar — como el Atractor de Lorenz-96
— para la validación empírica del Operador de Integridad Continuo y como fuentes
de referencia para el sensor de detección temprana de crisis (Early Warning).

El módulo expone sistemas dinámicos en forma calleable ``F(t, x)`` compatibles
con la firma :data:`~iobsolve.core.types.DynamicalSystem`, lo que los hace
inyectables directamente en :class:`~iobsolve.continuous.flow_theorem.FlowTheoremLocator`
sin adaptadores adicionales.

References
----------
.. [1] Lorenz, E. N. (1996). "Predictability — A problem partly solved".
       *Seminar on Predictability*, ECMWF, Reading, UK.
.. [2] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra: Aislamiento
       Topológico de Singularidades y Control de Bifurcaciones en Variedades
       Continuas".
"""

import torch
from iobsolve.core.types import ManifoldField


class Lorenz96System:
    r"""
    Atractor de Lorenz-96 (*Spatially Extended System*).

    Modelo meteorológico simplificado utilizado en la literatura para estudiar
    bifurcaciones caóticas, asimilación de datos y detección temprana de crisis.
    Exhibe un comportamiento altamente dependiente del término de forzamiento
    externo :math:`F`:

    - :math:`F < 4`: decaimiento hacia un atractor puntual (estado estacionario).
    - :math:`F \approx 4{-}5`: bifurcaciones periódicas.
    - :math:`F \geq 8`: caos desarrollado (comportamiento meteorológico estándar).

    Parameters
    ----------
    forcing_constant : float, default=8.0
        Término de forzamiento externo :math:`F`. Controla el régimen dinámico
        del sistema. El valor estándar de la literatura es :math:`F = 8.0`.

    Notes
    -----
    La ecuación gobernante del sistema acoplado de :math:`N` variables es:

    .. math::

        \frac{dx_i}{dt} = (x_{i+1} - x_{i-2})\, x_{i-1} - x_i + F

    con condición de contorno periódica: :math:`x_{i+N} \equiv x_i`.

    Esta implementación usa ``torch.roll`` para la periodicidad, evitando
    bucles Python y garantizando vectorización nativa en GPU/CPU.

    **Punto de equilibrio analítico**: Si :math:`x_i = F \;\forall i`, entonces
    :math:`\dot{x}_i = (F - F) \cdot F - F + F = 0`, es decir, el vector
    constante :math:`\mathbf{x} = F \cdot \mathbf{1}` es un punto fijo exacto.
    Este resultado es utilizado como *verdad base* en la suite de pruebas.

    Examples
    --------
    Evaluar el flujo en el punto fijo analítico:

    >>> import torch
    >>> from iobsolve.plugins.continuous.dynamics import Lorenz96System
    >>> lorenz = Lorenz96System(forcing_constant=8.0)
    >>> x_eq = torch.full((40,), 8.0, dtype=torch.float64)  # punto fijo
    >>> flow = lorenz(t=0.0, state_tensor=x_eq)
    >>> torch.allclose(flow, torch.zeros(40, dtype=torch.float64), atol=1e-12)
    True

    Simular un batch de 100 trayectorias simultáneas:

    >>> batch = torch.randn(100, 40, dtype=torch.float64)
    >>> lorenz(0.0, batch).shape
    torch.Size([100, 40])
    """

    def __init__(self, forcing_constant: float = 8.0) -> None:
        self.forcing = forcing_constant

    def __call__(self, t: float, state_tensor: ManifoldField) -> ManifoldField:
        r"""
        Evalúa el campo vectorial de fase :math:`d\mathbf{x}/dt`.

        Garantiza la invariancia traslacional mediante diferencias finitas
        periódicas implementadas con ``torch.roll``, preservando el grafo de
        diferenciación automática (Autograd) y permitiendo aceleración nativa
        en GPU.

        Parameters
        ----------
        t : float
            Instante de evaluación. El sistema de Lorenz-96 es autónomo, por lo
            que este parámetro no afecta al resultado; se mantiene por
            compatibilidad con la firma :data:`~iobsolve.core.types.DynamicalSystem`.
        state_tensor : ManifoldField
            Tensor conteniendo el estado espacial :math:`\mathbf{x}(t)`.
            Puede ser un vector 1-D de forma :math:`(N,)` o un batch
            N-D de forma :math:`(\text{batch}, N)`.

        Returns
        -------
        ManifoldField
            Campo vectorial :math:`\dot{\mathbf{x}}` con idéntica forma que la
            entrada.

        Complexity
        ----------
        :math:`\mathcal{O}(N)` escalar (o :math:`\mathcal{O}(B \cdot N)` en
        batch), eludiendo bucles Python gracias a la vectorización cíclica con
        ``torch.roll``.
        """
        x_plus_1  = torch.roll(state_tensor, shifts=-1, dims=-1)
        x_minus_1 = torch.roll(state_tensor, shifts=1,  dims=-1)
        x_minus_2 = torch.roll(state_tensor, shifts=2,  dims=-1)

        advection  = (x_plus_1 - x_minus_2) * x_minus_1
        dissipation = state_tensor

        return advection - dissipation + self.forcing
