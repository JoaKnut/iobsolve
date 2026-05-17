r"""
Módulo del Teorema del Flujo de Integridad.

Acopla el particionamiento espacial recursivo (IOB-QuadTree) con el mapeo
espectral de alta frecuencia (IOB-FFT). Este teorema formaliza la localización
asintóticamente acelerada de singularidades y raíces de equilibrio en campos
vectoriales sin recurrir a la linealización iterativa del Jacobiano.

El motor combina:

- **IOB-QuadTree**: bisección recursiva del espacio de fases hasta ``max_depth``.
- **TVI (Teorema del Valor Intermedio)**: filtro O(N) que descarta subdominios
  sin cambio de signo antes de ejecutar la FFT.
- **IOB-FFT**: confirmación espectral de singularidades interiores.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra: Aislamiento
       Topológico de Singularidades y Control de Bifurcaciones en Variedades Continuas".
"""

import torch
from typing import List, Callable, Union, Tuple

from iobsolve.core.types import DynamicalSystem, ManifoldField
from iobsolve.core.partition import Hypercube, SpatialPartitionEngine
from iobsolve.core.spectral import SpectralIntegrityMapper


class FlowTheoremLocator:
    r"""
    Instrumentación del Teorema del Flujo de Integridad.

    Acorrala las singularidades mediante la bisección recursiva del espacio de
    fases :math:`\Omega \subset \mathbb{R}^n`. Elude el cálculo polinomial del
    Jacobiano :math:`\mathcal{O}(N^3)` utilizando el espectro de alta frecuencia
    :math:`\mathcal{Q}_{\text{spec}}` como criterio inyectable, precedido por el
    filtro topológico de cambio de signo (TVI).

    Parameters
    ----------
    system_equation : DynamicalSystem
        La función continua :math:`\dot{x} = F(x, t)` a evaluar.
    grid_resolution : int, default=16
        Resolución de muestreo de la malla local (:math:`\Delta x`).
        Se recomienda mantener :math:`\leq 32` para que la FFT local opere
        en :math:`\mathcal{O}(1)` amortizado.
    spectral_threshold : float, default=1e-3
        Umbral crítico de estrés espectral (:math:`\tau_c`). Valores más
        bajos incrementan la sensibilidad y el tiempo de cómputo.
    require_sign_change : bool, default=True
        Si True, aplica el filtro TVI (cambio de signo en todos los
        componentes) antes del análisis espectral. Recomendado para
        localización de raíces de campos vectoriales.
        Desactivar solo para singularidades no asociadas a raíces.
    device : str | torch.device, default="cpu"
        Plataforma de aceleración tensorial ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        system_equation: DynamicalSystem,
        grid_resolution: int = 16,
        spectral_threshold: float = 1e-3,
        require_sign_change: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        self.system_equation = system_equation
        self.grid_resolution = min(grid_resolution, 32)  # cap de seguridad
        self.spectral_threshold = spectral_threshold
        self.require_sign_change = require_sign_change
        self.device = torch.device(device)

    def _sample_subdomain(self, domain: Hypercube, time_t: float = 0.0) -> ManifoldField:
        r"""
        Proyecta la ecuación del sistema sobre una malla discreta circunscrita
        al sub-hipercubo :math:`\Omega_i`.

        La instanciación tensorial se ejecuta directamente en el dispositivo
        configurado, mitigando cuellos de botella por transferencias H2D/D2H.
        """
        axes = [
            torch.linspace(
                float(d_min), float(d_max),
                steps=self.grid_resolution,
                dtype=torch.float64,
                device=self.device,
            )
            for d_min, d_max in domain
        ]
        grids = torch.meshgrid(*axes, indexing="ij")
        state_tensor = torch.stack(grids, dim=-1)
        return self.system_equation(time_t, state_tensor)

    def _integrity_criterion_closure(self, time_t: float) -> Callable[[Hypercube], bool]:
        r"""
        Cierre topológico inyectable para el particionador espacial.

        Encapsula el contexto temporal ``t`` para evaluar el criterio compuesto
        (TVI + FFT) en cada subdominio candidato.
        """
        def criterion(domain: Hypercube) -> bool:
            local_field = self._sample_subdomain(domain, time_t)
            return SpectralIntegrityMapper.evaluate_integrity_criterion(
                boundary_field=local_field,
                critical_stress=self.spectral_threshold,
                require_sign_change=self.require_sign_change,
            )
        return criterion

    def locate_roots(
        self,
        initial_domain: Hypercube,
        time_t: float = 0.0,
        max_depth: int = 10,
    ) -> List[Hypercube]:
        r"""
        Ejecuta la localización exhaustiva de singularidades topológicas sobre
        el hipercubo global.

        Parameters
        ----------
        initial_domain : Hypercube
            El subespacio :math:`\Omega` inicial a explorar.
        time_t : float, default=0.0
            Instante de evaluación del flujo diferencial.
        max_depth : int, default=10
            Límite de bisección espacial para contener la pila recursiva.
            Con ``n`` dimensiones, el árbol tiene como mucho :math:`2^{n \cdot d}`
            hojas; el filtro TVI descarta la mayoría antes de la FFT.

        Returns
        -------
        List[Hypercube]
            Subdominios terminales disjuntos :math:`\{\Omega_k\}` que encapsulan
            analíticamente los atractores, fuentes o sumideros del campo.
        """
        engine = SpatialPartitionEngine(max_depth=max_depth)
        criterion = self._integrity_criterion_closure(time_t)
        root_node = engine.isolate_singularities(
            current_domain=initial_domain,
            criterion=criterion,
            current_depth=0,
        )
        return engine.extract_singular_manifolds(root_node)

    def centroid(self, domain: Hypercube) -> Tuple[float, ...]:
        r"""
        Retorna el centroide del hipercubo terminal como estimado puntual de la raíz.

        Parameters
        ----------
        domain : Hypercube
            Subdominio terminal :math:`\Omega_k`.

        Returns
        -------
        Tuple[float, ...]
            Coordenadas del centro geométrico en cada dimensión.
        """
        return tuple((lo + hi) / 2.0 for lo, hi in domain)

    def locate_root_centroids(
        self,
        initial_domain: Hypercube,
        time_t: float = 0.0,
        max_depth: int = 10,
    ) -> List[Tuple[float, ...]]:
        r"""
        Atajo conveniente: localiza raíces y retorna sus centroides directamente.

        Parameters
        ----------
        initial_domain : Hypercube
            El subespacio inicial.
        time_t : float, default=0.0
            Instante de evaluación.
        max_depth : int, default=10
            Profundidad máxima de bisección.

        Returns
        -------
        List[Tuple[float, ...]]
            Lista de coordenadas puntuales estimadas de cada singularidad.
        """
        domains = self.locate_roots(initial_domain, time_t, max_depth)
        return [self.centroid(d) for d in domains]
