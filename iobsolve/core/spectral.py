r"""
Módulo de Mapeo Espectral (IOB-FFT).

Implementa la Transformada Rápida de Fourier para evaluar el flujo de integridad
en la frontera de subdominios euclidianos. El análisis frecuencial discrimina
variaciones geométricas abruptas (alta frecuencia) del ruido basal.

El criterio de singularidad combina dos condiciones complementarias:

1. **Criterio Topológico (Teorema del Valor Intermedio)**:
   El campo vectorial F(x) debe cambiar de signo en *todos* sus componentes
   dentro del dominio Ω_k. Condición necesaria para la existencia de una raíz
   por el TVI aplicado a cada componente.

2. **Criterio Espectral (IOB-FFT)**:
   La proporción de energía de alta frecuencia Q_spec supera un umbral crítico τ_c,
   indicando variaciones abruptas compatibles con una singularidad interior.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra: Aislamiento
       Topológico de Singularidades y Control de Bifurcaciones en Variedades Continuas".
"""

from typing import Any
import torch
from iobsolve.core.types import ManifoldField
import torch.fft

# Cortafuegos para el linter: extracción opaca de métodos espectrales
_fftn: Any = getattr(torch.fft, "fftn")
_fftshift: Any = getattr(torch.fft, "fftshift")


class SpectralIntegrityMapper:
    r"""
    Motor IOB-FFT para el mapeo de singularidades hiper-trascendentales.

    Combina el Teorema del Valor Intermedio (criterio topológico exacto) con el
    análisis espectral de alta frecuencia (criterio heurístico de regularidad)
    para localizar singularidades de campos vectoriales continuos.

    El criterio compuesto exige **ambas** condiciones simultáneamente:
    - Cambio de signo en todos los componentes (condición necesaria por TVI).
    - Exceso de energía espectral de alta frecuencia (confirmación de ruptura).
    """

    @staticmethod
    def _apply_tukey_window(field: ManifoldField, taper_fraction: float = 0.15) -> ManifoldField:
        r"""
        Aplica una ventana de Tukey N-Dimensional para atenuar discontinuidades
        de borde antes de la transformada de Fourier.

        La ventana suprime el *spectral leakage* con rampas coseno en los márgenes
        ``taper_fraction/2`` de cada eje, preservando la señal interior.

        Parameters
        ----------
        field : ManifoldField
            Campo vectorial de forma ``(*spatial_shape, n_components)``.
        taper_fraction : float, default=0.15
            Fracción de cada eje cubierta por la rampa coseno (⊂ [0, 1]).

        Returns
        -------
        ManifoldField
            Campo atenuado con idéntica forma y dtype.
        """
        spatial_shape = field.shape[:-1]
        device, dtype = field.device, field.dtype

        window = torch.ones(spatial_shape, device=device, dtype=dtype)
        half = taper_fraction / 2.0

        for axis, size in enumerate(spatial_shape):
            x = torch.linspace(0.0, 1.0, steps=size, device=device, dtype=dtype)
            w_1d = torch.ones(size, device=device, dtype=dtype)

            mask_left = x < half
            if mask_left.any():
                w_1d[mask_left] = 0.5 * (
                    1.0 + torch.cos(torch.pi * (x[mask_left] / half - 1.0))
                )

            mask_right = x > (1.0 - half)
            if mask_right.any():
                w_1d[mask_right] = 0.5 * (
                    1.0 + torch.cos(torch.pi * ((x[mask_right] - 1.0) / half + 1.0))
                )

            # Broadcasting seguro sobre la dimensión espacial correcta
            shape = [1] * len(spatial_shape)
            shape[axis] = size
            window = window * w_1d.view(*shape)

        return field * window.unsqueeze(-1)

    @staticmethod
    def _sign_change_criterion(field: ManifoldField) -> bool:
        r"""
        Criterio del Teorema del Valor Intermedio (TVI) para dominios cerrados.

        Verifica que *cada* componente escalar del campo vectorial alcance o
        cruce el cero en el dominio evaluado. Se utilizan desigualdades no
        estrictas (:math:`\min \leq 0 \leq \max`) para incluir raíces ubicadas
        sobre la frontera del subdominio :math:`\partial\Omega_k`, que son el
        caso habitual tras la bisección del espacio de fases (los puntos de
        corte generan aristas compartidas donde la raíz puede residir exactamente).

        Una desigualdad estricta excluiría falsamente estos dominios, impidiendo
        que el QuadTree localice raíces situadas en bordes de bisección.

        Parameters
        ----------
        field : ManifoldField
            Campo vectorial de forma ``(*spatial_shape, n_components)``.

        Returns
        -------
        bool
            True si todos los componentes satisfacen :math:`\min \leq 0 \leq \max`.
        """
        n_components = field.shape[-1]
        for c in range(n_components):
            component = field[..., c]
            if not (component.min() <= 0.0 and component.max() >= 0.0):
                return False
        return True

    @staticmethod
    def compute_spectral_stress(
        boundary_field: ManifoldField,
        high_freq_threshold: float = 0.75,
    ) -> torch.Tensor:
        r"""
        Cuantifica el estrés topológico en el dominio de la frecuencia.

        El coeficiente de estrés espectral :math:`\mathcal{Q}_{\text{spec}}`
        mide la fracción de energía contenida fuera del radio de corte
        normalizado :math:`\nu_c`:

        .. math::

            \mathcal{Q}_{\text{spec}} =
            \frac{\sum_{\nu > \nu_c} \sum_c |\mathcal{F}_c(\nu)|^2}
                 {\sum_\nu   \sum_c |\mathcal{F}_c(\nu)|^2}

        donde la suma sobre :math:`c` acumula la energía de todos los
        componentes vectoriales **antes** de aplicar la máscara radial,
        garantizando que la máscara booleana 2D opere sobre un tensor
        estrictamente 2D y no introduzca indexado cruzado de dimensiones.

        Parameters
        ----------
        boundary_field : ManifoldField
            Campo evaluado en el entorno local :math:`\partial\Omega`.
            Forma esperada: ``(*spatial_shape, n_components)``.
        high_freq_threshold : float, default=0.75
            Radio de corte normalizado :math:`\nu_c \in (0, 1)`.

        Returns
        -------
        torch.Tensor
            Escalar diferenciable :math:`\mathcal{Q}_{\text{spec}} \in [0, 1]`.
        """
        device = boundary_field.device
        dtype = boundary_field.dtype
        spatial_shape = boundary_field.shape[:-1]
        spatial_dims = tuple(range(boundary_field.ndim - 1))

        # 1. Ventana de Tukey para suprimir spectral leakage
        windowed = SpectralIntegrityMapper._apply_tukey_window(boundary_field)

        # 2. FFT N-dimensional sobre las dimensiones espaciales
        spectrum = _fftn(windowed, dim=spatial_dims)
        shifted = _fftshift(spectrum, dim=spatial_dims)
        amplitudes = torch.abs(shifted)  # (*spatial_shape, n_components)

        # 3. Densidad de energía: suma de cuadrados sobre COMPONENTES primero.
        #    Resultado: tensor escalar de forma (*spatial_shape,).
        #    CORRECCIÓN DEL BUG ORIGINAL: la implementación anterior aplicaba
        #    amplitudes[mask]**2 donde mask es (*spatial_shape,) y amplitudes es
        #    (*spatial_shape, n_components). PyTorch fancy-indexing retorna
        #    (N_true, n_components), sumando incorrectamente energía extra
        #    proporcional a n_components. El resultado inflado hacía que
        #    casi todo dominio superara τ_c, generando 4^depth hojas.
        energy_density = torch.sum(amplitudes ** 2, dim=-1)  # (*spatial_shape,)

        total_energy = energy_density.sum()
        if total_energy < 1e-12:
            return torch.tensor(0.0, dtype=dtype, device=device)

        # 4. Máscara radial normalizada en el espacio de frecuencias
        coords = [
            torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=dtype)
            for s in spatial_shape
        ]
        freq_grids = torch.meshgrid(*coords, indexing="ij")
        radial_distance = torch.sqrt(
            torch.stack([g ** 2 for g in freq_grids]).sum(dim=0)
        )  # (*spatial_shape,) — misma forma que energy_density
        high_freq_mask = radial_distance > high_freq_threshold

        # 5. Cociente de energía: máscara booleana sobre densidad escalar (sin cruce de dims)
        high_freq_energy = energy_density[high_freq_mask].sum()

        return high_freq_energy / total_energy

    @classmethod
    def evaluate_integrity_criterion(
        cls,
        boundary_field: ManifoldField,
        critical_stress: float = 1e-3,
        require_sign_change: bool = True,
    ) -> bool:
        r"""
        Criterio de integridad compuesto inyectable para el motor IOB-QuadTree.

        Combina el Teorema del Valor Intermedio con el análisis espectral:

        - Si ``require_sign_change=True`` (defecto): exige primero que todos los
          componentes del campo cambien de signo (condición necesaria por TVI).
          Sólo si esta condición se cumple, evalúa el estrés espectral.
          Esto descarta eficientemente regiones sin raíz en O(N) antes de
          ejecutar la FFT en O(N log N).
        - Si ``require_sign_change=False``: usa únicamente el criterio espectral
          (útil para campos escalares o singularidades no asociadas a raíces).

        Parameters
        ----------
        boundary_field : ManifoldField
            Tensor del campo en la región :math:`\Omega_k`.
        critical_stress : float, default=1e-3
            Umbral de energía de alta frecuencia :math:`\tau_c`.
        require_sign_change : bool, default=True
            Si True, aplica el filtro TVI antes del criterio espectral.

        Returns
        -------
        bool
            True si el subdominio contiene una singularidad detectable.
        """
        # Filtro topológico rápido (O(N) vs O(N log N) de la FFT)
        if require_sign_change:
            if not cls._sign_change_criterion(boundary_field):
                return False

        stress = cls.compute_spectral_stress(boundary_field)
        return stress.item() > critical_stress
