r"""
Suite de Pruebas: Operador de Bisagra Continuo y Mapeador Espectral.

Verifica la correctitud del campo de estrés H(x), la localización de singularidades,
el criterio TVI (Teorema del Valor Intermedio) y el análisis espectral (IOB-FFT)
con ventana de Tukey.
"""

import math
import pytest
import torch

from iobsolve.core.space import EuclideanManifold
from iobsolve.continuous.hinge import ContinuousIntegrityOperator
from iobsolve.core.spectral import SpectralIntegrityMapper


# =============================================================================
# CONTINUOUS INTEGRITY OPERATOR
# =============================================================================

class TestContinuousIntegrityOperator:
    """Pruebas para el Operador de Bisagra Continuo."""

    @pytest.fixture
    def manifold_2d(self):
        return EuclideanManifold(shape=(32, 32), grid_spacing=0.1)

    def test_campo_plano_estres_cero(self, manifold_2d):
        """
        Verdad Base: Un campo lineal (curvatura nula) debe producir estrés H = 0.
        """
        op = ContinuousIntegrityOperator()
        # Construir campo lineal f(x,y) = 2x + 3y
        x = torch.linspace(0, 3.1, 32, dtype=torch.float64)
        y = torch.linspace(0, 3.1, 32, dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        field = 2.0 * X + 3.0 * Y

        stress = op.compute_stress(field, manifold=manifold_2d, normalize=False)
        # El interior debe ser aproximadamente cero
        interior = stress[2:-2, 2:-2]
        assert interior.max().item() < 1e-6, (
            f"El estrés de un campo lineal debe ser ~0, max={interior.max().item()}"
        )

    def test_campo_cuadratico_estres_positivo(self, manifold_2d):
        """
        Un campo cuadrático f(x,y) = x^2 + y^2 tiene curvatura no nula (Laplaciano = 4).
        El tensor de estrés debe ser positivo en el interior.
        """
        op = ContinuousIntegrityOperator()
        x = torch.linspace(0, 3.1, 32, dtype=torch.float64)
        y = torch.linspace(0, 3.1, 32, dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        field = X ** 2 + Y ** 2

        stress = op.compute_stress(field, manifold=manifold_2d, normalize=True)
        interior = stress[2:-2, 2:-2]
        # El interior debe ser uniforme (curvatura constante) y positivo
        assert interior.min().item() > 0.0, "El estrés de x^2+y^2 debe ser positivo."

    def test_normalizacion_en_0_1(self, manifold_2d):
        """Con normalize=True, el estrés debe estar en [0, 1]."""
        op = ContinuousIntegrityOperator()
        field = torch.randn(32, 32, dtype=torch.float64)
        stress = op.compute_stress(field, manifold=manifold_2d, normalize=True)
        assert stress.min().item() >= -1e-12
        assert stress.max().item() <= 1.0 + 1e-12

    def test_locate_singularities_encuentra_picos(self, manifold_2d):
        """
        locate_singularities debe identificar las regiones de estrés máximo.
        """
        op = ContinuousIntegrityOperator()
        # Estrés artificial con un pico en (15, 15)
        stress = torch.zeros(32, 32, dtype=torch.float64)
        stress[15, 15] = 0.9
        stress[8, 8] = 0.3

        singular = op.locate_singularities(stress, threshold=0.5)
        rows, cols = singular

        assert (15 in rows.tolist() and 15 in cols.tolist()), (
            "El pico en (15,15) debe ser detectado como singular."
        )

    def test_dimensionalidad_preservada(self):
        """El tensor de estrés debe tener la misma forma que el campo de entrada."""
        op = ContinuousIntegrityOperator()
        for shape in [(20,), (16, 16)]:
            manifold = EuclideanManifold(shape=shape, grid_spacing=0.1)
            field = torch.randn(*shape, dtype=torch.float64)
            stress = op.compute_stress(field, manifold=manifold, normalize=True)
            assert stress.shape == field.shape


# =============================================================================
# SPECTRAL INTEGRITY MAPPER (IOB-FFT)
# =============================================================================

class TestSpectralIntegrityMapper:
    """Pruebas del mapeador espectral de alta frecuencia."""

    def test_campo_constante_estres_espectral_bajo(self):
        """
        Verdad Base: Un campo constante tiene casi toda su energía en frecuencia cero.
        La ventana de Tukey introduce pequeñas transiciones en el borde, por lo que
        Q_spec es pequeño pero no estrictamente cero. Debe ser << 0.2.
        """
        field = torch.ones(32, 32, 2, dtype=torch.float64)
        stress = SpectralIntegrityMapper.compute_spectral_stress(field, high_freq_threshold=0.5)
        assert stress.item() < 0.2, (
            f"El estrés espectral de un campo constante debe ser pequeño, obtenido {stress.item()}"
        )

    def test_campo_ruido_blanco_alto_estres(self):
        """
        Ruido blanco aleatorio tiene energía distribuida uniformemente en el espectro.
        Q_spec debe ser > 0.5 (más de la mitad de la energía en alta frecuencia).
        """
        torch.manual_seed(0)
        field = torch.randn(32, 32, 2, dtype=torch.float64)
        # Con threshold=0.5, la mitad del espectro radial es "alta frecuencia"
        stress = SpectralIntegrityMapper.compute_spectral_stress(field, high_freq_threshold=0.5)
        assert stress.item() > 0.5, (
            f"El ruido blanco debe tener Q_spec > 0.5, obtenido {stress.item():.4f}"
        )

    def test_q_spec_en_0_1(self):
        """Q_spec debe estar siempre en [0, 1]."""
        torch.manual_seed(1)
        for _ in range(5):
            field = torch.randn(16, 16, 3, dtype=torch.float64)
            q = SpectralIntegrityMapper.compute_spectral_stress(field)
            assert 0.0 <= q.item() <= 1.0, f"Q_spec fuera de [0,1]: {q.item()}"

    def test_criterio_tvi_cambio_signo(self):
        """
        El criterio de cambio de signo debe retornar True cuando todos
        los componentes cruzan el cero.
        """
        # Campo que cruza el cero en cada componente
        field = torch.linspace(-1.0, 1.0, 8, dtype=torch.float64)
        field_2c = torch.stack([field, -field], dim=-1)  # ambos cruzan cero
        assert SpectralIntegrityMapper._sign_change_criterion(field_2c) is True

    def test_criterio_tvi_sin_cambio_signo(self):
        """
        El criterio de cambio de signo debe retornar False si algún componente
        no cruza el cero.
        """
        field = torch.linspace(1.0, 5.0, 8, dtype=torch.float64)  # siempre > 0
        field_2c = torch.stack([field, -field], dim=-1)
        # componente 0 siempre positivo -> no cruza 0
        assert SpectralIntegrityMapper._sign_change_criterion(field_2c) is False

    def test_evaluate_integrity_criterion_con_tvi(self):
        """
        evaluate_integrity_criterion con require_sign_change=True debe rechazar
        dominios donde no hay cambio de signo, sin ejecutar FFT.
        """
        # Campo siempre positivo (sin cambio de signo)
        field = torch.full((16, 16, 2), 2.0, dtype=torch.float64)
        result = SpectralIntegrityMapper.evaluate_integrity_criterion(
            field, critical_stress=1e-3, require_sign_change=True
        )
        assert result is False

    def test_ventana_tukey_misma_forma(self):
        """La ventana de Tukey no debe cambiar la forma del tensor."""
        field = torch.randn(20, 20, 3, dtype=torch.float64)
        windowed = SpectralIntegrityMapper._apply_tukey_window(field, taper_fraction=0.2)
        assert windowed.shape == field.shape

    def test_ventana_tukey_reduce_bordes(self):
        """La ventana de Tukey debe atenuar los bordes sin tocar el centro."""
        N = 40
        field = torch.ones(N, N, 1, dtype=torch.float64)
        windowed = SpectralIntegrityMapper._apply_tukey_window(field, taper_fraction=0.25)

        # Los bordes deben ser menores que el interior
        border_val = windowed[0, 0, 0].item()
        center_val = windowed[N // 2, N // 2, 0].item()
        assert border_val < center_val, "La ventana debe atenuar los bordes."
        assert center_val == pytest.approx(1.0, abs=1e-10), "El centro debe ser 1."
