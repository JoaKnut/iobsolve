r"""
Suite de Pruebas: Operadores Laplacianos (core.laplacian).

Cubre la correctitud matemática del Laplaciano continuo por diferencias finitas
centrales y los Laplacianos discretos combinatorio y normalizado, incluyendo
casos límite como matrices dispersas, grafos desconectados y campos constantes.
"""

import math
import pytest
import torch

from iobsolve.core.laplacian import ContinuousLaplacian, DiscreteLaplacian


# =============================================================================
# LAPLACIANO CONTINUO
# =============================================================================

class TestContinuousLaplacian:
    """Pruebas de correctitud para el Laplaciano continuo de diferencias finitas."""

    def test_campo_constante_laplaciano_cero(self):
        """
        Verdad Base: El Laplaciano de un campo constante es exactamente 0.
        nabla^2 C = 0 para cualquier constante C.
        """
        field = torch.full((20, 20), 5.0, dtype=torch.float64)
        lap = ContinuousLaplacian.compute(field, grid_spacing=1.0)
        assert torch.allclose(lap, torch.zeros_like(lap), atol=1e-10), (
            "El Laplaciano de un campo constante debe ser cero."
        )

    def test_campo_lineal_laplaciano_cero(self):
        """
        Verdad Base: El Laplaciano de un campo lineal f(x,y) = ax + by es 0.
        La segunda derivada de una función lineal es 0.
        """
        x = torch.linspace(0.0, 1.0, steps=30, dtype=torch.float64)
        y = torch.linspace(0.0, 1.0, steps=30, dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        field = 3.0 * X + 2.0 * Y  # función lineal

        lap = ContinuousLaplacian.compute(field, grid_spacing=x[1] - x[0])
        # Borde excluido (diferencias finitas de borde tienen truncado mayor)
        interior = lap[2:-2, 2:-2]
        assert torch.allclose(interior, torch.zeros_like(interior), atol=1e-8), (
            "El Laplaciano de un campo lineal debe ser cero en el interior."
        )

    def test_campo_cuadratico_laplaciano_exacto(self):
        """
        Verdad Base: Para f(x) = x^2, nabla^2 f = 2 (exacto en 1D).
        Las diferencias finitas centrales reproducen esto con error O(h^2).
        """
        N = 200
        h = 0.01
        x = torch.linspace(0.0, (N - 1) * h, steps=N, dtype=torch.float64)
        field = x ** 2  # f(x) = x^2, f''(x) = 2

        lap = ContinuousLaplacian.compute(field, grid_spacing=h)
        interior = lap[2:-2]
        expected = torch.full_like(interior, 2.0)

        assert torch.allclose(interior, expected, atol=1e-6), (
            f"El Laplaciano de x^2 debe ser 2.0, max error: {(interior - expected).abs().max().item()}"
        )

    def test_dimensionalidad_preservada(self):
        """El tensor de salida debe tener la misma forma que el de entrada."""
        for shape in [(10,), (8, 8), (6, 6, 6)]:
            field = torch.randn(*shape, dtype=torch.float64)
            lap = ContinuousLaplacian.compute(field, grid_spacing=0.1)
            assert lap.shape == field.shape, (
                f"La forma de salida {lap.shape} difiere de la entrada {field.shape}."
            )

    def test_grid_spacing_escala_correcta(self):
        """
        El Laplaciano debe escalar con 1/h^2: duplicar h cuadruplica el error,
        equivalentemente, dividir h a la mitad cuadruplica la magnitud del Laplaciano
        para un campo dado con unidades implícitas.
        """
        N = 50
        field = torch.sin(torch.linspace(0, math.pi, N, dtype=torch.float64))
        lap_h1 = ContinuousLaplacian.compute(field, grid_spacing=1.0)
        lap_h2 = ContinuousLaplacian.compute(field, grid_spacing=2.0)
        # lap_h2 = lap / (2^2) = lap / 4 => lap_h1 ~ 4 * lap_h2
        ratio = (lap_h1[2:-2].abs() / (lap_h2[2:-2].abs() + 1e-15)).mean()
        assert math.isclose(ratio.item(), 4.0, rel_tol=0.05), (
            f"Escala incorrecta: se esperaba ratio ~4.0, obtenido {ratio.item():.4f}"
        )


# =============================================================================
# LAPLACIANO DISCRETO
# =============================================================================

class TestDiscreteLaplacian:
    """Pruebas de correctitud para el Laplaciano discreto (combinatorio y normalizado)."""

    @pytest.fixture
    def grafo_camino_5(self):
        """Grafo de camino simple: 0-1-2-3-4."""
        adj = torch.zeros((5, 5), dtype=torch.float64)
        for i in range(4):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        return adj

    @pytest.fixture
    def grafo_completo_4(self):
        """Grafo completo K_4: todos los nodos conectados entre sí."""
        N = 4
        adj = torch.ones((N, N), dtype=torch.float64) - torch.eye(N, dtype=torch.float64)
        return adj

    def test_laplaciano_combinatorio_semidefinido_positivo(self, grafo_camino_5):
        """
        Verdad Espectral: El Laplaciano combinatorio L = D - A es semidefinido positivo.
        Todos sus autovalores deben ser >= 0.
        """
        L = DiscreteLaplacian.compute_combinatorial(grafo_camino_5)
        eigenvalues = torch.linalg.eigvalsh(L)
        assert (eigenvalues >= -1e-10).all(), (
            f"Autovalores negativos detectados: {eigenvalues.min().item()}"
        )

    def test_laplaciano_combinatorio_vector_uno_nulo(self, grafo_camino_5):
        """
        Verdad Espectral: El vector constante 1 es siempre autovector de L con autovalor 0.
        L * 1 = (D - A) * 1 = d - d = 0.
        """
        L = DiscreteLaplacian.compute_combinatorial(grafo_camino_5)
        ones = torch.ones(5, dtype=torch.float64)
        result = L @ ones
        assert torch.allclose(result, torch.zeros(5, dtype=torch.float64), atol=1e-12), (
            "L * 1 debe ser el vector cero."
        )

    def test_laplaciano_combinatorio_simetrico(self, grafo_completo_4):
        """El Laplaciano combinatorio de un grafo no dirigido debe ser simétrico."""
        L = DiscreteLaplacian.compute_combinatorial(grafo_completo_4)
        assert torch.allclose(L, L.T, atol=1e-12), "L debe ser simétrico."

    def test_laplaciano_normalizado_diagonal_unitaria(self, grafo_camino_5):
        """
        Verdad Espectral: La diagonal del Laplaciano normalizado calL debe ser 1
        para todos los nodos con grado > 0.
        """
        calL = DiscreteLaplacian.compute_normalized(grafo_camino_5)
        diag = torch.diag(calL)
        # Nodos interiores del camino tienen grado 2 -> diagonal = 1
        assert torch.allclose(diag, torch.ones(5, dtype=torch.float64), atol=1e-12), (
            f"La diagonal del Laplaciano normalizado debe ser 1: {diag}"
        )

    def test_laplaciano_normalizado_autovalores_en_0_2(self, grafo_camino_5):
        """
        Verdad Espectral: Los autovalores del Laplaciano normalizado están en [0, 2].
        """
        calL = DiscreteLaplacian.compute_normalized(grafo_camino_5)
        eigenvalues = torch.linalg.eigvalsh(calL)
        assert (eigenvalues >= -1e-10).all(), "Autovalores negativos en calL."
        assert (eigenvalues <= 2.0 + 1e-10).all(), (
            f"Autovalor > 2 detectado: {eigenvalues.max().item()}"
        )

    def test_laplaciano_sparse_vs_denso_equivalentes(self, grafo_camino_5):
        """
        El Laplaciano de la versión sparse y la versión densa de la misma
        matriz de adyacencia deben ser numéricamente iguales.
        """
        adj_dense = grafo_camino_5
        adj_sparse = adj_dense.to_sparse_coo()

        L_dense = DiscreteLaplacian.compute_combinatorial(adj_dense)
        L_sparse = DiscreteLaplacian.compute_combinatorial(adj_sparse)

        assert torch.allclose(L_dense, L_sparse.to_dense(), atol=1e-12), (
            "Las versiones sparse y densa del Laplaciano difieren."
        )

    def test_laplaciano_normalizado_sparse_vs_denso(self, grafo_camino_5):
        """El Laplaciano normalizado sparse y denso deben ser equivalentes."""
        adj_dense = grafo_camino_5
        adj_sparse = adj_dense.to_sparse_coo()

        calL_dense = DiscreteLaplacian.compute_normalized(adj_dense)
        calL_sparse = DiscreteLaplacian.compute_normalized(adj_sparse)

        assert torch.allclose(calL_dense, calL_sparse.to_dense(), atol=1e-10), (
            "Las versiones sparse y densa del Laplaciano normalizado difieren."
        )

    def test_grafo_desconectado_laplaciano_bloque(self):
        """
        Un grafo con dos componentes conexas produce un Laplaciano bloque-diagonal.
        El bloque fuera de la diagonal debe ser exactamente cero.
        """
        # Dos triángulos disjuntos: nodos {0,1,2} y {3,4,5}
        adj = torch.zeros((6, 6), dtype=torch.float64)
        for i, j in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

        L = DiscreteLaplacian.compute_combinatorial(adj)
        # El bloque cruzado (componente 1 x componente 2) debe ser 0
        cross_block = L[:3, 3:]
        assert torch.allclose(cross_block, torch.zeros(3, 3, dtype=torch.float64), atol=1e-12), (
            "El bloque cruzado del Laplaciano de un grafo desconectado no es cero."
        )
