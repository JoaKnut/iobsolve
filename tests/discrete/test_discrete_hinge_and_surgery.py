r"""
Suite de Pruebas: Operador de Bisagra Discreto y Cirugía Topológica.

Verifica el Índice de Estrés Nodal (Q_i), el estimador Z-Score Robusto (MAD)
y las operaciones de cirugía algorítmica (extirpación y poda asimétrica).
"""

import pytest
import torch

from iobsolve.core.space import DiscreteTopology
from iobsolve.discrete.hinge import DiscreteIntegrityOperator
from iobsolve.discrete.estimators import RecursiveTopologicalZScore
from iobsolve.discrete.surgery import TopologicalSurgeon


# =============================================================================
# DISCRETE INTEGRITY OPERATOR (D-IOB)
# =============================================================================

class TestDiscreteIntegrityOperator:
    """Pruebas de correctitud para el Operador de Bisagra Discreto."""

    @pytest.fixture
    def topology_anillo(self):
        """Grafo anillo de 10 nodos (todos con grado 2)."""
        N = 10
        adj = torch.zeros((N, N), dtype=torch.float64)
        for i in range(N):
            adj[i, (i + 1) % N] = 1.0
            adj[i, (i - 1) % N] = 1.0
        return DiscreteTopology(adjacency=adj)

    @pytest.fixture
    def topology_estrella(self):
        """Grafo estrella de 11 nodos (nodo 0 central)."""
        N = 11
        adj = torch.zeros((N, N), dtype=torch.float64)
        adj[0, 1:] = 1.0
        adj[1:, 0] = 1.0
        return DiscreteTopology(adjacency=adj)

    def test_estrés_campo_homogeneo_anillo(self, topology_anillo):
        """
        Verdad Base: En un grafo anillo con estado uniforme x_i = c,
        el Laplaciano L*X = 0 -> el estrés normalizado debe ser cero.
        """
        op = DiscreteIntegrityOperator()
        N = 10
        state = torch.ones(N, dtype=torch.float64)  # campo constante
        stress = op.compute_stress(state, topology=topology_anillo, normalize_output=True)

        assert stress.shape == (N,), "El vector de estrés debe tener un valor por nodo."
        assert torch.allclose(stress, torch.zeros(N, dtype=torch.float64), atol=1e-10), (
            "El estrés nodal de un campo uniforme debe ser exactamente 0."
        )

    def test_estrés_normalizado_en_0_1(self, topology_estrella):
        """El estrés normalizado debe estar en el intervalo [0, 1]."""
        op = DiscreteIntegrityOperator()
        state = torch.randn(11, dtype=torch.float64)
        stress = op.compute_stress(state, topology=topology_estrella, normalize_output=True)
        assert stress.min() >= -1e-12
        assert stress.max() <= 1.0 + 1e-12

    def test_estrés_tensor_multidimensional(self, topology_anillo):
        """El operador debe funcionar con tensores de estado N x m."""
        op = DiscreteIntegrityOperator()
        N, m = 10, 4
        state = torch.randn(N, m, dtype=torch.float64)
        stress = op.compute_stress(state, topology=topology_anillo, normalize_output=False)
        assert stress.shape == (N,), f"Forma esperada ({N},), obtenida {stress.shape}"

    def test_laplaciano_combinatorio_vs_normalizado(self, topology_anillo):
        """
        Ambos tipos de Laplaciano deben producir valores distintos en general,
        pero ambos deben ser cero para campos constantes.
        """
        op = DiscreteIntegrityOperator()
        state_const = torch.ones(10, dtype=torch.float64)

        stress_comb = op.compute_stress(
            state_const, topology=topology_anillo,
            laplacian_type="combinatorial", normalize_output=True
        )
        stress_norm = op.compute_stress(
            state_const, topology=topology_anillo,
            laplacian_type="normalized", normalize_output=True
        )

        assert torch.allclose(stress_comb, torch.zeros(10, dtype=torch.float64), atol=1e-10)
        assert torch.allclose(stress_norm, torch.zeros(10, dtype=torch.float64), atol=1e-10)

    def test_locate_singularities_umbral(self, topology_estrella):
        """
        locate_singularities debe retornar solo los nodos que superan el umbral.
        """
        op = DiscreteIntegrityOperator()
        N = 11
        # Crear stress tensor con un pico artificial
        stress = torch.zeros(N, dtype=torch.float64)
        stress[3] = 0.9
        stress[7] = 0.5

        singular = op.locate_singularities(stress, threshold=0.6)
        indices = singular[0]

        assert 3 in indices.tolist(), "El nodo 3 (0.9 > 0.6) debería ser singular."
        assert 7 not in indices.tolist(), "El nodo 7 (0.5 < 0.6) no debería ser singular."


# =============================================================================
# ROBUST Z-SCORE ESTIMATOR
# =============================================================================

class TestRecursiveTopologicalZScore:
    """Pruebas del estimador Z-Score Robusto con MAD."""

    def test_campo_uniforme_zscore_cero(self):
        """
        Verdad Base: Con un campo completamente homogéneo, el MAD = 0 y
        todos los Z-Scores deben ser 0 (prevención por epsilon).
        """
        estimator = RecursiveTopologicalZScore(num_nodes=10)
        stress = torch.ones(10, dtype=torch.float64) * 5.0
        z = estimator.update_and_compute(stress)
        assert torch.allclose(z, torch.zeros(10, dtype=torch.float64), atol=1e-10)

    def test_outlier_detectado(self):
        """Un outlier extremo debe producir un Z-Score >> 3."""
        estimator = RecursiveTopologicalZScore(num_nodes=20)
        stress = torch.ones(20, dtype=torch.float64)
        stress[5] = 1000.0  # outlier masivo

        z = estimator.update_and_compute(stress)
        assert z[5].item() > 3.0, (
            f"El outlier debería producir Z > 3, obtenido Z = {z[5].item():.2f}"
        )

    def test_decay_factor_actualiza_historial(self):
        """
        Después de la primera llamada, _prev_median debe estar definido.
        En la segunda llamada, el resultado debe ser diferente al de la primera
        cuando el input cambia (el historial influye).
        """
        estimator = RecursiveTopologicalZScore(num_nodes=5, decay_factor=0.5)
        stress = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)

        assert estimator._prev_median is None
        z1 = estimator.update_and_compute(stress)
        assert estimator._prev_median is not None

        # Segunda llamada con input diferente
        stress2 = torch.tensor([10.0, 10.0, 10.0, 10.0, 100.0], dtype=torch.float64)
        z2 = estimator.update_and_compute(stress2)

        # Con decay=0.5 la mediana blended será diferente a la mediana pura
        # Los resultados deben ser distintos
        assert not torch.allclose(z1, z2)

    def test_consistencia_bajo_gaussiano(self):
        """
        Bajo distribución gaussiana, el Z-Score robusto con factor 0.6745
        debe ser consistente con la desviación estándar: median |X - median| ≈ 0.6745 * sigma.
        Por tanto, los Z-Scores deben tener std ~ 1.0.
        """
        torch.manual_seed(42)
        N = 10000
        estimator = RecursiveTopologicalZScore(num_nodes=N, decay_factor=0.0)
        stress = torch.randn(N, dtype=torch.float64)  # sigma=1, media=0
        z = estimator.update_and_compute(stress)
        # El std del Z-Score robusto gaussiano debe ser ~1
        assert abs(z.std().item() - 1.0) < 0.1, (
            f"std del Z-Score robusto debe ser ~1.0, obtenido {z.std().item():.3f}"
        )


# =============================================================================
# TOPOLOGICAL SURGEON
# =============================================================================

class TestTopologicalSurgeon:
    """Pruebas del motor de Cirugía Topológica."""

    @pytest.fixture
    def topology_estrella_10(self):
        """Topología estrella de 10 nodos."""
        N = 10
        adj = torch.zeros((N, N), dtype=torch.float64)
        adj[0, 1:] = 1.0
        adj[1:, 0] = 1.0
        return DiscreteTopology(adjacency=adj)

    def test_isolate_vertices_denso(self, topology_estrella_10):
        """
        Aislar el nodo central (0) debe poner a cero toda su fila y columna.
        """
        surgeon = TopologicalSurgeon(topology=topology_estrella_10)
        indices = torch.tensor([0])
        new_topo = surgeon.isolate_vertices(indices)

        adj = new_topo.adjacency
        assert (adj[0, :] == 0.0).all(), "La fila del nodo extirpado debe ser 0."
        assert (adj[:, 0] == 0.0).all(), "La columna del nodo extirpado debe ser 0."

    def test_isolate_vertices_sparse(self):
        """La extirpación funciona correctamente en matrices dispersas."""
        N = 5
        adj = torch.zeros((N, N), dtype=torch.float64)
        for i in range(N - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        adj_sparse = adj.to_sparse_coo()
        topo = DiscreteTopology(adjacency=adj_sparse)

        surgeon = TopologicalSurgeon(topology=topo)
        indices = torch.tensor([2])  # nodo central del camino
        new_topo = surgeon.isolate_vertices(indices)

        adj_result = new_topo.adjacency.to_dense()
        assert (adj_result[2, :] == 0.0).all()
        assert (adj_result[:, 2] == 0.0).all()
        # Los demás bordes deben estar intactos
        assert adj_result[0, 1] == 1.0
        assert adj_result[3, 4] == 1.0

    def test_isolate_vertices_lista_vacia(self, topology_estrella_10):
        """Aislar un conjunto vacío no debe modificar la topología."""
        surgeon = TopologicalSurgeon(topology=topology_estrella_10)
        original_nnz = (topology_estrella_10.adjacency != 0).sum().item()
        indices = torch.tensor([], dtype=torch.int64)
        new_topo = surgeon.isolate_vertices(indices)
        new_nnz = (new_topo.adjacency != 0).sum().item()
        assert original_nnz == new_nnz

    def test_prune_asymmetric_edges_denso(self):
        """
        La poda asimétrica debe eliminar únicamente las aristas especificadas
        manteniendo el resto de la conectividad intacta.
        """
        N = 6
        adj = torch.zeros((N, N), dtype=torch.float64)
        # Conectar nodo 0 con todos
        for j in range(1, N):
            adj[0, j] = 1.0
            adj[j, 0] = 1.0

        topo = DiscreteTopology(adjacency=adj)
        surgeon = TopologicalSurgeon(topology=topo)

        # Podar solo la arista 0-1 y 0-2
        target = torch.tensor([1, 2])
        new_topo = surgeon.prune_asymmetric_edges(source_index=0, target_indices=target)
        adj_r = new_topo.adjacency

        assert adj_r[0, 1] == 0.0, "La arista 0-1 debe ser podada."
        assert adj_r[0, 2] == 0.0, "La arista 0-2 debe ser podada."
        assert adj_r[0, 3] == 1.0, "La arista 0-3 debe mantenerse intacta."
        assert adj_r[0, 4] == 1.0, "La arista 0-4 debe mantenerse intacta."
