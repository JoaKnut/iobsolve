r"""
Suite de Pruebas: Espacios Topológicos y Partición Espacial (core.space, core.partition).

Verifica las abstracciones de EuclideanManifold, DiscreteTopology y el motor
IOB-QuadTree (SpatialPartitionEngine), incluyendo validaciones dimensionales,
subdivisión de hipercubos y extracción de hojas singulares.
"""

import pytest
import torch

from iobsolve.core.space import (
    EuclideanManifold,
    DiscreteTopology,
    TopologicalInconsistencyError,
)
from iobsolve.core.partition import SpatialPartitionEngine, PartitionNode


# =============================================================================
# EUCLIDEAN MANIFOLD
# =============================================================================

class TestEuclideanManifold:
    """Pruebas para la variedad euclidiana continua."""

    def test_creacion_valida_1d(self):
        """Crea una variedad 1D sin errores."""
        m = EuclideanManifold(shape=(50,), grid_spacing=0.1)
        assert m.dimension == 1
        assert m.shape == (50,)
        assert m.grid_spacing == 0.1

    def test_creacion_valida_3d(self):
        """Crea una variedad 3D sin errores."""
        m = EuclideanManifold(shape=(10, 20, 30), grid_spacing=0.5)
        assert m.dimension == 3

    def test_medida_lebesgue_correcta(self):
        """
        La medida de Lebesgue mu(Omega) debe ser el producto de (size * spacing)
        para cada dimensión.
        """
        m = EuclideanManifold(shape=(10, 20), grid_spacing=0.5)
        expected = (10 * 0.5) * (20 * 0.5)  # 5.0 * 10.0 = 50.0
        assert abs(m.measure - expected) < 1e-10

    def test_dimension_cero_lanza_error(self):
        """Una dimensión de tamaño 0 debe lanzar TopologicalInconsistencyError."""
        with pytest.raises(TopologicalInconsistencyError):
            EuclideanManifold(shape=(0, 10))

    def test_grid_spacing_negativo_lanza_error(self):
        """Un grid_spacing <= 0 debe lanzar TopologicalInconsistencyError."""
        with pytest.raises(TopologicalInconsistencyError):
            EuclideanManifold(shape=(10,), grid_spacing=-1.0)
        with pytest.raises(TopologicalInconsistencyError):
            EuclideanManifold(shape=(10,), grid_spacing=0.0)

    def test_validate_field_campo_correcto(self):
        """Un campo con la forma correcta debe validarse sin excepción."""
        m = EuclideanManifold(shape=(8, 8))
        field = torch.randn(8, 8, dtype=torch.float64)
        m.validate_field(field)  # No debe lanzar

    def test_validate_field_campo_incorrecto_lanza_error(self):
        """Un campo con forma incompatible debe lanzar TopologicalInconsistencyError."""
        m = EuclideanManifold(shape=(8, 8))
        field = torch.randn(10, 8, dtype=torch.float64)  # primera dim incorrecta
        with pytest.raises(TopologicalInconsistencyError):
            m.validate_field(field)


# =============================================================================
# DISCRETE TOPOLOGY
# =============================================================================

class TestDiscreteTopology:
    """Pruebas para la topología discreta (grafo)."""

    def test_creacion_valida(self):
        """Crea una topología válida con matriz cuadrada."""
        adj = torch.eye(5, dtype=torch.float64)
        topo = DiscreteTopology(adjacency=adj)
        assert topo.dimension == 1
        assert topo.measure == 5.0

    def test_matriz_no_cuadrada_lanza_error(self):
        """Una matriz no cuadrada debe lanzar TopologicalInconsistencyError."""
        adj = torch.zeros((3, 5), dtype=torch.float64)
        with pytest.raises(TopologicalInconsistencyError):
            DiscreteTopology(adjacency=adj)

    def test_validate_nodal_state_correcto(self):
        """Un vector de estado con la cardinalidad correcta no lanza excepción."""
        adj = torch.eye(7, dtype=torch.float64)
        topo = DiscreteTopology(adjacency=adj)
        state = torch.randn(7, dtype=torch.float64)
        topo.validate_nodal_state(state)  # No debe lanzar

    def test_validate_nodal_state_incorrecto_lanza_error(self):
        """Un vector de estado con cardinalidad incorrecta lanza error."""
        adj = torch.eye(7, dtype=torch.float64)
        topo = DiscreteTopology(adjacency=adj)
        bad_state = torch.randn(5, dtype=torch.float64)
        with pytest.raises(TopologicalInconsistencyError):
            topo.validate_nodal_state(bad_state)

    def test_watchdog_grafo_conectado(self):
        """Un grafo con aristas debe pasar el watchdog."""
        adj = torch.ones((5, 5), dtype=torch.float64) - torch.eye(5, dtype=torch.float64)
        topo = DiscreteTopology(adjacency=adj)
        assert topo.check_integrity_watchdog() is True

    def test_watchdog_grafo_vacio(self):
        """Un grafo completamente vacío debe fallar el watchdog."""
        adj = torch.zeros((5, 5), dtype=torch.float64)
        topo = DiscreteTopology(adjacency=adj)
        assert topo.check_integrity_watchdog() is False

    def test_watchdog_grafo_sparse_conectado(self):
        """El watchdog funciona también con matrices dispersas."""
        adj_dense = torch.ones((5, 5), dtype=torch.float64) - torch.eye(5, dtype=torch.float64)
        adj_sparse = adj_dense.to_sparse_coo()
        topo = DiscreteTopology(adjacency=adj_sparse)
        assert topo.check_integrity_watchdog() is True


# =============================================================================
# SPATIAL PARTITION ENGINE (IOB-QUADTREE)
# =============================================================================

class TestSpatialPartitionEngine:
    """Pruebas para el motor de partición espacial IOB-QuadTree."""

    def test_subdivision_hipercubo_2d(self):
        """Un hipercubo 2D debe subdividirse en exactamente 4 sub-hipercubos."""
        engine = SpatialPartitionEngine()
        domain = ((-1.0, 1.0), (-1.0, 1.0))
        children = engine.subdivide_hypercube(domain)
        assert len(children) == 4

    def test_subdivision_hipercubo_3d(self):
        """Un hipercubo 3D debe subdividirse en exactamente 8 sub-hipercubos."""
        engine = SpatialPartitionEngine()
        domain = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        children = engine.subdivide_hypercube(domain)
        assert len(children) == 8

    def test_subdivision_preserva_volumen(self):
        """La suma de los volúmenes de los hijos debe igualar al volumen del padre."""
        engine = SpatialPartitionEngine()
        domain = ((0.0, 2.0), (0.0, 4.0))
        parent_volume = 2.0 * 4.0  # 8.0
        children = engine.subdivide_hypercube(domain)

        total_child_volume = sum(
            (x[1] - x[0]) * (y[1] - y[0])
            for (x, y) in children
        )
        assert abs(total_child_volume - parent_volume) < 1e-10

    def test_criterio_siempre_verdadero_profundidad_maxima(self):
        """
        Con un criterio que siempre retorna True, el árbol debe alcanzar
        exactamente max_depth y producir 2^(n*depth) hojas.
        """
        engine = SpatialPartitionEngine(max_depth=3)
        domain = ((0.0, 1.0), (0.0, 1.0))  # 2D

        def always_true(d):
            return True

        root = engine.isolate_singularities(domain, always_true)
        leaves = engine.extract_singular_manifolds(root)
        # Con depth=3 en 2D: 4^3 = 64 hojas
        assert len(leaves) == 4 ** 3

    def test_criterio_siempre_falso_sin_singularidades(self):
        """
        Con un criterio que siempre retorna False, no debe haber ninguna singularidad.
        """
        engine = SpatialPartitionEngine(max_depth=5)
        domain = ((0.0, 1.0), (0.0, 1.0))

        def always_false(d):
            return False

        root = engine.isolate_singularities(domain, always_false)
        leaves = engine.extract_singular_manifolds(root)
        assert len(leaves) == 0

    def test_nodo_hoja_sin_hijos(self):
        """Un nodo hoja no debe tener hijos (children is None o lista vacía)."""
        engine = SpatialPartitionEngine(max_depth=0)
        domain = ((0.0, 1.0),)

        def always_true(d):
            return True

        root = engine.isolate_singularities(domain, always_true)
        assert root.is_leaf

    def test_tolerancia_volumen_minimo(self):
        """
        La partición debe detenerse cuando el volumen cae por debajo de
        min_volume_tolerance, evitando recursión infinita.
        """
        engine = SpatialPartitionEngine(
            max_depth=100,
            min_volume_tolerance=1e-5,
        )
        # Dominio muy pequeño
        domain = ((0.0, 1e-4),)

        def always_true(d):
            return True

        # No debe exceder la pila de recursión
        root = engine.isolate_singularities(domain, always_true)
        assert root is not None
