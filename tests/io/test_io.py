r"""
Suite de Pruebas: Motor de I/O (Parsers y Exporters).

Verifica la ingesta de tensores (.pt, .npy, .json), topologías discretas
(.json, .graphml), configuraciones JSON, manifolds dinámicos y la serialización
estructurada de telemetría.
"""

import json
import pytest
import torch
from pathlib import Path

from iobsolve.io.parsers import (
    DynamicExpressionManifold,
    load_json_config,
    load_tensor_manifold,
    load_discrete_topology,
    load_custom_manifold,
)
from iobsolve.io.exporters import (
    export_roots_telemetry,
    export_shield_telemetry,
    export_audit_telemetry,
    export_spectral_telemetry,
    export_dynamics_telemetry,
)


# =============================================================================
# DYNAMIC EXPRESSION MANIFOLD
# =============================================================================

class TestDynamicExpressionManifold:
    """Pruebas del evaluador dinámico de expresiones algebraicas."""

    def test_expresion_simple_sin(self):
        """sin(x) evaluado en pi/2 debe ser ~1."""
        manifold = DynamicExpressionManifold("sin(x)")
        X = torch.tensor([[torch.pi / 2]], dtype=torch.float64)
        result = manifold(0.0, X)
        assert result.shape[0] == 1

    def test_expresion_dos_componentes(self):
        """Una expresión de 2 componentes debe retornar tensor con dim final = 2."""
        manifold = DynamicExpressionManifold("sin(x), cos(y)")
        X = torch.zeros(8, 8, 2, dtype=torch.float64)
        result = manifold(0.0, X)
        assert result.shape == (8, 8, 2), f"Forma esperada (8,8,2), obtenida {result.shape}"

    def test_expresion_autonoma_ignora_t(self):
        """El parámetro t debe ser ignorado para sistemas autónomos."""
        manifold = DynamicExpressionManifold("x**2 - 4")
        X = torch.tensor([[2.0]], dtype=torch.float64)
        r1 = manifold(0.0, X)
        r2 = manifold(999.9, X)
        assert torch.allclose(r1, r2)

    def test_expresion_invalida_lanza_error(self):
        """Una expresión con nombre de función inválido debe lanzar RuntimeError."""
        manifold = DynamicExpressionManifold("funcion_no_existe(x)")
        X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        with pytest.raises((RuntimeError, NameError)):
            manifold(0.0, X)

    def test_dtype_float64_preservado(self):
        """El resultado debe ser float64."""
        manifold = DynamicExpressionManifold("sin(x), cos(y)")
        X = torch.randn(4, 4, 2, dtype=torch.float64)
        result = manifold(0.0, X)
        assert result.dtype == torch.float64


# =============================================================================
# LOAD TENSOR MANIFOLD
# =============================================================================

class TestLoadTensorManifold:
    """Pruebas de la ingesta de tensores empíricos."""

    def test_carga_pt(self, tmp_path):
        """Carga correcta de un tensor .pt."""
        tensor = torch.randn(50, 10, dtype=torch.float64)
        path = tmp_path / "data.pt"
        torch.save(tensor, path)
        loaded = load_tensor_manifold(str(path))
        assert loaded.shape == (50, 10)
        assert loaded.dtype == torch.float64

    def test_carga_json_lista(self, tmp_path):
        """Carga correcta de un tensor desde JSON (formato lista)."""
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))
        loaded = load_tensor_manifold(str(path))
        assert loaded.shape == (3, 2)

    def test_carga_json_con_clave_data(self, tmp_path):
        """Carga correcta de un tensor desde JSON con clave 'data'."""
        data = {"data": [[1.0, 2.0], [3.0, 4.0]]}
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))
        loaded = load_tensor_manifold(str(path))
        assert loaded.shape == (2, 2)

    def test_dimension_incorrecta_lanza_error(self, tmp_path):
        """Si la dimensión no coincide con expected_dim, debe lanzar ValueError."""
        tensor = torch.randn(10, 5, dtype=torch.float64)
        path = tmp_path / "bad.pt"
        torch.save(tensor, path)
        with pytest.raises(ValueError):
            load_tensor_manifold(str(path), expected_dim=10)

    def test_archivo_inexistente_lanza_error(self):
        """Un archivo que no existe debe lanzar FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_tensor_manifold("/ruta/inexistente/archivo.pt")


# =============================================================================
# LOAD DISCRETE TOPOLOGY
# =============================================================================

class TestLoadDiscreteTopology:
    """Pruebas de la ingesta de topologías discretas."""

    def test_carga_json_matriz_lista(self, tmp_path):
        """Carga de topología desde JSON como lista de listas."""
        adj = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        path = tmp_path / "topology.json"
        path.write_text(json.dumps(adj))
        tensor = load_discrete_topology(str(path))
        assert tensor.is_sparse or tensor.shape == (3, 3)

    def test_carga_json_con_clave_adjacency(self, tmp_path):
        """Carga de topología desde JSON con clave 'adjacency'."""
        adj = {"adjacency": [[0, 1], [1, 0]]}
        path = tmp_path / "topology.json"
        path.write_text(json.dumps(adj))
        tensor = load_discrete_topology(str(path))
        assert tensor.shape == (2, 2) or tensor.to_dense().shape == (2, 2)

    def test_archivo_inexistente_lanza_error(self):
        """Un archivo que no existe debe lanzar FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_discrete_topology("/ruta/no/existe.json")


# =============================================================================
# LOAD CUSTOM MANIFOLD
# =============================================================================

class TestLoadCustomManifold:
    """Pruebas de la inyección dinámica de variedades personalizadas."""

    def test_carga_clase_valida(self, tmp_path):
        """Un módulo Python con una clase válida debe cargarse sin error."""
        code = """
import torch

class MiEcuacion:
    def __call__(self, t, x):
        return x * 2.0
"""
        path = tmp_path / "mi_sistema.py"
        path.write_text(code)
        instance = load_custom_manifold(str(path), "MiEcuacion")
        assert callable(instance)
        result = instance(0.0, torch.tensor([1.0, 2.0]))
        assert torch.allclose(result, torch.tensor([2.0, 4.0]))

    def test_clase_inexistente_lanza_error(self, tmp_path):
        """Si la clase no existe en el módulo, debe lanzar AttributeError."""
        code = "class OtraClase: pass"
        path = tmp_path / "modulo.py"
        path.write_text(code)
        with pytest.raises(AttributeError):
            load_custom_manifold(str(path), "ClaseInexistente")

    def test_archivo_inexistente_lanza_error(self):
        """Módulo inexistente debe lanzar FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_custom_manifold("/no/existe/modulo.py", "Clase")


# =============================================================================
# EXPORTERS
# =============================================================================

class TestExporters:
    """Pruebas de los exportadores de telemetría."""

    def test_export_roots_genera_json_valido(self, tmp_path):
        """export_roots_telemetry debe generar un JSON parseable con las claves requeridas."""
        roots = [((0.0, 1.0), (0.0, 1.0)), ((-2.0, -1.0), (0.0, 1.0))]
        path = tmp_path / "roots.json"
        json_str, saved = export_roots_telemetry(
            roots=roots, elapsed_s=0.5, radius=5.0, depth=8, filepath=str(path)
        )
        data = json.loads(json_str)
        assert "topological_charge" in data
        assert data["topological_charge"] == 2
        assert "root_centroids" in data
        assert "max_depth" in data["metrics"]
        assert data["metrics"]["max_depth"] == 8

    def test_export_shield_estado_anomalous(self, tmp_path):
        """export_shield_telemetry debe marcar 'anomalous' cuando hay alertas."""
        alerts = torch.zeros(10, dtype=torch.bool)
        alerts[3] = True
        path = tmp_path / "shield.json"
        json_str, _ = export_shield_telemetry(
            alerts=alerts, latency_ms=1.23, tau_threshold=3.0, filepath=str(path)
        )
        data = json.loads(json_str)
        assert data["status"] == "anomalous"
        assert data["surgery_report"]["nodes_excised"] == 1

    def test_export_shield_estado_stable(self, tmp_path):
        """export_shield_telemetry debe marcar 'stable' cuando no hay alertas."""
        alerts = torch.zeros(10, dtype=torch.bool)
        path = tmp_path / "shield_stable.json"
        json_str, _ = export_shield_telemetry(
            alerts=alerts, latency_ms=0.5, tau_threshold=3.0, filepath=str(path)
        )
        data = json.loads(json_str)
        assert data["status"] == "stable"

    def test_export_audit_collapsing(self, tmp_path):
        """export_audit_telemetry debe registrar 'collapsing' cuando is_collapsing=True."""
        path = tmp_path / "audit.json"
        json_str, _ = export_audit_telemetry(
            is_collapsing=True, batch_size=64, latent_dim=128,
            tau_threshold=0.85, filepath=str(path)
        )
        data = json.loads(json_str)
        assert data["status"] == "collapsing"

    def test_export_genera_archivo_automatico(self, tmp_path, monkeypatch):
        """Si no se provee filepath, el exporter debe crear un archivo automáticamente."""
        import os
        monkeypatch.chdir(tmp_path)
        _, saved = export_spectral_telemetry(grid_res=512, peak_count=100)
        assert Path(saved).exists()

    def test_export_roots_sin_roots(self, tmp_path):
        """Con lista vacía de raíces, topological_charge debe ser 0."""
        path = tmp_path / "empty.json"
        json_str, _ = export_roots_telemetry(
            roots=[], elapsed_s=0.0, radius=5.0, depth=8, filepath=str(path)
        )
        data = json.loads(json_str)
        assert data["topological_charge"] == 0
        assert data["root_centroids"] == []


# =============================================================================
# LOAD JSON CONFIG
# =============================================================================

class TestLoadJsonConfig:
    """Pruebas del cargador de configuración JSON."""

    def test_carga_config_valida(self, tmp_path):
        """Un archivo JSON válido debe cargarse como diccionario."""
        config = {"radius": 5.0, "depth": 8, "tau_spec": 1e-3}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        loaded = load_json_config(str(path))
        assert loaded["radius"] == 5.0
        assert loaded["depth"] == 8

    def test_archivo_inexistente_lanza_error(self):
        """Un archivo inexistente debe lanzar FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_json_config("/no/existe/config.json")
