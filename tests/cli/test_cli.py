
import json
import torch
import pytest
import subprocess
from pathlib import Path

# Definimos el comando base para invocar el CLI como módulo
CLI_CMD = ["python", "-m", "iobsolve.cli"]

@pytest.fixture
def workspace(tmp_path):
    """Crea un espacio de trabajo temporal para los archivos de entrada y salida."""
    return tmp_path

def test_cli_roots_complex_manifold(workspace):
    """
    CASO 1: ROOTS (Motor Continuo)
    Inyecta un sistema de ecuaciones no lineales personalizado desde un archivo
    Python externo creado en tiempo de ejecución.
    """
    # 1. Crear un sistema dinámico
    manifold_code = """
import torch
class LorenzModificado:
    def __call__(self, t, x):
        # x es un tensor de shape (..., 2)
        u = x[..., 0]
        v = x[..., 1]
        du = torch.sin(u) * torch.exp(-v**2) - 0.5
        dv = torch.cos(v) * u - 0.2
        return torch.stack([du, dv], dim=-1)
"""
    manifold_file = workspace / "custom_lorenz.py"
    manifold_file.write_text(manifold_code)
    out_json = workspace / "roots_out.json"

    # 2. Ejecutar CLI
    cmd = CLI_CMD + [
        "roots",
        "--manifold", f"{manifold_file}:LorenzModificado",
        "--radius", "4.0",
        "--depth", "10",      # Alta profundidad de bisección
        "--res", "32",        # Máxima resolución de FFT local recomendada
        "--tau-spec", "1e-4", # Umbral muy sensible
        "--format", "json",
        "--out-file", str(out_json),
        "-q" # Modo silencioso
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Error en CLI roots: {result.stderr}"
    assert out_json.exists(), "No se exportó la telemetría JSON"
    
    # Validar el contenido
    with open(out_json, "r") as f:
        data = json.load(f)
        assert "topological_charge" in data
        assert data["metrics"]["max_depth"] == 10

def test_cli_shield_massive_ddos(workspace):
    """
    CASO 2: SHIELD (Motor Discreto)
    Simula una topología grande (10,000 nodos) y un ataque de inyección asimétrica,
    forzando al motor a extirpar la anomalía.
    """
    out_json = workspace / "shield_out.json"
    
    cmd = CLI_CMD + [
        "shield",
        "--nodes", "10000",   # Topología masiva
        "--attack",           # Inyectar anomalía
        "--tau", "2.5",       # Umbral quirúrgico estricto
        "--format", "json",
        "--out-file", str(out_json),
        "-q"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Error en CLI shield: {result.stderr}"
    
    with open(out_json, "r") as f:
        data = json.load(f)
        assert data["status"] == "anomalous"
        assert data["surgery_report"]["nodes_excised"] > 0
        assert data["metrics"]["latency_ms"] > 0

def test_cli_audit_latent_collapse(workspace):
    """
    CASO 3: AUDIT (Auditoría IA)
    Genera un tensor pesado simulando embeddings de un LLM o ViT y evalúa
    la varianza y cohesión topológica para detectar colapso modal.
    """
    # 1. Crear un tensor simulado (Batch: 512, Dim: 1536) guardado en .pt
    embeddings_file = workspace / "llm_embeddings.pt"
    # Forzamos colapso modal creando vectores muy similares
    base_vector = torch.randn(1536, dtype=torch.float64)
    collapsed_batch = base_vector.unsqueeze(0).repeat(512, 1)
    torch.save(collapsed_batch, embeddings_file)
    
    out_json = workspace / "audit_out.json"

    # 2. Ejecutar CLI
    cmd = CLI_CMD + [
        "audit",
        "-i", str(embeddings_file),
        "--dim", "1536",
        "--tau", "0.95",
        "--format", "json",
        "--out-file", str(out_json),
        "-q"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Error en CLI audit: {result.stderr}"
    
    with open(out_json, "r") as f:
        data = json.load(f)
        # El sistema DEBE detectar la degeneración porque le inyectamos ruido minúsculo
        assert data["status"] == "collapsing"

def test_cli_dynamics_trajectory(workspace):
    """
    CASO 4: DYNAMICS (Sensor Predictivo)
    Inyecta una serie temporal (trayectoria empírica) de un sistema de 
    alta dimensionalidad.
    """
    trajectory_file = workspace / "climate_trajectory.pt"
    # Simular trayectoria: 1000 instantes de tiempo en 128 dimensiones
    trajectory = torch.randn(1000, 128, dtype=torch.float64)
    torch.save(trajectory, trajectory_file)
    
    out_json = workspace / "dynamics_out.json"

    cmd = CLI_CMD + [
        "dynamics",
        "-i", str(trajectory_file),
        "--dim", "128",
        "--l-metric", "2.5",
        "--format", "json",
        "--out-file", str(out_json),
        "-q"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Error en CLI dynamics: {result.stderr}"
    assert out_json.exists()