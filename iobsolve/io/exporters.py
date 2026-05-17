"""
Motor de Exportación Estructurada (I/O) para IOB-Solve.

Este módulo provee las rutinas de serialización asintótica para los estados 
topológicos, aislamiento de singularidades y telemetría de red.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

class TopologicalEncoder(json.JSONEncoder):
    """
    Codificador JSON extendido para la serialización estricta de variedades 
    tensoriales de PyTorch y atractores en el plano complejo.
    """
    def default(self, o: Any) -> Any:
        if isinstance(o, torch.Tensor):
            return o.cpu().detach().tolist()
        if isinstance(o, complex):
            return {"real": o.real, "imag": o.imag}
        return super().default(o)

def _commit_telemetry(payload: Dict[str, Any], engine_name: str, filepath: Optional[str]) -> Tuple[str, str]:
    """
    Consolida el payload JSON y asegura siempre la escritura en almacenamiento local.
    Retorna la cadena JSON y la ruta final donde fue guardado.
    """
    json_str = json.dumps(payload, cls=TopologicalEncoder, indent=4)
    
    # Auto-generación de archivo si el usuario no proveyó uno
    if not filepath:
        timestamp = int(time.time())
        safe_name = engine_name.split('.')[-1]
        filepath = f"telemetry_{safe_name}_{timestamp}.json"
        
    path = Path(filepath)
    path.write_text(json_str, encoding="utf-8")
    
    return json_str, str(path)

def export_shield_telemetry(alerts: torch.Tensor, latency_ms: float, tau_threshold: float, filepath: Optional[str] = None) -> Tuple[str, str]:
    """Serializa la telemetría de la Cirugía Topológica D-IOB (Anti-DDoS)."""
    is_compromised = bool(alerts.any())
    payload = {
        "engine": "discrete.network_shield",
        "status": "anomalous" if is_compromised else "stable",
        "metrics": {
            "latency_ms": round(latency_ms, 4),
            "critical_tau": tau_threshold
        },
        "surgery_report": {
            "nodes_excised": int(alerts.sum().item()),
            "anomalous_indices": torch.where(alerts)[0] if is_compromised else []
        }
    }
    return _commit_telemetry(payload, "network_shield", filepath)

def export_audit_telemetry(is_collapsing: bool, batch_size: int, latent_dim: int, tau_threshold: float, filepath: Optional[str] = None) -> Tuple[str, str]:
    """Serializa el diagnóstico de isometría para la detección de colapso modal en IA."""
    payload = {
        "engine": "discrete.mode_collapse",
        "status": "collapsing" if is_collapsing else "isometric",
        "manifold_params": {
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "tau_tolerance": tau_threshold
        },
        "recommendation": "Halt training, gradient collapse detected." if is_collapsing else "Nominal dynamics."
    }
    return _commit_telemetry(payload, "mode_collapse", filepath)

def export_roots_telemetry(roots: List[Any], elapsed_s: float, radius: float, depth: int, filepath: Optional[str] = None) -> Tuple[str, str]:
    """Serializa la carga topológica neta localizada vía IOB-QuadTree + TVI + FFT.

    Incluye tanto los subdominios terminales como los centroides estimados
    de cada singularidad (estimado puntual de la raíz).
    """
    # Centroides: punto medio de cada subdominio terminal
    centroids = [
        {"x": (d[0][0] + d[0][1]) / 2.0, "y": (d[1][0] + d[1][1]) / 2.0}
        for d in roots
        if len(d) >= 2
    ]

    payload = {
        "engine": "continuous.singularities",
        "algorithm": "IOB-QuadTree + TVI + FFT",
        "metrics": {
            "elapsed_seconds": round(elapsed_s, 5),
            "search_radius": radius,
            "max_depth": depth,
        },
        "topological_charge": len(roots),
        "root_centroids": centroids,
        "bounding_domains": roots,
    }
    return _commit_telemetry(payload, "singularities", filepath)

def export_spectral_telemetry(grid_res: int, peak_count: int, filepath: Optional[str] = None) -> Tuple[str, str]:
    """Serializa los picos de densidad obtenidos vía Mapeo Espectral Global (IOB-FFT)."""
    payload = {
        "engine": "continuous.spectral",
        "status": "nominal",
        "grid_resolution": grid_res,
        "topological_peaks_detected": peak_count
    }
    return _commit_telemetry(payload, "spectral", filepath)

def export_dynamics_telemetry(dim: int, l_metric: str, critical_t: float, filepath: Optional[str] = None) -> Tuple[str, str]:
    """Serializa las pre-alertas de bifurcación y cizallamiento en el espacio de fases."""
    payload = {
        "engine": "continuous.dynamics",
        "status": "crisis_warning",
        "space_dimension": dim,
        "optical_resolution_l": l_metric,
        "estimated_bifurcation_delta_t": critical_t
    }
    return _commit_telemetry(payload, "dynamics", filepath)