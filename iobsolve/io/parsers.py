"""
Motor de Ingesta Estructurada (I/O) para IOB-Solve.

Este módulo provee las rutinas de parseo para transformar conjuntos de datos empíricos,
topologías discretas (grafos) y sistemas dinámicos definidos por el usuario en 
variedades tensoriales procesables por el Operador de Integridad de Bisagra.
"""

import json
import importlib
import importlib.util
from typing import Any, Optional, Dict, List
from pathlib import Path
import sys

import torch


class DynamicExpressionManifold:
    """
    Variedad matemática continua generada dinámicamente a partir de una 
    expresión algebraica en formato string pasada por el CLI.
    
    Soporta diferenciación automática nativa y evaluación adaptativa de escalares.
    """
    def __init__(self, expression: str) -> None:
        self.expression = expression
        self.expressions: List[str] = [expr.strip() for expr in expression.split(",")]
        self.compiled_codes = [compile(expr, "<string>", "eval") for expr in self.expressions]
        
    def __call__(self, *args: Any) -> Any:
        """
        Evalúa las expresiones adaptándose al protocolo DynamicalSystem (t, state_tensor).

        La firma canónica del motor IOB es ``F(t, state)`` donde ``state`` es un
        tensor de forma ``(*spatial_shape, n_dims)``. Las expresiones del usuario
        pueden referenciar ``x`` e ``y`` (primeras dos dimensiones espaciales).
        """
        if len(args) == 2 and isinstance(args[1], torch.Tensor):
            # Firma estándar DynamicalSystem: F(t, state_tensor)
            # args[0] = t (float, ignorado para sistemas autónomos)
            # args[1] = state_tensor de forma (*spatial_shape, n_dims)
            X = args[1]
        elif len(args) == 1:
            X = args[0]
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float64)
        else:
            raise ValueError(
                f"DynamicExpressionManifold esperaba F(t, state) o F(state), "
                f"recibió {len(args)} argumentos con tipos {[type(a).__name__ for a in args]}."
            )

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)

        x = X[..., 0]
        y = X[..., 1] if X.shape[-1] > 1 else torch.zeros_like(x)
        device = X.device

        context: Dict[str, Any] = {
            "torch": torch, "x": x, "y": y,
            "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
            "exp": torch.exp, "log": torch.log, "sqrt": torch.sqrt,
            "pi": torch.pi, "sinh": torch.sinh, "cosh": torch.cosh, "tanh": torch.tanh,
            "abs": torch.abs,
        }

        results: List[torch.Tensor] = []
        for compiled in self.compiled_codes:
            try:
                res = eval(compiled, {"__builtins__": {}}, context)
                if not isinstance(res, torch.Tensor):
                    res = torch.full(x.shape, float(res), dtype=torch.float64, device=device)
                res = res.to(torch.float64)
                if res.shape != x.shape:
                    res = torch.broadcast_to(res, x.shape).clone()
                results.append(res)
            except Exception as e:
                raise RuntimeError(
                    f"Fallo matemático al evaluar '{self.expression}': {e}"
                )

        if len(results) == 1:
            return results[0]
        return torch.stack(results, dim=-1)


def load_json_config(filepath: str) -> Dict[str, Any]:
    """Ingesta y parsea un archivo de configuración estructural en formato JSON."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"El archivo de configuración no existe: {filepath}")
        
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    return data


def load_tensor_manifold(filepath: str, expected_dim: Optional[int] = None) -> torch.Tensor:
    """
    Ingesta una trayectoria o batch de embeddings latentes. Soporta .pt, .npy y .json.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"El locus de datos especificado no existe: {filepath}")

    if path.suffix == ".npy":
        try:
            import numpy as np
            np_array = np.load(path)
            tensor = torch.from_numpy(np_array).to(torch.float64)
        except ImportError:
            raise ImportError("La ingesta de archivos .npy requiere el paquete 'numpy'. Instale el entorno [vis].")
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ["data", "embeddings", "trajectory", "tensor", "values"]:
                if key in data:
                    raw_list = data[key]
                    break
            else:
                raise ValueError("El JSON debe contener una clave indexable como 'data' o 'trajectory'.")
        elif isinstance(data, list):
            raw_list = data
        else:
            raise ValueError("Estructura JSON inválida para colector tensorial.")
        tensor = torch.tensor(raw_list, dtype=torch.float64)
    else:
        tensor = torch.load(path).to(torch.float64)
        
    if expected_dim is not None and tensor.shape[-1] != expected_dim:
         raise ValueError(
             f"Anomalía topológica en la ingesta: Se esperaba dimensión latente {expected_dim}, "
             f"pero el colector de datos tiene dimensión {tensor.shape[-1]}."
         )
         
    return tensor


def load_discrete_topology(filepath: str) -> torch.Tensor:
    """Reconstruye la adyacencia rala a partir de GraphML o matrices en JSON."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Topología discreta no encontrada: {filepath}")

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "adjacency" in data:
            matrix_list = data["adjacency"]
        elif isinstance(data, list):
            matrix_list = data
        else:
            raise ValueError("El JSON de topología debe contener una matriz o la clave 'adjacency'.")
        
        dense_tensor = torch.tensor(matrix_list, dtype=torch.float64)
        indices = torch.where(dense_tensor != 0)
        values = dense_tensor[indices]
        indices_tensor = torch.stack(indices)
        return torch.sparse_coo_tensor(indices_tensor, values, dense_tensor.shape).coalesce()

    try:
        import networkx as nx
    except ImportError:
         raise ImportError("El parseo de grafos requiere el paquete 'networkx'. Instale el entorno [vis].")

    if path.suffix == ".graphml":
        graph = nx.read_graphml(path)
        scipy_sparse = nx.to_scipy_sparse_array(graph)
        coo = scipy_sparse.tocoo()
        
        indices = torch.tensor([coo.row, coo.col], dtype=torch.int64)
        values = torch.tensor(coo.data, dtype=torch.float64)
        shape = coo.shape
        
        adjacency = torch.sparse_coo_tensor(indices, values, shape).coalesce()
        return adjacency
    else:
        raise NotImplementedError(f"Formato topológico no soportado: {path.suffix}")


def load_custom_manifold(filepath: str, class_name: str) -> Any:
    """
    Inyecta dinámicamente un sistema hiper-trascendental definido por el usuario 
    en el motor de localización continuo.
    
    Args:
        filepath: Ruta al módulo Python (.py) del usuario.
        class_name: Nombre de la clase que hereda o implementa la ecuación del sistema.
        
    Returns:
        Any: Instancia de la variedad matemática inyectada.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"El módulo de ecuaciones dinámicas no existe: {filepath}")

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Fallo crítico al instrumentar el módulo: {filepath}")
        
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, class_name):
        raise AttributeError(f"El operador '{class_name}' no fue hallado en el módulo {module_name}.")
        
    manifold_class = getattr(module, class_name)
    return manifold_class()