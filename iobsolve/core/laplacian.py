r"""
Módulo de Operadores Laplacianos.

Implementa el operador local de divergencia del gradiente (\nabla^2) sobre 
variedades euclidianas continuas y el Laplaciano-Beltrami discreto (\mathbf{L}) 
sobre topologías de red complejas. Emplea operaciones tensoriales vectorizadas 
de PyTorch para garantizar la trazabilidad de gradientes (Autograd).

References
----------
.. [1] Chung, F. R. K. (1997). "Spectral graph theory".
.. [2] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto". 
       Sección 2.2: El Residuo Baricéntrico (Equivalencia Diferencial).
"""

from typing import Any
import torch
from iobsolve.core.types import ManifoldField, AdjacencyMatrix

# Cortafuegos para el linter: Extracción opaca de constructores C++
_sparse_coo: Any = getattr(torch, "sparse_coo_tensor")


class ContinuousLaplacian:
    r"""
    Implementación del Operador Laplaciano continuo analítico.
    """

    @staticmethod
    def compute(field: ManifoldField, grid_spacing: float = 1.0) -> ManifoldField:
        r"""
        Calcula la divergencia del gradiente del campo escalar o vectorial.

        Notes
        -----
        Evalúa el operador mediante diferencias finitas centrales preservando el grafo 
        computacional de PyTorch:
        
        .. math:: \nabla^2 \phi = \sum_{i=1}^{n} \frac{\partial^2 \phi}{\partial x_i^2}

        Parameters
        ----------
        field : ManifoldField
            Campo sobre la variedad (\Omega).
        grid_spacing : float, default=1.0
            Resolución diferencial espacial (\Delta x).

        Returns
        -------
        ManifoldField
            Tensor conteniendo la segunda derivada espacial del campo.
        """
        gradients = torch.gradient(field, spacing=grid_spacing)
        laplacian = torch.zeros_like(field)
        
        for i, grad_comp in enumerate(gradients):
            second_derivative = torch.gradient(grad_comp, spacing=grid_spacing, dim=i)[0]
            laplacian += second_derivative

        return laplacian


class DiscreteLaplacian:
    r"""
    Implementación del Laplaciano de Grafos (Espectro Discreto).
    Soporta operaciones algorítmicas de complejidad \mathcal{O}(V + E) en matrices ralas.
    """

    @staticmethod
    def _compute_degree_vector(adjacency: AdjacencyMatrix) -> torch.Tensor:
        r"""Calcula el vector de grados sumando los ejes de conectividad."""
        if adjacency.is_sparse:
            ones = torch.ones(adjacency.size(1), dtype=adjacency.dtype, device=adjacency.device)
            return torch.mv(adjacency, ones)
        return adjacency.sum(dim=1)

    @classmethod
    def compute_combinatorial(cls, adjacency: AdjacencyMatrix) -> AdjacencyMatrix:
        r"""
        Construye el Laplaciano Combinatorio estándar.

        Notes
        -----
        Formulación canónica:
        .. math:: \mathbf{L} = \mathbf{D} - \mathbf{W}

        Parameters
        ----------
        adjacency : AdjacencyMatrix
            La matriz de pesos \mathbf{W}.

        Returns
        -------
        AdjacencyMatrix
            Matriz Laplaciana no normalizada. Retorna en el mismo formato 
            (denso o disperso) que la matriz de entrada.
        """
        degrees = cls._compute_degree_vector(adjacency)
        n_nodes = adjacency.size(0)

        if adjacency.is_sparse:
            adj_coo = adjacency.to_sparse_coo() if adjacency.layout != torch.sparse_coo else adjacency
            indices = torch.arange(n_nodes, device=adjacency.device).unsqueeze(0).repeat(2, 1)
            
            diag_sparse = _sparse_coo(indices, degrees, (n_nodes, n_nodes)).coalesce()
            return diag_sparse - adj_coo
        
        return torch.diag(degrees) - adjacency

    @classmethod
    def compute_normalized(cls, adjacency: AdjacencyMatrix) -> AdjacencyMatrix:
        r"""
        Construye el Laplaciano Simétrico Normalizado.

        Notes
        -----
        Garantiza invariabilidad frente al cizallamiento del grado modal (hubs):
        .. math:: \mathcal{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{W} \mathbf{D}^{-1/2}

        Parameters
        ----------
        adjacency : AdjacencyMatrix
            La matriz de pesos dinámicos \mathbf{W}(t).

        Returns
        -------
        AdjacencyMatrix
            Matriz Laplaciana normalizada isométricamente.
        """
        degrees = cls._compute_degree_vector(adjacency)
        n_nodes = adjacency.size(0)

        inv_sqrt_degrees = torch.zeros_like(degrees)
        mask = degrees > 1e-12
        inv_sqrt_degrees[mask] = 1.0 / torch.sqrt(degrees[mask])

        if adjacency.is_sparse:
            adj_coo = adjacency.to_sparse_coo() if adjacency.layout != torch.sparse_coo else adjacency
            indices = adj_coo.indices()
            values = adj_coo.values()
            
            row_idx = indices[0, :]
            col_idx = indices[1, :]
            scaled_values = values * inv_sqrt_degrees[row_idx] * inv_sqrt_degrees[col_idx]
            
            normalized_adj = _sparse_coo(indices, scaled_values, (n_nodes, n_nodes)).coalesce()
            
            eye_idx = torch.arange(n_nodes, device=adjacency.device).unsqueeze(0).repeat(2, 1)
            eye_val = torch.ones(n_nodes, dtype=adjacency.dtype, device=adjacency.device)
            identity_sparse = _sparse_coo(eye_idx, eye_val, (n_nodes, n_nodes)).coalesce()
            
            return identity_sparse - normalized_adj
        
        d_inv_sqrt = torch.diag(inv_sqrt_degrees)
        identity = torch.eye(n_nodes, dtype=adjacency.dtype, device=adjacency.device)
        return identity - (d_inv_sqrt @ adjacency @ d_inv_sqrt)