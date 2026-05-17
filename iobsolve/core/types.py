r"""
Módulo de Definiciones de Tipos (Type Hints) para IOB-Solve.

Establece las primitivas algebraicas y topológicas utilizando tensores diferenciables.
El ecosistema emplea PyTorch para soportar diferenciación automática (Autograd)
y aceleración por hardware (CUDA/MPS), garantizando la estabilidad asintótica de 
las operaciones descritas en los marcos teóricos I y II.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra: Aislamiento 
       Topológico de Singularidades y Control de Bifurcaciones en Variedades Continuas".
.. [2] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto: 
       Laplacianos de Grafos, Detección de Anomalías Asíncronas y Colapsos en Redes Complejas".
"""

from typing import Callable, TypeAlias
import torch

# =============================================================================
# DOMINIO CONTINUO (Paper I: Variedades Euclidianas Diferenciables)
# =============================================================================

#: Representa un campo escalar \phi(x) o vectorial v(x) mapeado sobre la variedad \Omega.
#: Se asume dtype=torch.float64 para prevenir pérdida de precisión en derivadas de alto orden.
ManifoldField: TypeAlias = torch.Tensor

#: Coordenadas espaciales del hipercubo evaluado (\Omega \subset \mathbb{R}^n).
SpatialDomain: TypeAlias = tuple[torch.Tensor, ...]

#: Definición de un sistema dinámico como un campo vectorial de flujos: dx/dt = F(x, t).
DynamicalSystem: TypeAlias = Callable[[float, ManifoldField], ManifoldField]

# =============================================================================
# DOMINIO DISCRETO (Paper II: Topologías No Euclidianas y Redes Complejas)
# =============================================================================

#: Matriz de Adyacencia A_{ij}. Puede instanciarse en memoria densa o dispersa 
#: (torch.sparse_coo / torch.sparse_csr) para garantizar complejidad \mathcal{O}(k_i).
AdjacencyMatrix: TypeAlias = torch.Tensor

#: Matriz Diagonal de Grados D_{ii} = \sum_j A_{ij}.
DegreeMatrix: TypeAlias = torch.Tensor

#: Vector de estado \mathbf{x}_i(t) o tensor latente proyectado sobre los vértices del grafo.
NodalStateVector: TypeAlias = torch.Tensor

# =============================================================================
# OPERADORES Y MÉTRICAS
# =============================================================================

#: Tensor de Estrés \mathcal{H} o \mathcal{Q}_i. Cuantifica la divergencia o 
#: el cizallamiento geométrico evaluado por el motor del IOB.
StressTensor: TypeAlias = torch.Tensor