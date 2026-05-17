r"""
Módulo de Abstracciones Espaciales.

Define y acota formalmente las variedades euclidianas continuas (Paper I) 
y las topologías discretas ralas (Paper II), garantizando la isometría y 
coherencia dimensional de los tensores de estado evaluados.
"""

from abc import ABC, abstractmethod
from iobsolve.core.types import ManifoldField, AdjacencyMatrix, NodalStateVector
import torch

class TopologicalInconsistencyError(ValueError):
    r"""
    Excepción lanzada ante una violación axiomática o dimensional.
    Se dispara cuando el tensor de estado \mathbf{X} evaluado no es homeomórfico 
    respecto al espacio topológico subyacente que lo contiene.
    """
    pass

class TopologicalSpace(ABC):
    r"""
    Clase abstracta que define un espacio medible en la teoría del IOB.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        r"""Retorna la dimensión intrínseca topológica (\dim)."""
        pass

    @property
    @abstractmethod
    def measure(self) -> float:
        r"""
        Retorna la medida espacial del sistema.
        Equivalente al volumen continuo de Lebesgue \mu(\Omega) o a la cardinalidad 
        del conjunto de vértices discreto |V|.
        """
        pass


class EuclideanManifold(TopologicalSpace):
    r"""
    Representación de una variedad continua e isométricamente plana (\Omega \subset \mathbb{R}^n).
    Instrumentada de forma nativa por el IOB Continuo.

    Parameters
    ----------
    shape : tuple[int, ...]
        Discretización volumétrica de los ejes de la variedad.
    grid_spacing : float, default=1.0
        Resolución de la métrica diferencial (\Delta x).
    """

    def __init__(self, shape: tuple[int, ...], grid_spacing: float = 1.0) -> None:
        if any(dim <= 0 for dim in shape):
            raise TopologicalInconsistencyError(
                "Las dimensiones de la variedad euclidiana deben ser estrictamente positivas."
            )
        if grid_spacing <= 0.0:
            raise TopologicalInconsistencyError(
                r"La resolución de la métrica (\Delta x) debe ser mayor a cero."
            )

        self._shape = shape
        self._grid_spacing = grid_spacing
        self._dimension = len(shape)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def grid_spacing(self) -> float:
        return self._grid_spacing

    @property
    def measure(self) -> float:
        r"""
        Computa la integral volumétrica \mu(\Omega) basándose en el producto 
        tensorial de las dimensiones y el espaciado de la malla.
        """
        volume = 1.0
        for size in self._shape:
            volume *= (size * self._grid_spacing)
        return float(volume)

    def validate_field(self, field: ManifoldField) -> None:
        r"""
        Verifica que el campo escalar/vectorial sea isomórfico con \Omega.

        Raises
        ------
        TopologicalInconsistencyError
            Si las dimensiones del tensor de entrada difieren de la malla base.
        """
        if field.shape != self._shape:
            raise TopologicalInconsistencyError(
                f"Divergencia isomórfica: El campo tensoral de dimensiones {field.shape} "
                f"no coincide con la topología de la variedad subyacente {self._shape}."
            )


class DiscreteTopology(TopologicalSpace):
    r"""
    Representación de un 1-esqueleto o grafo \mathcal{G}(V, E, W) carente de métrica euclidiana.
    Instrumentado por el Operador de Bisagra Discreto (D-IOB).

    Parameters
    ----------
    adjacency : AdjacencyMatrix
        Tensor cuadrado (denso o disperso) \mathbf{W}(t) que dicta los pesos y la 
        conectividad baricéntrica de la red.
    """

    def __init__(self, adjacency: AdjacencyMatrix) -> None:
        shape = adjacency.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise TopologicalInconsistencyError(
                "Ruptura de la paridad estructural: La matriz de adyacencia "
                "debe ser un endomorfismo cuadrado (N x N)."
            )

        self._adjacency = adjacency
        self._num_nodes = shape[0]

    @property
    def dimension(self) -> int:
        r"""Para topologías de red, la dimensión intrínseca se abstrae como 1 (1-esqueleto)."""
        return 1

    @property
    def measure(self) -> float:
        r"""Retorna la cardinalidad del conjunto de vértices dinámico |V|."""
        return float(self._num_nodes)

    @property
    def adjacency(self) -> AdjacencyMatrix:
        r"""Retorna el operador matricial de conectividad."""
        return self._adjacency

    def validate_nodal_state(self, state_vector: NodalStateVector) -> None:
        r"""
        Verifica que el vector de estado proyectado sea un homeomorfismo válido 
        sobre la cardinalidad |V|.

        Raises
        ------
        TopologicalInconsistencyError
            Si el número de nodos latentes no coincide con el grafo de conectividad.
        """
        if state_vector.shape[0] != self._num_nodes:
            raise TopologicalInconsistencyError(
                f"Desgarro topológico: El vector de estado de dimensión {state_vector.shape[0]} "
                f"no puede inyectarse en el grafo de cardinalidad {self._num_nodes}."
            )
        
    def check_integrity_watchdog(self, min_density: float = 1e-5) -> bool:
        """
        Watchdog para resolver la Ambigüedad del Vacío Topológico.
        Valida que la red no haya colapsado a un estado desconectado.
        """
        if self.adjacency.is_sparse:
            # Para matrices ralas, verificamos que existan aristas activas
            nnz = self.adjacency._nnz()
            total_elements = self.adjacency.shape[0] * self.adjacency.shape[1]
            current_density = nnz / total_elements
        else:
            current_density = torch.count_nonzero(self.adjacency) / self.adjacency.numel()

        if current_density < min_density:
            print("[!] WARNING: Vacío Topológico detectado. La red está desconectada.")
            return False
        return True