r"""
Módulo de Monitoreo de Espacios Latentes (AI).

Detecta fragmentaciones y colapsos modales en arquitecturas de aprendizaje profundo 
(e.g., GANs, Autoencoders) mediante el Operador de Bisagra Discreto (D-IOB). 
Transforma el volumen latente en un grafo de interpolación estocástico para evaluar 
la varianza de regularidad topológica.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto". 
       Sección 4.2: Dinámica Latente y Detección del Colapso Modal (IA).
"""

from iobsolve.core.space import DiscreteTopology
from iobsolve.core.types import NodalStateVector
from iobsolve.discrete.hinge import DiscreteIntegrityOperator

class ModeCollapseDetector:
    r"""
    Sensor de Colapso Modal para Arquitecturas Profundas.
    
    Evalúa si la energía de activación se está concentrando asimétricamente 
    en un subconjunto ínfimo de la red, indicando una pérdida de diversidad
    en las representaciones latentes (efecto atractor espurio).

    Parameters
    ----------
    topology : DiscreteTopology
        La malla k-NN (\mathcal{G}) generada dinámicamente a partir del mini-batch.
    collapse_threshold : float, default=0.85
        Tolerancia máxima de estrés isométrico \tau_c normalizado en el intervalo [0, 1].
    """

    def __init__(self, topology: DiscreteTopology, collapse_threshold: float = 0.85):
        self.topology = topology
        self.operator = DiscreteIntegrityOperator()
        self.collapse_threshold = collapse_threshold

    def scan_activations(self, activation_vector: NodalStateVector) -> bool:
        r"""
        Analiza el espacio latente y evalúa la degeneración de la variedad.

        Parameters
        ----------
        activation_vector : NodalStateVector
            Tensor de características o embeddings (\mathbf{x}_i \in \mathbb{R}^m).

        Returns
        -------
        bool
            Retorna True si más del 10% del hiperespacio ha colapsado hacia 
            un atractor puntual, superando la tolerancia paramétrica.
            
        Complexity
        ----------
        \mathcal{O}(k \cdot B) dependiente del tamaño del lote (B) y la conectividad k, 
        asegurando un overhead de diagnóstico estrictamente sub-milisegundo durante 
        las iteraciones de entrenamiento (backward pass).
        """
        # 1. Calculamos el estrés (diferencia entre vecinos)
        stress = self.operator.compute_stress(
            state_tensor=activation_vector,
            topology=self.topology,
            laplacian_type='combinatorial', 
            normalize_output=True
        )
        
        # 2. Invertimos el campo: Cohesión = 1.0 - Estrés
        cohesion = 1.0 - stress
        
        # 3. Filtramos los nodos que superan la tolerancia de cohesión (ej: > 0.85)
        collapsed_nodes = cohesion[cohesion > self.collapse_threshold]
        
        # .numel() extrae la cantidad de elementos tensoriales de forma segura
        collapse_ratio = float(collapsed_nodes.numel()) / self.topology.measure
        return collapse_ratio > 0.10