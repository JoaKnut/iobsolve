r"""
Módulo de Monitoreo de Espacios Latentes (AI/ML).

Detecta fragmentaciones y colapsos modales en arquitecturas de aprendizaje
profundo (p.ej., GANs, VAEs, Autoencoders, LLMs) mediante el Operador de
Bisagra Discreto (D-IOB). El módulo transforma el volumen latente en un grafo
de interpolación estocástico para evaluar la varianza de regularidad topológica
del espacio de representaciones.

**Motivación**: El colapso modal se produce cuando los gradientes del generador
o encoder colapsan todas las representaciones latentes hacia un único punto
atractor, reduciendo la diversidad efectiva del modelo. El D-IOB lo detecta
como una degeneración homogénea de la isometría baricéntrica: todos los nodos
tienen cohesión alta (:math:`1 - Q_i \approx 1`), indicando que no hay
varianza entre vecinos.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto".
       Sección 4.2: Dinámica Latente y Detección del Colapso Modal (IA).
.. [2] Goodfellow, I. et al. (2014). "Generative adversarial nets".
       *NeurIPS 2014*.
"""

from iobsolve.core.space import DiscreteTopology
from iobsolve.core.types import NodalStateVector
from iobsolve.discrete.hinge import DiscreteIntegrityOperator


class ModeCollapseDetector:
    r"""
    Sensor de Colapso Modal para Arquitecturas Profundas.

    Evalúa si la energía de activación se está concentrando asimétricamente en
    un subconjunto ínfimo de la red, indicando una pérdida de diversidad en las
    representaciones latentes (efecto atractor espurio).

    El detector opera en dos pasos:

    1. **Cómputo del estrés nodal** :math:`Q_i \in [0,1]` via D-IOB
       (Laplaciano Combinatorio sobre la topología de batch).
    2. **Inversión a cohesión**: :math:`\text{cohesion}_i = 1 - Q_i`. Los nodos
       con alta cohesión (baja varianza con sus vecinos) son los colapsados.
    3. **Umbral de alarma**: si más del 10 % del espacio latente supera
       ``collapse_threshold``, se declara colapso.

    Parameters
    ----------
    topology : DiscreteTopology
        La malla k-NN :math:`\mathcal{G}` generada dinámicamente a partir del
        mini-batch. Típicamente una topología completamente conectada
        :math:`K_B` (todos los embeddings son vecinos de todos).
    collapse_threshold : float, default=0.85
        Tolerancia máxima de cohesión normalizada :math:`\tau_c \in [0, 1]`.
        Un embedding con cohesión :math:`> \tau_c` es considerado colapsado.
        Valores bajos (~0.7) son más sensibles; valores altos (~0.95) solo
        detectan colapsos severos.

    Attributes
    ----------
    operator : DiscreteIntegrityOperator
        Instancia interna del D-IOB con tolerancia predeterminada.

    Examples
    --------
    Detección de colapso en un batch donde todos los vectores son idénticos:

    >>> import torch
    >>> from iobsolve.core.space import DiscreteTopology
    >>> from iobsolve.plugins.discrete.mode_collapse import ModeCollapseDetector
    >>> N, D = 64, 128
    >>> base = torch.randn(D, dtype=torch.float64)
    >>> collapsed = base.repeat(N, 1)  # todos idénticos
    >>> adj = torch.ones((N, N)) - torch.eye(N)
    >>> detector = ModeCollapseDetector(DiscreteTopology(adj), collapse_threshold=0.8)
    >>> detector.scan_activations(collapsed)
    True

    Detección de isometría saludable con vectores ortogonales:

    >>> healthy = torch.eye(N, dtype=torch.float64)
    >>> detector.scan_activations(healthy)
    False
    """

    def __init__(
        self,
        topology: DiscreteTopology,
        collapse_threshold: float = 0.85,
    ) -> None:
        self.topology = topology
        self.operator = DiscreteIntegrityOperator()
        self.collapse_threshold = collapse_threshold

    def scan_activations(self, activation_vector: NodalStateVector) -> bool:
        r"""
        Analiza el espacio latente y evalúa la degeneración de la variedad.

        Parameters
        ----------
        activation_vector : NodalStateVector
            Tensor de características o embeddings
            :math:`\mathbf{x}_i \in \mathbb{R}^{B \times m}`, donde :math:`B`
            es el tamaño del batch y :math:`m` la dimensión latente.
            Debe ser compatible con la cardinalidad :math:`|V|` de la topología.

        Returns
        -------
        bool
            ``True`` si más del 10 % del hiperespacio ha colapsado hacia un
            atractor puntual (cohesión :math:`> \tau_c`), indicando
            degeneración activa. ``False`` si la isometría latente es saludable.

        Notes
        -----
        **Interpretación de la cohesión**: El estrés D-IOB :math:`Q_i` mide la
        divergencia de un nodo respecto a su vecindad. Un valor alto (:math:`Q_i
        \approx 1`) indica que el nodo es *diferente* de sus vecinos — estado
        saludable. Un valor bajo (:math:`Q_i \approx 0`) implica que el nodo es
        *idéntico* a sus vecinos — indicativo de colapso. Por eso la cohesión
        se define como :math:`1 - Q_i`.

        Complexity
        ----------
        :math:`\mathcal{O}(k \cdot B)` dependiente del tamaño del lote
        :math:`B` y la conectividad :math:`k`, asegurando un overhead de
        diagnóstico sub-milisegundo durante el *backward pass* de entrenamiento.
        """
        # 1. Calcular el estrés (divergencia baricéntrica)
        stress = self.operator.compute_stress(
            state_tensor=activation_vector,
            topology=self.topology,
            laplacian_type="combinatorial",
            normalize_output=True,
        )

        # 2. Invertir a cohesión: 1 - Q_i
        cohesion = 1.0 - stress

        # 3. Ratio de nodos colapsados
        collapsed_nodes = cohesion[cohesion > self.collapse_threshold]
        collapse_ratio = float(collapsed_nodes.numel()) / self.topology.measure
        return collapse_ratio > 0.10
