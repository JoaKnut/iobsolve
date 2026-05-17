r"""
Módulo de Ciberdefensa Topológica (DDoS Shield).

Orquesta el D-IOB, los estimadores estadísticos recursivos (Z-Score Robusto / MAD)
y la Cirugía Topológica algorítmica para mitigar ataques de Denegación de Servicio
Distribuidos (DDoS) en tiempo real.

El flujo de procesamiento sigue cuatro etapas:

1. **Watchdog de Vacío Topológico**: Aborta si el grafo está desconectado.
2. **Cómputo del Estrés Nodal** (``Q_i``): Magnitud absoluta del payload por nodo.
3. **Z-Score Topológico Robusto** (``M_i``): Normalización resistente al Efecto de Enmascaramiento.
4. **Cirugía Topológica**: Extirpación de vértices con ``M_i > tau`` mediante
   :class:`~iobsolve.discrete.surgery.TopologicalSurgeon`.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra Discreto".
       Sección 4.1: Topologías de Flujo Asíncrono (Mitigación DDoS).
"""

import torch
from iobsolve.core.space import DiscreteTopology
from iobsolve.core.types import NodalStateVector
from iobsolve.discrete.estimators import RecursiveTopologicalZScore
from iobsolve.discrete.surgery import TopologicalSurgeon


class DDoSShield:
    r"""
    Escudo Activo contra Ataques Distribuidos de Denegación de Servicio.

    Ingiere vectores de tráfico asíncronos :math:`\mathbf{x}(t)`, detecta
    deformaciones aisladas en la isometría de flujo mediante el Z-Score
    Topológico Robusto y extirpa los vértices anómalos (*Topological Pruning*)
    sin afectar el ancho de banda legítimo.

    La cirugía se delega a :class:`~iobsolve.discrete.surgery.TopologicalSurgeon`,
    que opera en :math:`\mathcal{O}(k_i)` sobre matrices dispersas (Sparse COO),
    garantizando latencias sub-milisegundo incluso en topologías de escala web.

    Parameters
    ----------
    topology : DiscreteTopology
        El 1-esqueleto base :math:`\mathcal{G}(V, E, W)` de la red a proteger.
        Frecuentemente una topología de estrella bipartita (servidor–clientes).
    critical_threshold : float, default=5.0
        Umbral crítico :math:`\tau` para el Z-Score Robusto
        :math:`\mathcal{M}_i(t)`. Evaluado bajo la desigualdad estricta de
        Chebyshev: nodos con :math:`\mathcal{M}_i > \tau` son extirpados.
        Valores bajos (:math:`\tau \approx 2.5`) aumentan la sensibilidad;
        valores altos (:math:`\tau \geq 5.0`) reducen falsos positivos.

    Attributes
    ----------
    topology : DiscreteTopology
        Variedad topológica activa. Se actualiza *in-place* con cada cirugía.
    threshold : float
        Umbral de corte almacenado para uso interno.
    estimator : RecursiveTopologicalZScore
        Estimador de Z-Score Robusto (MAD) asociado a esta instancia.

    Examples
    --------
    Simulación de un ataque DDoS en una red estrella de 50 nodos:

    >>> import torch
    >>> from iobsolve.core.space import DiscreteTopology
    >>> from iobsolve.plugins.discrete.network_shield import DDoSShield
    >>> N = 50
    >>> adj = torch.zeros((N, N), dtype=torch.float64)
    >>> adj[0, 1:] = 1.0; adj[1:, 0] = 1.0
    >>> topology = DiscreteTopology(adjacency=adj)
    >>> shield = DDoSShield(topology=topology, critical_threshold=3.0)
    >>> traffic = torch.ones(N, dtype=torch.float64)
    >>> traffic[0] = 9999.0  # Inyección volumétrica
    >>> safe_topo, alerts = shield.process_telemetry(traffic)
    >>> bool(alerts[0])
    True
    """

    def __init__(self, topology: DiscreteTopology, critical_threshold: float = 5.0):
        self.topology = topology
        self.threshold = critical_threshold
        self.estimator = RecursiveTopologicalZScore(num_nodes=topology._num_nodes)

    def process_telemetry(
        self, traffic_vector: NodalStateVector
    ) -> tuple[DiscreteTopology, torch.Tensor]:
        r"""
        Procesa asíncronamente una ventana de telemetría y ejecuta cicatrización local.

        El sistema evalúa el Z-Score robusto :math:`\mathcal{M}_i(t)` y, si
        :math:`\mathcal{M}_i(t) > \tau`, fuerza analíticamente
        :math:`w_{ij}(t^+) = 0` aislando la firma atacante del resto de la malla
        mediante :class:`~iobsolve.discrete.surgery.TopologicalSurgeon`.

        Parameters
        ----------
        traffic_vector : NodalStateVector
            Volumen de carga o *payloads* ingresados por la periferia
            :math:`\mathcal{N}_{\text{clients}}`. Tensor 1-D de longitud
            :math:`|V|`.

        Returns
        -------
        topology : DiscreteTopology
            Variedad cicatrizada. Idéntica a la entrada si no hubo ataque.
        alerts : torch.Tensor
            Tensor booleano de longitud :math:`|V|` con ``True`` en los
            vértices extirpados por la cirugía.

        Notes
        -----
        **Flujo interno**:

        1. Watchdog de densidad — aborta si el grafo está desconectado.
        2. :meth:`_compute_nodal_stress` — calcula :math:`Q_i = |x_i|`.
        3. :meth:`~iobsolve.discrete.estimators.RecursiveTopologicalZScore.update_and_compute`
           — normaliza con MAD para obtener :math:`\mathcal{M}_i`.
        4. :class:`~iobsolve.discrete.surgery.TopologicalSurgeon` — extirpa los
           vértices con :math:`\mathcal{M}_i > \tau`.

        Complexity
        ----------
        :math:`\mathcal{O}(k_i)` en matrices dispersas; :math:`\mathcal{O}(|V|^2)`
        en matrices densas (no recomendado para :math:`|V| > 10^4`).
        """
        # 1. WATCHDOG: Verificamos si la red sigue viva antes de analizarla
        if not self.topology.check_integrity_watchdog():
            print("[CRÍTICO] Watchdog de Densidad disparado: Grafo desconectado.")
            return self.topology, torch.zeros(
                self.topology._num_nodes, dtype=torch.bool
            )

        # 2. Computar el Estrés Nodal (Q_i = |payload_i|)
        q_stress = self._compute_nodal_stress(traffic_vector)

        # 3. Z-SCORE ROBUSTO (M_i)
        m_zscore = self.estimator.update_and_compute(q_stress)

        # 4. CIRUGÍA TOPOLÓGICA: delegar a TopologicalSurgeon
        alerts = m_zscore > self.threshold
        if alerts.any():
            self._perform_topological_surgery(alerts)

        return self.topology, alerts

    def _compute_nodal_stress(self, payload: NodalStateVector) -> torch.Tensor:
        r"""
        Calcula el estrés nodal instantáneo :math:`Q_i = |x_i|`.

        Toma el valor absoluto del payload por nodo como proxy del estrés
        baricéntrico. Esta simplificación es válida cuando el vector de tráfico
        ya representa magnitudes (volumen de peticiones, bytes por segundo, etc.).

        Parameters
        ----------
        payload : NodalStateVector
            Vector de carga de longitud :math:`|V|`.

        Returns
        -------
        torch.Tensor
            Vector de estrés :math:`Q_i \geq 0` de longitud :math:`|V|`.
        """
        return torch.abs(payload)

    def _perform_topological_surgery(self, alerts: torch.Tensor) -> None:
        r"""
        Ejecuta la extirpación topológica de los vértices anómalos.

        Delega la operación a :class:`~iobsolve.discrete.surgery.TopologicalSurgeon`,
        que garantiza :math:`\mathcal{O}(k_i)` para matrices dispersas. La
        topología interna del escudo se actualiza *in-place*.

        Parameters
        ----------
        alerts : torch.Tensor
            Tensor booleano de longitud :math:`|V|`. Los vértices con ``True``
            serán aislados (todas sus aristas anuladas).
        """
        singular_indices = torch.where(alerts)[0]
        surgeon = TopologicalSurgeon(topology=self.topology)
        self.topology = surgeon.isolate_vertices(singular_indices)
