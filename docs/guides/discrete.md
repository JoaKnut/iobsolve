# Dominio Discreto (Paper II)

## Descripción General

El motor discreto opera sobre topologías de grafo $\mathcal{G}(V, E, W)$,
cuantificando el *índice de estrés nodal* $Q_i$ y ejecutando cirugía topológica
en $\mathcal{O}(k_i)$ sobre matrices Sparse COO.

## Operador de Integridad Discreto (D-IOB)

El D-IOB proyecta el vector de estado nodal sobre el Laplaciano del grafo:

**Combinatorio:** $\mathbf{L} = \mathbf{D} - \mathbf{W}$

**Normalizado:** $\mathcal{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$

```python
import torch
from iobsolve.core.space import DiscreteTopology
from iobsolve.discrete.hinge import DiscreteIntegrityOperator

adj = torch.rand(100, 100, dtype=torch.float64)
adj = (adj + adj.T) / 2  # simetrizar
topologia = DiscreteTopology(adjacency=adj)

op = DiscreteIntegrityOperator()
estado = torch.randn(100, 16, dtype=torch.float64)  # N nodos × m características

estres = op.compute_stress(
    estado,
    topology=topologia,
    laplacian_type="normalized",   # o "combinatorial"
    normalize_output=True,         # escalar a [0,1]
)
```

## Z-Score Robusto (MAD)

El `RecursiveTopologicalZScore` reemplaza los estimadores vulnerables de media/varianza
con la Desviación Absoluta de la Mediana (MAD), alcanzando un punto de ruptura del 50 %:

$$\mathcal{M}_i(t) = \frac{0.6745 \cdot (Q_i - \tilde{Q}^*(t))}
                          {\max(\text{MAD}(t),\, \varepsilon)}$$

El factor de olvido exponencial $\lambda$ previene el *concept drift*:

```python
from iobsolve.discrete.estimators import RecursiveTopologicalZScore

estimador = RecursiveTopologicalZScore(num_nodes=1000, decay_factor=0.99)
z_scores = estimador.update_and_compute(estres)  # retorna M_i(t)
anomalias = z_scores > 3.0  # τ = 3 (cota de Chebyshev)
```

## Cirugía Topológica

```python
from iobsolve.discrete.surgery import TopologicalSurgeon

cirujano = TopologicalSurgeon(topology=topologia)

# Aislamiento completo de vértices (todas las aristas de nodos anómalos → 0)
indices_singulares = torch.where(anomalias)[0]
topologia_limpia = cirujano.isolate_vertices(indices_singulares)

# Poda selectiva de aristas (un nodo, objetivos específicos)
topologia_limpia = cirujano.prune_asymmetric_edges(
    source_index=0,
    target_indices=torch.tensor([3, 7, 42]),
)
```

## Escudo DDoS

```python
from iobsolve.plugins.discrete.network_shield import DDoSShield

shield = DDoSShield(topology=topologia, critical_threshold=3.0)
trafico = torch.abs(torch.randn(N, dtype=torch.float64))
trafico[0] = 9999.0  # ataque simulado

topo_segura, alertas = shield.process_telemetry(trafico)
# topo_segura: topología con nodos anómalos extirpados
# alertas: tensor booleano, True en nodos comprometidos
```

## Detección de Colapso Modal

```python
from iobsolve.plugins.discrete.mode_collapse import ModeCollapseDetector

B = 256
adj = torch.ones((B, B)) - torch.eye(B)
topo = DiscreteTopology(adjacency=adj)

detector = ModeCollapseDetector(topo, collapse_threshold=0.85)
embeddings = modelo.encode(batch)  # (B, dim_latente)
esta_colapsando = detector.scan_activations(embeddings)

if esta_colapsando:
    print("⚠️  Colapso modal detectado — deteniendo entrenamiento.")
```
