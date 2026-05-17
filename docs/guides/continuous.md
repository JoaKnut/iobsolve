# Dominio Continuo (Paper I)

## Descripción General

El motor continuo aísla singularidades (raíces y atractores) de campos vectoriales
$F : \mathbb{R}^n \to \mathbb{R}^n$ sin linealizar el Jacobiano.
En su lugar, combina tres componentes:

1. **IOB-QuadTree** — bisección recursiva del espacio de fases.
2. **Filtro TVI** — test de cambio de signo en $\mathcal{O}(N)$ para descartar subdominios sin raíces.
3. **IOB-FFT** — razón de energía espectral de alta frecuencia como criterio de integridad final.

## IOB-QuadTree

El `SpatialPartitionEngine` bisecta recursivamente un hipercubo $\Omega \subset \mathbb{R}^n$
en $2^n$ subdominios. En cada nodo, el cierre `CriterioIntegridad` decide si el
subdominio contiene una singularidad; si no, la rama se poda. El árbol tiene como máximo
$2^{n \cdot d}$ hojas para profundidad $d$.

```python
from iobsolve.core.partition import SpatialPartitionEngine

motor = SpatialPartitionEngine(max_depth=10)
dominio = ((-5.0, 5.0), (-5.0, 5.0))

def mi_criterio(subdominio):
    # Retorna True si el subdominio puede contener una singularidad
    return ...

nodo_raiz = motor.isolate_singularities(dominio, mi_criterio)
hojas = motor.extract_singular_manifolds(nodo_raiz)
```

## Filtro TVI

Antes de calcular la FFT, el mapeador verifica si **cada componente** de $F$ cambia
de signo dentro del subdominio. Este es el Teorema del Valor Intermedio aplicado
componente a componente — condición necesaria para la existencia de una raíz:

$$\min_\Omega F_c \leq 0 \leq \max_\Omega F_c \quad \forall c$$

El filtro TVI elimina la mayoría de los subdominios en $\mathcal{O}(N)$
(una sola pasada de min/max), evitando la FFT de $\mathcal{O}(N \log N)$
sobre regiones vacías.

## Criterio Espectral IOB-FFT

Para los subdominios que pasan el TVI, se calcula el estrés espectral $\mathcal{Q}_\text{spec}$:

$$\mathcal{Q}_\text{spec} = \frac{\sum_{\nu > \nu_c} \sum_c |\mathcal{F}_c(\nu)|^2}
                                  {\sum_\nu \sum_c |\mathcal{F}_c(\nu)|^2}$$

Una ventana de Tukey con `taper_fraction=0.15` suprime el *spectral leakage* antes de la FFT.
El subdominio se marca como singular si $\mathcal{Q}_\text{spec} > \tau_c$.

## FlowTheoremLocator

```python
from iobsolve.continuous.flow_theorem import FlowTheoremLocator
from iobsolve.plugins.continuous.singularities import TranscendentalManifold

locator = FlowTheoremLocator(
    system_equation=TranscendentalManifold(),
    grid_resolution=16,        # malla FFT local
    spectral_threshold=1e-3,   # τ_c
    require_sign_change=True,
    device="cpu",              # o "cuda"
)

raices = locator.locate_root_centroids(
    initial_domain=((-10.0, 10.0), (-2.0, 2.0)),
    max_depth=8,
)
```

## Operador de Integridad Continuo

El `ContinuousIntegrityOperator` calcula $\mathcal{H}(x) = |\nabla^2\phi(x)|$
como un campo de estrés diferenciable utilizable como pérdida en PyTorch:

```python
import torch
from iobsolve.core.space import EuclideanManifold
from iobsolve.continuous.hinge import ContinuousIntegrityOperator

op = ContinuousIntegrityOperator()
variedad = EuclideanManifold(shape=(64, 64), grid_spacing=0.05)
campo = torch.randn(64, 64, dtype=torch.float64, requires_grad=True)

estres = op.compute_stress(campo, manifold=variedad, normalize=True)
perdida = estres.mean()
perdida.backward()  # los gradientes fluyen a través del Laplaciano
```
