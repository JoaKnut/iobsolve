<div align="center">

# IOB-Solve

**Framework de Integridad Topológica para Aislamiento de Singularidades y Cirugía de Anomalías en Redes**

[[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20016070.svg)](https://doi.org/10.5281/zenodo.20016070)](https://doi.org/10.5281/zenodo.20262050)
[![Versión](https://img.shields.io/badge/versión-0.2.0-blue.svg)](https://github.com/JoaKnut/iobsolve)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![Licencia: MIT](https://img.shields.io/badge/licencia-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/backend-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Documentación](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://JoaKnut.github.io/iobsolve/)


</div>

---

## Descripción General

**IOB-Solve** es un framework Python que implementa el *Operador de Integridad de Bisagra* (IOB) — un motor matemático que cuantifica el estrés topológico en variedades continuas y topologías de red discretas. Opera nativamente sobre tensores PyTorch diferenciables, con soporte para funciones de pérdida compatibles con Autograd y aceleración por hardware (CUDA / MPS).

El motor está fundamentado en dos artículos teóricos complementarios:

- **Paper I** — *Formalismo de Integridad de Bisagra: Aislamiento Topológico de Singularidades y Control de Bifurcaciones en Variedades Continuas* (Knuttzen, 2026)
- **Paper II** — *Formalismo de Integridad de Bisagra Discreto: Laplacianos de Grafos, Detección de Anomalías Asíncronas y Colapsos en Redes Complejas* (Knuttzen, 2026)

📚 **Documentación completa:** https://JoaKnut.github.io/iobsolve/

### Capacidades Principales

| Dominio | Motor | Aplicación |
|---------|-------|------------|
| Continuo | IOB-QuadTree + TVI + FFT | Aislamiento de singularidades y raíces en campos vectoriales |
| Continuo | IOB-FFT (SpectralIntegrityMapper) | Mapeo de estrés de alta frecuencia sobre variedades |
| Discreto | D-IOB (Laplaciano-Beltrami) | Cuantificación de estrés nodal en grafos |
| Discreto | Cirugía Topológica | Extirpación de anomalías en red en tiempo real (DDoS) |
| IA / ML | ModeCollapseDetector | Auditoría de isometría del espacio latente en modelos profundos |

---

## Fundamentos Matemáticos

### Dominio Continuo (Paper I)

El Operador de Integridad de Bisagra Continuo evalúa el campo de estrés topológico $\mathcal{H}(x)$ como la magnitud de la divergencia del gradiente (Laplaciano):

$$\mathcal{H}(x) = \left| \nabla^2 \phi(x) \right|$$

calculado mediante diferencias finitas centrales de segundo orden sobre una variedad euclidiana discretizada $\Omega \subset \mathbb{R}^n$.

**Aislamiento de Raíces (IOB-QuadTree):** El `FlowTheoremLocator` bisecta el espacio de fases recursivamente, combinando dos criterios complementarios:

1. **Filtro TVI (Teorema del Valor Intermedio)** — test de cambio de signo en $\mathcal{O}(N)$; descarta subdominios sin raíces antes de la FFT.
2. **Criterio espectral (IOB-FFT)** — la razón de energía de alta frecuencia $\mathcal{Q}_{\text{spec}} \in [0, 1]$ confirma discontinuidades geométricas:

$$\mathcal{Q}_{\text{spec}} = \frac{\displaystyle\sum_{\nu > \nu_c} \sum_c \left|\mathcal{F}_c(\nu)\right|^2}{\displaystyle\sum_\nu \sum_c \left|\mathcal{F}_c(\nu)\right|^2}$$

Una ventana de Tukey suprime el *spectral leakage* antes de cada FFT local.

### Dominio Discreto (Paper II)

El Operador de Bisagra Discreto proyecta el vector de estado nodal sobre el Laplaciano del grafo $\mathbf{L} = \mathbf{D} - \mathbf{W}$ (combinatorio) o su variante normalizada $\mathcal{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$ para extraer el *residuo baricéntrico*:

$$\mathbf{R}_i(t) = -(\mathbf{L}\,\mathbf{X})_i$$

El **Z-Score Topológico Robusto** (basado en MAD) normaliza el estrés instantáneo $Q_i$ contra la dispersión poblacional, elevando el punto de ruptura estadístico al 50 %:

$$\mathcal{M}_i(t) = \frac{0.6745 \cdot \left(Q_i(t) - \tilde{Q}^*(t)\right)}{\max\!\left(\text{MAD}(t),\, \varepsilon\right)}$$

donde $\tilde{Q}^*(t)$ es la mediana mezclada exponencialmente (factor de olvido $\lambda$) que previene el *concept drift* en tráfico no estacionario.

---

## Instalación

### Mínima (solo motor central)
```bash
pip install iobsolve
```

### Con soporte de visualización
```bash
pip install "iobsolve[vis]"
```

### Entorno de desarrollo (tests + linting)
```bash
git clone https://github.com/JoaKnut/iobsolve.git
cd iobsolve
pip install -e ".[dev,vis]"
```

**Requisitos:** Python ≥ 3.10, PyTorch ≥ 2.0

---

## Inicio Rápido

### 1. Localizar raíces de un campo vectorial (Continuo)

```python
import torch
from iobsolve.continuous.flow_theorem import FlowTheoremLocator
from iobsolve.plugins.continuous.singularities import TranscendentalManifold

locator = FlowTheoremLocator(
    system_equation=TranscendentalManifold(),
    grid_resolution=16,
    spectral_threshold=1e-3,
    require_sign_change=True,
)

dominio = ((-10.0, 10.0), (-2.0, 2.0))  # espacio de fases 2D
raices = locator.locate_root_centroids(dominio, max_depth=8)
print(f"Raíces encontradas: {len(raices)} en {raices}")
```

### 2. Detectar ataques DDoS en una red (Discreto)

```python
import torch
from iobsolve.core.space import DiscreteTopology
from iobsolve.plugins.discrete.network_shield import DDoSShield

N = 1000
adj = torch.zeros((N, N), dtype=torch.float64)
adj[0, 1:] = 1.0; adj[1:, 0] = 1.0

topology = DiscreteTopology(adjacency=adj)
shield   = DDoSShield(topology=topology, critical_threshold=3.0)

trafico = torch.abs(torch.randn(N, dtype=torch.float64))
trafico[0] = 9999.0   # ataque volumétrico simulado en el hub

topologia_segura, alertas = shield.process_telemetry(trafico)
print(f"Nodos anómalos: {alertas.nonzero().squeeze().tolist()}")
```

### 3. Auditar colapso del espacio latente (IA)

```python
import torch
from iobsolve.core.space import DiscreteTopology
from iobsolve.plugins.discrete.mode_collapse import ModeCollapseDetector

B, D = 256, 512
embeddings = torch.randn(B, D, dtype=torch.float64)

adj = torch.ones((B, B), dtype=torch.float64) - torch.eye(B, dtype=torch.float64)
topology  = DiscreteTopology(adjacency=adj)
detector  = ModeCollapseDetector(topology=topology, collapse_threshold=0.85)

colapso_activo = detector.scan_activations(embeddings)
print("Colapso modal detectado:", colapso_activo)
```

---

## Interfaz de Línea de Comandos

IOB-Solve incluye una CLI completa accesible mediante el comando `iobsolve`.

```
iobsolve <comando> [opciones]
```

### Comandos

| Comando | Descripción |
|---------|-------------|
| `roots` | Localiza raíces de campos vectoriales vía IOB-QuadTree + TVI + FFT |
| `spectral` | Mapeo global de densidad topológica (IOB-FFT) |
| `dynamics` | Sensor de alerta temprana para bifurcaciones caóticas |
| `shield` | Cirugía topológica anti-DDoS sobre grafos discretos |
| `audit` | Auditoría de isometría del espacio latente en arquitecturas de IA |
| `check` | Verifica el entorno, versión de PyTorch y hardware disponible |

### Ejemplos

```bash
# Buscar raíces de la variedad trascendental predeterminada en [-10,10]^2
iobsolve roots --radius 10 --depth 10

# Usar una expresión algebraica personalizada
iobsolve roots --expr "sin(x) - y, cos(y) - x" --radius 4 --depth 10

# Inyectar un sistema Python externo y exportar resultados
iobsolve roots --manifold mi_ode.py:MiSistema --radius 6 \
         --format json --out-file raices.json --plot raices.png

# Simular un ataque DDoS en una red de 500 nodos
iobsolve shield --nodes 500 --attack --tau 2.5 --plot topologia.png

# Auditar un tensor latente desde un archivo .pt
iobsolve audit -i embeddings.pt --dim 768 --tau 0.9 --format json

# Verificar el entorno
iobsolve check
```

---

## Estructura del Proyecto

```
iobsolve/
├── core/                        # Primitivas matemáticas
│   ├── types.py                 # Alias de tipos (ManifoldField, StressTensor, …)
│   ├── base_operator.py         # BaseIntegrityOperator abstracto
│   ├── laplacian.py             # ContinuousLaplacian, DiscreteLaplacian
│   ├── space.py                 # EuclideanManifold, DiscreteTopology
│   ├── spectral.py              # SpectralIntegrityMapper (IOB-FFT)
│   └── partition.py             # SpatialPartitionEngine (IOB-QuadTree)
├── continuous/                  # Motor del Paper I
│   ├── hinge.py                 # ContinuousIntegrityOperator
│   └── flow_theorem.py          # FlowTheoremLocator (aislamiento de raíces)
├── discrete/                    # Motor del Paper II
│   ├── hinge.py                 # DiscreteIntegrityOperator (D-IOB)
│   ├── estimators.py            # RecursiveTopologicalZScore (MAD)
│   └── surgery.py               # TopologicalSurgeon (poda)
├── plugins/                     # Aplicaciones específicas de dominio
│   ├── continuous/
│   │   ├── dynamics.py          # Sistema caótico de Lorenz-96
│   │   └── singularities.py     # TranscendentalManifold (referencia)
│   └── discrete/
│       ├── network_shield.py    # DDoSShield
│       └── mode_collapse.py     # ModeCollapseDetector
├── io/                          # Capa de E/S
│   ├── parsers.py               # Ingesta de tensores / topologías / configs
│   ├── exporters.py             # Serialización de telemetría JSON
│   └── visualizers.py           # Renderizadores Matplotlib / NetworkX
└── cli.py                       # Punto de entrada argparse
```

---

## Tests

La suite de pruebas cubre 108 tests en categorías de unidad, integración y benchmarks.

```bash
# Ejecutar la suite completa
pytest

# Con informe de cobertura
pytest --cov=iobsolve --cov-report=term-missing

# Solo un módulo específico
pytest tests/core/test_laplacian.py -v

# Excluir benchmarks (ejecución rápida en CI)
pytest -m "not benchmark"
```

| Directorio | Cobertura |
|------------|-----------|
| `tests/core/` | Laplacianos, espacios, QuadTree |
| `tests/continuous/` | Operador IOB, mapeador espectral, FlowTheoremLocator |
| `tests/discrete/` | D-IOB, Z-Score, Cirugía, DDoS Shield |
| `tests/io/` | Parsers (pt/npy/json/graphml), exporters |
| `tests/cli/` | Tests de CLI extremo a extremo |
| `tests/vis/` | Smoke tests de renderizado Matplotlib |
| `tests/benchmarks/` | Aserciones de escalabilidad (O(k), O(N log N)) |

---

## Archivos de Configuración

IOB-Solve acepta archivos de configuración JSON mediante `--config`:

```json
{
  "radius": 10.0,
  "depth": 12,
  "tau_spec": 5e-4,
  "format": "json",
  "out_file": "resultados.json"
}
```

```bash
iobsolve roots --config config.json
```

---

## Notas de Arquitectura

- **Compatible con Autograd**: todas las operaciones tensoras preservan el grafo de cómputo de PyTorch; los tensores de estrés pueden usarse directamente como funciones de pérdida.
- **Sparse-first**: el motor discreto usa `torch.sparse_coo_tensor` para evaluación del Laplaciano en $\mathcal{O}(k_i)$, evitando productos matriciales densos en $\mathcal{O}(N^3)$.
- **Agnóstico al dispositivo**: los cómputos se ejecutan nativamente en CPU, CUDA o MPS — pase `device="cuda"` al `FlowTheoremLocator` para aceleración GPU.
- **Arquitectura de plugins**: sistemas, variedades y topologías personalizadas pueden inyectarse vía CLI (`--manifold archivo.py:Clase`) o la API Python.

---

## Cita

Si utiliza IOB-Solve en trabajo académico, cite los artículos fundamentales:

```bibtex
@article{knuttzen2026continuo,
  author  = {Knuttzen, Joaquín},
  title   = {Formalismo de Integridad de Bisagra: Aislamiento Topológico de
             Singularidades y Control de Bifurcaciones en Variedades Continuas},
  year    = {2026}
}

@article{knuttzen2026discreto,
  author  = {Knuttzen, Joaquín},
  title   = {Formalismo de Integridad de Bisagra Discreto: Laplacianos de Grafos,
             Detección de Anomalías Asíncronas y Colapsos en Redes Complejas},
  year    = {2026}
}
```

**DOI oficial del framework:** [10.5281/zenodo.20262050](https://doi.org/10.5281/zenodo.20262050)

---

## Licencia

MIT — véase [LICENSE](LICENSE) para más detalles.
