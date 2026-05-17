# IOB-Solve

**Framework de Integridad Topológica para Aislamiento de Singularidades y Cirugía de Anomalías en Redes**

---

IOB-Solve es un framework Python que implementa el *Operador de Integridad de Bisagra* (IOB) — un motor matemático que cuantifica el estrés topológico en variedades continuas y topologías de red discretas. Está fundamentado en dos artículos de investigación de Joaquín Knuttzen (2026) y opera nativamente sobre tensores PyTorch diferenciables.

## Capacidades de un Vistazo

=== "Dominio Continuo (Paper I)"

    - **Aislamiento de raíces** vía IOB-QuadTree (bisección recursiva + filtro TVI + IOB-FFT)
    - **Mapeo de campo de estrés** $\mathcal{H}(x) = |\nabla^2\phi(x)|$ sobre variedades euclidianas
    - **Detección espectral de singularidades** vía FFT N-dimensional con ventana de Tukey
    - Salidas diferenciables compatibles con PyTorch Autograd

=== "Dominio Discreto (Paper II)"

    - **Cuantificación de estrés nodal** $Q_i$ vía Laplaciano-Beltrami discreto
    - **Normalización Z-Score Robusta** (basada en MAD, punto de ruptura del 50 %)
    - **Cirugía topológica** — extirpación de aristas en $\mathcal{O}(k_i)$ sobre grafos Sparse COO
    - **Detección y mitigación de DDoS** en topologías de red en tiempo real
    - **Auditoría de colapso modal** para espacios latentes de aprendizaje profundo

## Instalación Rápida

```bash
pip install iobsolve                # motor central (solo PyTorch)
pip install "iobsolve[vis]"         # + pila de visualización
pip install -e ".[dev,vis]"         # instalación de desarrollo
```

## Ejemplo Mínimo

```python
from iobsolve.continuous.flow_theorem import FlowTheoremLocator
from iobsolve.plugins.continuous.singularities import TranscendentalManifold

locator = FlowTheoremLocator(
    system_equation=TranscendentalManifold(),
    grid_resolution=16,
    spectral_threshold=1e-3,
)
raices = locator.locate_root_centroids(
    initial_domain=((-10.0, 10.0), (-2.0, 2.0)),
    max_depth=8,
)
# → raíces en (nπ, 0) para n ∈ ℤ
```

## Navegación

- [**Primeros Pasos**](guides/getting_started.md) — Instalación, primeros pasos y quickstart de la CLI.
- [**Dominio Continuo**](guides/continuous.md) — IOB-QuadTree, filtro TVI, IOB-FFT explicados.
- [**Dominio Discreto**](guides/discrete.md) — D-IOB, Z-Score Robusto, cirugía topológica.
- [**Referencia de API**](api/index.md) — Autodoc completo de cada clase y función pública.
- [**Fundamentos Matemáticos**](reference/math.md) — Bases teóricas del framework.
