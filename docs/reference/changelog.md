# Registro de Cambios

## v0.2.0 (2026-05-16)

### Nuevas Funcionalidades
- **`FlowTheoremLocator.locate_root_centroids`**: método de conveniencia que ejecuta `locate_roots` y retorna centroides en una sola llamada.
- **`SpectralIntegrityMapper`**: implementación completa del IOB-FFT con ventana de Tukey N-D y criterio compuesto TVI+FFT.
- **`RecursiveTopologicalZScore.decay_factor`**: mezcla exponencial de la mediana histórica para prevenir el *concept drift* en tráfico no estacionario.
- **`DDoSShield._perform_topological_surgery`**: conectado a `TopologicalSurgeon.isolate_vertices` — la cirugía ahora modifica efectivamente la topología (anteriormente era un stub).
- **CLI `--expr`**: expresiones algebraicas en línea para el comando `roots` vía `DynamicExpressionManifold`.
- **CLI `--config`**: archivo de configuración JSON que sobrescribe todas las banderas de la CLI.
- **Suite de tests completa**: 108 tests en core, continuo, discreto, E/S, CLI, visualización y benchmarks.
- **Documentación MkDocs**: sitio autodoc completo con referencia de API, guías y fundamentos matemáticos.

### Correcciones de Errores
- **`DDoSShield._perform_topological_surgery`** era un stub sin operación (`pass`). Ahora delega a `TopologicalSurgeon` y actualiza `self.topology` en lugar.
- **`RecursiveTopologicalZScore.decay_factor`** estaba definido pero nunca utilizado. Ahora aplicado como media móvil exponencial sobre la mediana histórica.
- **`SpectralIntegrityMapper.compute_spectral_stress`**: corregido error de indexado enmascarado donde `amplitudes[mask]` sobre un tensor `(*espacial, n_componentes)` producía energía de alta frecuencia inflada por un factor `n_componentes`. Corregido sumando primero la energía de componentes (`energy_density = amplitudes.pow(2).sum(-1)`) antes de aplicar la máscara radial.

### Cambios Internos
- `pyproject.toml`: agregados `pytest-cov`, meta-extra `[all]`, secciones `[tool.pytest.ini_options]`, `[tool.coverage]`, `[tool.ruff]`.
- Todos los módulos: docstrings completos estilo NumPy con notación matemática, ejemplos y anotaciones de complejidad algorítmica.

## v0.1.0 (2026-05-12)

- Lanzamiento inicial: framework IOB-Solve con motores continuo y discreto.
