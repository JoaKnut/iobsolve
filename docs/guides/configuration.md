# Configuración

## Archivos de Configuración JSON

Cualquier bandera de la CLI puede preestablecerse en un archivo de configuración JSON y cargarse con `--config`:

```bash
iobsolve roots --config config.json
```

Los nombres de las banderas usan guiones bajos (convención Python), coincidiendo con los
atributos internos de `argparse.Namespace`. Las banderas con guiones (`--tau-spec`) se mapean
a atributos con guion bajo (`tau_spec`):

```json
{
  "radius":   10.0,
  "depth":    12,
  "res":      24,
  "tau_spec": 5e-4,
  "format":   "json",
  "out_file": "resultados.json"
}
```

Las banderas en línea sobrescriben los valores del archivo de configuración:

```bash
iobsolve roots --config config.json --depth 6  # depth=6 tiene prioridad
```

## Detección del Dispositivo

IOB-Solve detecta automáticamente el mejor dispositivo de cómputo disponible:

```python
import torch
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
```

Para forzar un dispositivo específico, páselo a la API Python:

```python
locator = FlowTheoremLocator(..., device="cuda")
```

La CLI siempre usa el mejor dispositivo disponible de forma automática.

## Control de Sensibilidad

| Parámetro | Efecto |
|-----------|--------|
| `--tau-spec` más bajo | Más singularidades detectadas, más falsos positivos |
| `--depth` más alto | Localización más fina, cómputo exponencialmente mayor |
| `--res` más alto | FFT más precisa, $\mathcal{O}(N \log N)$ por nodo |
| `--tau` (shield) más bajo | Detección DDoS más temprana, más falsos positivos |
| `--tau` (audit) más alto | Solo detecta colapsos severos |
