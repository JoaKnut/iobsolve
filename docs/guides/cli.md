# Referencia de la CLI

IOB-Solve expone una interfaz de línea de comandos completa mediante el punto de entrada `iobsolve`.

## Opciones Globales

Disponibles en cada subcomando:

| Bandera | Por Defecto | Descripción |
|---------|-------------|-------------|
| `--format {text,json}` | `text` | Formato de serialización de salida |
| `--out-file RUTA` | auto | Ruta de exportación para telemetría JSON |
| `-q / --quiet` | false | Suprimir logs; solo JSON en stdout |
| `--plot RUTA` | — | Guardar visualización en `.png` (requiere `[vis]`) |
| `--config RUTA` | — | Archivo de configuración JSON que sobreescribe las banderas |

---

## `roots` — Aislamiento de Singularidades

```
iobsolve roots [opciones]
```

| Bandera | Por Defecto | Descripción |
|---------|-------------|-------------|
| `--radius R` | 5.0 | Semilado del dominio cuadrado inicial $\Omega$ |
| `--depth D` | 8 | Profundidad máxima de recursión del QuadTree |
| `--res N` | 16 | Resolución de la malla FFT local (límite: 32) |
| `--tau-spec T` | 1e-3 | Umbral de energía de alta frecuencia $\tau_c$ |
| `--no-sign-filter` | off | Deshabilitar el pre-filtro TVI |
| `--manifold ARCHIVO:CLASE` | — | Clase Python de sistema externo |
| `--expr EXPR` | — | Expresión algebraica en línea |

**Ejemplos:**

```bash
# Variedad trascendental predeterminada
iobsolve roots --radius 10 --depth 10

# Expresión en línea personalizada
iobsolve roots --expr "sin(x) - y, cos(y) - x" --radius 4

# Sistema externo + exportación completa
iobsolve roots --manifold ode.py:VanDerPol --radius 3 \
               --format json --out-file raices.json --plot raices.png
```

---

## `shield` — Cirugía Anti-DDoS

```
iobsolve shield [opciones]
```

| Bandera | Por Defecto | Descripción |
|---------|-------------|-------------|
| `--nodes N` | 1000 | Número de nodos simulados |
| `--tau T` | 3.0 | Umbral Z-Score para la cirugía |
| `--attack` | off | Inyectar ataque volumétrico asimétrico |
| `-i / --input RUTA` | — | Cargar topología `.graphml` o JSON |
| `--traffic RUTA` | — | Cargar vector de tráfico (`.npy`, `.pt`) |

---

## `audit` — Isometría del Espacio Latente

```
iobsolve audit [opciones]
```

| Bandera | Por Defecto | Descripción |
|---------|-------------|-------------|
| `--batch B` | 128 | Tamaño del batch |
| `--dim D` | 256 | Dimensión latente |
| `--tau T` | 0.85 | Umbral de cohesión para colapso |
| `-i / --input RUTA` | — | Cargar tensor de embeddings (`.npy`, `.pt`) |

---

## `dynamics` — Sensor de Alerta Temprana

```
iobsolve dynamics [opciones]
```

| Bandera | Por Defecto | Descripción |
|---------|-------------|-------------|
| `--dim D` | 40 | Dimensionalidad del espacio de fases |
| `--l-metric` | auto | Resolución óptica (`auto` o flotante) |
| `-i / --input RUTA` | — | Trayectoria empírica (`.npy`, `.pt`) |

---

## `check` — Diagnóstico del Entorno

```bash
iobsolve check
```

Imprime las versiones de Python, PyTorch, CUDA, NumPy y Matplotlib.

---

## Configuración JSON

Cualquier bandera puede preestablecerse en un archivo de configuración JSON:

```json
{
  "radius": 10.0,
  "depth": 12,
  "tau_spec": 5e-4,
  "res": 24,
  "format": "json",
  "out_file": "resultados.json"
}
```

```bash
iobsolve roots --config config.json
```
