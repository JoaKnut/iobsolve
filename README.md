# IOB-Solve: Topological Integrity Operator Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20016070.svg)](https://doi.org/10.5281/zenodo.20016070)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**IOB-Solve** (v0.2.0) es un framework de análisis numérico de alto rendimiento basado en el **Operador de Integridad de Bisagra (IOB)**. Está diseñado para detectar, aislar y controlar rupturas en la topología de sistemas continuos y grafos discretos *antes* de que ocurran colapsos sistémicos o degeneraciones espaciales.

Desarrollado por **Joaquín V. Knuttzen**, el framework elude los cálculos Jacobianos prohibitivos $\mathcal{O}(N^3)$ reemplazándolos con Laplacianos Geométricos instantáneos, particiones recursivas (QuadTree) y mapeo espectral de alta frecuencia (FFT). Esto otorga una **Latencia Positiva** (alerta predictiva temprana) frente a singularidades inminentes.

---

## 🔬 Arquitectura Core e Implementaciones

El ecosistema se divide en dos motores principales apoyados por papers de investigación.

### 1. Dominio Continuo (Teorema del Flujo de Integridad)
Instrumenta la axiomatización geométrica para variedades euclidianas diferenciables. Aísla raíces y singularidades sin recurrir a la linealización iterativa.
* **IOB-QuadTree + TVI:** Ejecuta una bisección recursiva del espacio de fases $\Omega \subset \mathbb{R}^n$. Aplica el Teorema del Valor Intermedio como filtro topológico $\mathcal{O}(N)$ para descartar subdominios sin cambio de signo antes de un costoso análisis tensorial.
* **IOB-FFT:** Confirma singularidades interiores midiendo el estrés armónico de alta frecuencia $\mathcal{Q}_{spec}$ sobre mallas discretizadas locales.

### 2. Dominio Discreto (Laplacianos de Flujo)
Traslada la medición de curvatura a grafos, redes y espacios hiper-dimensionales latentes.
* **Network Shield (DDoS):** Calcula el estrés topológico sobre matrices ralas (sparse). Un nodo con flujo asimétrico masivo deforma el equilibrio baricéntrico del grafo, permitiendo aislarlo (Cirugía Topológica) en tiempo sub-milisegundo $\mathcal{O}(k_i)$.
* **Auditoría IA (Mode Collapse):** Mide la métrica dual de *Estrés/Cohesión*. Si los tensores de activación de un modelo degeneran hacia un atractor puntual (colapso modal), la varianza se reduce a cero. El IOB lo detecta evaluando el Laplaciano Combinatorio del lote de inferencia.

---

## 🚀 Instalación

IOB-Solve está optimizado para entornos de hardware tensorial (CUDA/MPS). 

    git clone https://github.com/JoaKnut/iobsolve.git
    cd iobsolve
    pip install -e .[vis,test]

---

## 💻 Interfaz de Línea de Comandos (CLI)

El módulo interactivo expone 6 sub-comandos principales (`roots`, `spectral`, `dynamics`, `shield`, `audit`, `check`). 

### Opciones Globales de I/O
Todos los comandos aceptan las siguientes banderas globales para manipular las salidas:
* `--format`: Formato de serialización de salida. Opciones: `text` (default), `json`.
* `--out-file PATH`: Ruta física para exportar la telemetría (ej: `reporte.json`).
* `-q`, `--quiet`: Suprime encabezados y logs de consola. Ideal para pipelines CI/CD automatizados.
* `--plot PATH`: Genera una renderización gráfica (PNG) del diagnóstico en la ruta especificada.
* `--config PATH`: Archivo JSON que sobrescribe dinámicamente los flags del CLI.

### 1. Comando `roots`: Localización de Singularidades
Localiza raíces de campos vectoriales vía IOB-QuadTree + TVI + FFT.

* **Parámetros:**
  * `--expr EXPR`: Expresión matemática directa (ej: `'sin(x)-y, cos(y)-x'`).
  * `--manifold FILE:CLASE`: Inyecta un sistema complejo desde un archivo Python externo.
  * `--radius FLOAT`: Radio inicial del dominio de búsqueda (default: `5.0`).
  * `--depth INT`: Profundidad máxima de bisección recursiva (default: `8`).
  * `--res INT`: Resolución de muestreo local para la FFT (default: `16`, max: `32`).
  * `--tau-spec FLOAT`: Umbral de estrés espectral $\tau_c$ (default: `1e-3`).
  * `--no-sign-filter`: Desactiva el filtro TVI para buscar singularidades no-raíz.
* **Ejemplo Complejo:**
    
        iobsolve roots --manifold "mi_sistema.py:Lorenz" --radius 10.0 --depth 12 --tau-spec 1e-4 --plot atractores.png --format json --out-file raices.json

### 2. Comando `shield`: Cirugía Topológica Anti-DDoS
Extirpación asíncrona de nodos anómalos en grafos discretos.

* **Parámetros:**
  * `-i`, `--input PATH`: Topología exportada del usuario en formato `.graphml`.
  * `--traffic PATH`: Carga de tráfico asíncrono o telemetría en formato `.npy` o `.pt`.
  * `--nodes INT`: Cardinalidad de la red a simular en ausencia de un archivo de entrada (default: `1000`).
  * `--tau FLOAT`: Umbral crítico del Z-Score Topológico para ejecutar cirugía (default: `3.0`).
  * `--l-metric METRIC`: Calibración del ruido basal (`auto` o float).
  * `--attack`: Flag booleana que inyecta tráfico asimétrico masivo para probar las defensas.
* **Ejemplo Complejo:**

        iobsolve shield -i red_corporativa.graphml --traffic logs_flujo.pt --tau 2.5 --l-metric auto --plot cirugia.png

### 3. Comando `audit`: Colapso Modal en IA
Auditoría de isometría latente para modelos de Deep Learning.

* **Parámetros:**
  * `-i`, `--input PATH`: Tensor de representaciones latentes extraídas de la red neuronal (`.npy`, `.pt`).
  * `--batch INT`: Tamaño del tensor de mini-batch a simular (default: `128`).
  * `--dim INT`: Dimensionalidad del hiperespacio latente (default: `256`).
  * `--tau FLOAT`: Tolerancia máxima al estrés de cohesión topológica (default: `0.85`).
* **Ejemplo Complejo:**

        iobsolve audit -i embeddings_llama3.pt --dim 4096 --tau 0.90 --format json

### 4. Comando `dynamics`: Sensor Predictivo (Early Warning)
Monitor dinámico predictivo para trayectorias de alta dimensionalidad.

* **Parámetros:**
  * `-i`, `--input PATH`: Archivo de trayectorias empíricas (`.npy`, `.pt`).
  * `--dim INT`: Dimensionalidad del espacio de fases a simular (default: `40`).
  * `--l-metric METRIC`: Resolución óptica de la varianza topológica (`auto` o float).
* **Ejemplo Complejo:**

        iobsolve dynamics -i trayectoria_climatologica.npy --dim 128 --l-metric 3.05 --format json

### 5. Comando `spectral`: Mapeo de Densidad Topológica
Ejecuta la Transformada Rápida de Fourier (IOB-FFT) a nivel global.

* **Parámetros:**
  * `--grid INT`: Resolución de la malla computacional total (default: `1024`).
* **Ejemplo:**

        iobsolve spectral --grid 2048 --out-file densidad_espectral.json

### 6. Comando `check`: Diagnóstico del Entorno
Verifica las versiones, librerías, y la disponibilidad de aceleradores de hardware tensorial (CUDA/MPS) sin ejecutar cálculos.

* **Ejemplo:**

        iobsolve check

---

## 🧪 Pruebas de Software Científico (Reproducibilidad)

IOB-Solve cuenta con un framework de pruebas automatizado que certifica su solidez. Para ejecutar la validación completa:

    pytest tests/ -v

* **`tests/continuous/`:** Valida que el motor converja exactamente sobre las raíces analíticas de las variedades, solucionando matemáticamente el solapamiento de frontera (*Boundary Overlap*).
* **`tests/discrete/`:** Audita que el Laplaciano distinga entre isometría, flujos asimétricos hostiles (ataques a redes) y colapso modal en inteligencia artificial.
* **`tests/benchmarks/`:** Confirma el rendimiento empírico bajo estrés masivo (ej: procesamiento topológico en grafos de 100,000 nodos en fracciones de segundo).
* **`tests/vis/`:** Validaciones en modo *headless* para la renderización de espectros, aptas para canales de CI/CD.

---

## 📜 Publicaciones y Respaldo Teórico

El fundamento matemático de este framework está detallado en los siguientes documentos de investigación provistos en el directorio `/paper`:

1.  **Knuttzen, J. (2026).** *"Formalismo de Integridad de Bisagra: Aislamiento Topológico de Singularidades y Control de Bifurcaciones en Variedades Continuas"*.
2.  **Knuttzen, J. (2026).** *"Formalismo de Integridad de Bisagra Discreto: Laplacianos de Flujo y Cirugía Topológica en Grafos"*.