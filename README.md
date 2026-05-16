# IOB-Solve: Topological Integrity Operator Framework (v0.2.0-beta)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20016070.svg)](https://doi.org/10.5281/zenodo.20016070)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Typing: Strict](https://img.shields.io/badge/typing-strict-green.svg)](https://microsoft.github.io/pyright/)

**IOB-Solve** es un framework de análisis numérico de alto rendimiento basado en el **Operador de Integridad de Bisagra (IOB)**. Está diseñado para detectar, medir y neutralizar rupturas en la variedad geométrica de sistemas complejos *antes* de que ocurran catástrofes cinéticas, fallos estructurales o colapsos sistémicos.

A partir de la versión **0.2.0**, el motor *core* ha sido instrumentado en **PyTorch nativo**. Esto permite capitalizar la diferenciación automática (Autograd) y las operaciones con tensores dispersos en GPU. Esto otorga al sistema **Latencia Positiva** (alerta predictiva temprana) al evaluar el Laplaciano Geométrico instantáneo del espacio de fases sin requerir entrenamiento previo.

---

## 🔬 Arquitectura y Capacidades Core

El framework divide el análisis en dos grandes dominios geométricos para maximizar la precisión según la topología del sistema:

### 1. Dominio Continuo (`iobsolve.continuous`)
Diseñado para variedades euclidianas diferenciables y sistemas dinámicos.
* **Teorema del Flujo de Integridad:** Aísla singularidades y raíces de funciones hiper-trascendentales eludiendo la explosión asintótica de los métodos espectrales clásicos.
* **Muestreo de Importancia Geométrico (GIS):** Reduce la evaluación de sistemas de alta dimensionalidad a una complejidad $\mathcal{O}(M)$ donde $M \ll N$, permitiendo telemetría continua en milisegundos.

### 2. Dominio Discreto (`iobsolve.discrete`)
Optimizado para topologías de red ralas, grafos y series temporales asíncronas.
* **Cirugía Topológica $\mathcal{O}(k_i)$:** Extirpación lógica de anomalías basada en el grado de conectividad local, evitando el recálculo global de la matriz Laplaciana en tiempo polinomial.
* **Z-Score Topológico Robusto:** Estimadores estadísticos inmunes al efecto de enmascaramiento (*Masking Effect*) basados en la Mediana y la Desviación Absoluta (MAD).

### 🛠️ Auto-Tuning $L_{metric}$
El framework conserva su calibrador dinámico (`l_metric="auto"`). Esta función ajusta automáticamente la resolución óptica ($L$) aprendiendo la varianza del ruido basal del sistema durante una fase de "warm-up" inicial. Esto previene falsos positivos y adapta la sensibilidad del sensor a las condiciones específicas del entorno sin requerir *Magic Number Tuning* manual.

---

## 📦 Ecosistema de Plugins Aplicados

IOB-Solve es altamente modular. La v0.2.0 incluye los siguientes motores instrumentados sobre la arquitectura central:

### 🌊 Dynamics & Control (`plugins.continuous.dynamics`)
Supresión activa de bifurcaciones en fluidos no-lineales y sistemas caóticos (Ej. Lorenz-96). 
* Funciona como un sensor continuo que detecta el cizallamiento del espacio de fases, permitiendo instrumentar maniobras de control y pre-alertas de transiciones críticas con latencia positiva.

### 🧮 Advanced Math & Roots (`plugins.continuous.singularities`)
Localización topológica de raíces y polos en el plano complejo puro. 
* Implementa la partición **IOB-QuadTree** para aislar semillas hiper-trascendentales con precisiones del límite de hardware, y el mapeo **IOB-FFT** para resolver clústeres de altísima densidad topológica, evadiendo el caos fractal de los métodos iterativos tradicionales.

### 🛡️ Network Security (`plugins.discrete.network_shield`)
Prevención de ataques volumétricos y asimétricos (DDoS). 
* Trata el flujo de red asíncrono como una variedad topológica. Cuando las peticiones deforman el equilibrio baricéntrico de forma anómala, el motor aísla los tensores (IPs hostiles) en tiempo sub-milisegundo, manteniendo intacta la experiencia de los usuarios legítimos.

### 🧠 AI Stability (`plugins.discrete.mode_collapse`)
Auditoría de espacios latentes en arquitecturas de aprendizaje profundo. 
* Extrae la *Varianza de Regularidad Topológica* ($\sigma_{\mathcal{Q}}^2$) como un sensor predictivo de **Colapso Modal**, registrando la degeneración isométrica de los *embeddings* iteraciones antes de que la función de pérdida tradicional reporte el fracaso del gradiente.

---

## 🚀 Instalación y Uso

Para instalar el framework base junto a las herramientas de desarrollo, validación y visualización:

    git clone https://github.com/JoaKnut/iobsolve.git
    cd iobsolve
    pip install -e ".[dev,vis]"

### Ejemplo de Integración: DDoS Shield con Auto-Tuning (FastAPI)

    from fastapi import FastAPI
    from iobsolve.plugins.discrete.network_shield import IOBASGIMiddleware

    app = FastAPI()

    # El IOB aprenderá la latencia basal automáticamente (l_metric="auto")
    # durante los primeros ciclos antes de armar la Cirugía Topológica.
    app.add_middleware(
        IOBASGIMiddleware,
        alert_threshold=12.0,
        l_metric="auto",
        quarantine_sec=120
    )

---

## 🎛️ Interfaz de Línea de Comandos (CLI)

El CLI permite instrumentar los motores topológicos directamente desde la terminal para diagnósticos rápidos o automatización de pipelines.

**Verificación de entorno y aceleración GPU:**

    python cli.py check

**Localización de singularidades complejas (IOB-QuadTree):**

    python cli.py roots --radius 5.0 --depth 8

**Despliegue de Escudo Topológico Discreto (Simulación):**

    python cli.py shield --nodes 1000 --tau 3.0 --attack

**Auditoría de Colapso Modal en IA:**

    python cli.py audit --batch 128 --dim 256 --tau 0.85

---

## 🚧 Limitaciones Conocidas (Roadmap v0.3.0)

1. **Ambigüedad del Vacío Topológico:** El D-IOB identifica una red totalmente desconectada con la misma firma analítica que la perfección geométrica pura. En producción, esto exige un *watchdog* paralelo que supervise que la densidad de red global no caiga a cero.
2. **Aliasing Topológico:** En el dominio continuo, si el radio de muestreo $L$ supera la semidistancia mínima entre raíces adyacentes, el motor sufre de aliasing (promediando cuencas diferentes). El Auto-Tuning mitiga esto, pero requiere precaución en entornos extremadamente densos.
3. **Maldición Estocástica en Alta Dimensionalidad:** Al evaluar flujos continuos masivos, la integración de la hiperesfera exige cuadraturas de Monte Carlo. Esto inyecta varianza estocástica que retrasa la pre-alerta si no se calibra correctamente el Muestreo de Importancia (GIS).
4. **Cicatrización de Supernodos (Hubs):** La extirpación de nodos con grado de conectividad extremo en una matriz rala requiere recalcular el baricentro para miles de vértices dependientes, introduciendo picos de latencia computacional focalizados.

---

## 📜 Publicaciones y Teoría Matemática

Este framework es la implementación oficial en software de los axiomas presentados en la serie de investigaciones matemáticas de **Joaquín Knuttzen**:

* 📄 **[Paper I]** *Formalismo de Integridad de Bisagra: Aislamiento Topológico de Singularidades y Control de Bifurcaciones en Variedades Continuas* (2026).
* 📄 **[Paper II]** *Formalismo de Integridad de Bisagra Discreto: Laplacianos de Grafos, Detección de Anomalías Asíncronas y Colapsos en Redes Complejas* (2026).

Para indagar en las demostraciones matemáticas, equivalencias diferenciales y complejidades asintóticas detalladas, refiérase a los manuscritos ubicados en el directorio `/paper/` incluido en este repositorio.

---
*Desarrollado con rigor científico para la comunidad de Matemáticas Aplicadas y Ciencias de la Computación.*