# Primeros Pasos

## Instalación

### Central (solo PyTorch)

```bash
pip install iobsolve
```

Instala únicamente el motor matemático con una sola dependencia: `torch >= 2.0`.
No se requieren librerías de visualización para la API Python ni para los comandos matemáticos de la CLI.

### Con Visualización

La bandera `--plot` y la E/S de `.npy` / `.graphml` requieren dependencias opcionales:

```bash
pip install "iobsolve[vis]"
```

Incluye: `numpy`, `matplotlib`, `seaborn`, `scipy`, `networkx`, `scikit-learn`.

### Entorno de Desarrollo

```bash
git clone https://github.com/JoaKnut/iobsolve.git
cd iobsolve
pip install -e ".[dev,vis]"
```

Agrega: `pytest`, `pytest-cov`, `mypy`, `pyright`, `ruff`.

## Requisitos de Hardware

| Componente | Mínimo |
|------------|--------|
| Python     | ≥ 3.10 |
| PyTorch    | ≥ 2.0  |
| RAM        | 4 GB (16 GB para auditorías a escala LLM) |
| GPU        | Opcional — pase `device="cuda"` al `FlowTheoremLocator` |

## Primera Ejecución

Verifique su instalación:

```bash
iobsolve check
```

Salida esperada:

```
============================================================
[*] IOB-Solve v0.2.0 | Diagnóstico del Entorno IOB-Solve
============================================================
  Python  : 3.12.x
  PyTorch : 2.x.x
  CUDA    : No disponible
  NumPy   : 1.x.x
  Matplotlib: 3.x.x

[+] Entorno verificado correctamente.
```

## Su Primera Localización de Raíces

```bash
iobsolve roots --radius 10 --depth 8
```

Ejecuta la `TranscendentalManifold` predeterminada — un campo vectorial 2D con raíces analíticas en $(n\pi, 0)$ — e imprime los centroides estimados de cada singularidad detectada.

Agregue `--plot raices.png` para renderizar un gráfico de dispersión de las regiones detectadas.
