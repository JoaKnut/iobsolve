import numpy as np
import logging
from iobsolve.plugins.dynamics.continuous import ContinuousDynamicSystem, DynamicsCrisisMonitor
from iobsolve.utils.visualize import plot_hovmoller, plot_early_warning_telemetry

# Configuración de logs para capturar las alertas de crisis topológicas
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iobsolve.tests")

def lorenz96_vector_field(t: float, x: np.ndarray, F: float = 8.0) -> np.ndarray:
    """
    Implementación del campo vectorial de Lorenz-96 para N variables.
    La dinámica se rige por la ecuación diferencial:
    $$ \frac{dx_i}{dt} = (x_{i+1} - x_{i-2})x_{i-1} - x_i + F $$
    """
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        # Aplicación de condiciones de contorno periódicas
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt

def test_lorenz96_benchmark():
    """
    Reproducción de la validación empírica para sistemas dinámicos continuos.
    Valida la capacidad anticipatoria del funcional de estrés métrico Psi_c(t).[cite: 1]
    """
    # 1. Definición del Sistema (Régimen caótico F=8, N=100 dimensiones)[cite: 1]
    N = 100
    F = 8.0
    lorenz_sys = ContinuousDynamicSystem(lambda t, x: lorenz96_vector_field(t, x, F), dim=N)

    # 2. Configuración del Monitor IOB[cite: 1]
    # Se utiliza un radio métrico L=0.1 y activación del sampler GIS.[cite: 1]
    # El GIS permite reducir drásticamente la varianza estocástica en el muestreo.[cite: 1]
    monitor = DynamicsCrisisMonitor(
        system=lorenz_sys, 
        alert_threshold=5.0, 
        l_metric=0.1, 
        use_gis=True
    )

    # 3. Estado Inicial: Equilibrio forzado con una perturbación central[cite: 1]
    # Emula el inicio de la inestabilidad climática o estructural.[cite: 1]
    x0 = np.full(N, F)
    x0[N // 2] += 0.01 

    # 4. Simulación y Monitoreo Lagrangiano[cite: 1]
    # Se emplean solo 6 muestras direccionales (M=6) para optimizar el tiempo de CPU.[cite: 1]
    logger.info(f"Iniciando telemetría IOB sobre Lorenz-96 (N={N}). Análisis en tiempo real...")
    times, trajectory, psi_c = monitor.simulate_and_monitor(
        initial_state=x0,
        t_span=(0, 20),
        dt=0.05,
        samples_per_step=6
    )

    # 5. Análisis de Resultados y Detección de Latencia Positiva[cite: 1]
    # Identificamos el primer instante donde Psi_c supera el umbral de alerta[cite: 1]
    alert_mask = psi_c > 5.0
    t_alert = times[alert_mask][0] if np.any(alert_mask) else None
    
    if t_alert:
        logger.warning(f"¡PRE-ALERTA IOB DETECTADA! Inestabilidad topológica en t={t_alert:.3f}")

    # Visualización 1: Diagrama de Hovmöller (Evolución espacio-temporal)[cite: 1]
    plot_hovmoller(
        space_data=trajectory, 
        times=times, 
        title=f"Benchmark IOB: Evolución del Fluido Lorenz-96 (N={N})",
        alert_time=t_alert
    )

    # Visualización 2: Telemetría de Alerta Temprana (Psi_c vs Energía Cinética)[cite: 1]
    # La energía se calcula como la norma L2 del estado del sistema.[cite: 1]
    energy = np.linalg.norm(trajectory, axis=1)
    plot_early_warning_telemetry(
        times=times, 
        psi_c=psi_c, 
        threshold=5.0, 
        system_energy=energy
    )

if __name__ == "__main__":
    test_lorenz96_benchmark()