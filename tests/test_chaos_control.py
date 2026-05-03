import numpy as np
import logging
import matplotlib.pyplot as plt

from iobsolve.plugins.dynamics.continuous import ContinuousDynamicSystem, DynamicsCrisisMonitor
from iobsolve.plugins.control.sys_control import TopologicalSurgeon

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("iobsolve.tests.control")

def lorenz96_vector_field(t: float, x: np.ndarray, F: float = 8.0) -> np.ndarray:
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt

def test_topological_chaos_control():
    N = 100
    F = 8.0
    sys = ContinuousDynamicSystem(lambda t, x: lorenz96_vector_field(t, x, F), dim=N)

    # Estado Inicial inestable
    x0 = np.full(N, F)
    x0[N // 2] += 0.01 
    t_span = (0, 20)
    dt = 0.05

    # ===================================================================
    # UNIVERSO 1: SISTEMA LIBRE (Colapso inevitable)
    # ===================================================================
    logger.info("-> Simulando Universo 1 (Sin Control)...")
    monitor = DynamicsCrisisMonitor(sys, alert_threshold=5.0, l_metric=0.1, use_gis=True)
    times_u1, traj_u1, psi_c_u1 = monitor.simulate_and_monitor(x0, t_span, dt=dt)
    energy_u1 = np.linalg.norm(traj_u1, axis=1)

    # ===================================================================
    # UNIVERSO 2: SISTEMA CONTROLADO POR IOB
    # ===================================================================
    logger.info("-> Simulando Universo 2 (Cirugía Topológica Activa)...")
    # control_gain=0.15 indica cuánta "fricción" inyectar por cada punto de estrés métrico extra
    surgeon = TopologicalSurgeon(sys, alert_threshold=5.0, control_gain=0.15, l_metric=0.1, use_gis=True)
    times_u2, traj_u2, psi_c_u2, ctrl_signal = surgeon.simulate_and_control(x0, t_span, dt=dt)
    energy_u2 = np.linalg.norm(traj_u2, axis=1)

    # ===================================================================
    # RESULTADOS VISUALES
    # ===================================================================
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor('#111111')

    # 1. Energía de ambos universos
    ax1.set_facecolor('#111111')
    ax1.plot(times_u1, energy_u1, color='red', alpha=0.8, label="Energía (Sin Control) - Colapso Caótico")
    ax1.plot(times_u2, energy_u2, color='lime', linewidth=2, label="Energía (IOB Control) - Estabilizado")
    ax1.set_title("Comparativa Fïsica: Supresión de la Bifurcación", color='white')
    ax1.set_ylabel("Energía L2", color='white')
    ax1.legend()
    ax1.tick_params(colors='white')

    # 2. Telemetría IOB del Sistema Controlado
    ax2.set_facecolor('#111111')
    ax2.plot(times_u2, psi_c_u2, color='cyan', label=r"Sensor $\Psi_c(t)$ (Sistema Controlado)")
    ax2.axhline(5.0, color='magenta', linestyle=':', label="Umbral de Alerta")
    ax2.set_title("Tensor Geométrico del Sistema Controlado", color='white')
    ax2.set_ylabel(r"Aceleración de Estrés ($\Psi_c$)", color='white')
    ax2.legend()
    ax2.tick_params(colors='white')

    # 3. Señal de Control Inyectada
    ax3.set_facecolor('#111111')
    ax3.fill_between(times_u2, 0, ctrl_signal, color='yellow', alpha=0.3, label=r"Fuerza Correctora $\mathbf{U}(t)$")
    ax3.plot(times_u2, ctrl_signal, color='yellow')
    ax3.set_title("Intervenciones Quirúrgicas del Controlador", color='white')
    ax3.set_ylabel("Fricción Inducida", color='white')
    ax3.set_xlabel("Tiempo (t)", color='white')
    ax3.legend()
    ax3.tick_params(colors='white')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_topological_chaos_control()