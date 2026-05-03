import numpy as np
import matplotlib.pyplot as plt
import logging

from iobsolve.plugins.finance.market import TopologicalMarketMonitor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("iobsolve.tests.finance")

def test_flash_crash():
    # Semilla para reproducibilidad
    np.random.seed(42)
    n_assets = 100
    n_steps = 200
    
    # Parámetros del mercado base (Movimiento Browniano)
    mu = 0.0002      # Deriva (crecimiento ligero)
    sigma = 0.008    # Volatilidad base
    
    prices = np.zeros((n_steps, n_assets))
    prices[0] = 100.0 # Precio inicial paritario
    
    # ===================================================================
    # 1. GENERACIÓN DEL MERCADO ESTOCÁSTICO
    # ===================================================================
    for t in range(1, n_steps):
        if 120 <= t < 140:
            # SHOCK SISTÉMICO (Flash Crash)
            # Deriva negativa y alta volatilidad correlacionada
            crash_mu = -0.04
            crash_sigma = 0.04
            # El "pánico" (Z) afecta a todos los activos en la misma dirección
            Z = np.random.normal(loc=-1.0, scale=0.5, size=n_assets)
            prices[t] = prices[t-1] * np.exp((crash_mu - 0.5 * crash_sigma**2) + crash_sigma * Z)
        else:
            # MERCADO NORMAL (Ruido blanco independiente)
            Z = np.random.randn(n_assets)
            prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * Z)

    # ===================================================================
    # 2. TELEMETRÍA IOB DE ALTA FRECUENCIA
    # ===================================================================
    # Usamos un l_metric pequeño para capturar micro-deformaciones
    monitor = TopologicalMarketMonitor(n_assets=n_assets, alert_threshold=6.0, l_metric=0.01)
    
    psi_c_history = np.zeros(n_steps)
    crisis_flags = np.zeros(n_steps, dtype=bool)
    
    logger.info("Iniciando escáner topológico de microestructura...")
    
    for t in range(n_steps):
        is_crisis, psi_c, q_d = monitor.push_market_state(prices[t])
        psi_c_history[t] = psi_c
        crisis_flags[t] = is_crisis
        
        # Imprimir alerta solo en el milisegundo en que se rompe la métrica
        if is_crisis and not crisis_flags[t-1]:
            logger.warning(f"⚠️ [ALERTA IOB] Colapso de liquidez inminente en el tick {t} (Psi_c = {psi_c:.2f})")

    # ===================================================================
    # 3. VISUALIZACIÓN
    # ===================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.patch.set_facecolor('#111111')
    
    # --- Gráfico Superior: Evolución de Precios ---
    ax1.set_facecolor('#111111')
    # Graficamos solo 15 activos de fondo para no saturar la vista
    for i in range(15): 
        ax1.plot(prices[:, i], alpha=0.3, color='gray')
    
    # Índice de Mercado (Promedio)
    ax1.plot(np.mean(prices, axis=1), color='white', linewidth=2, label="Índice Promedio del Mercado")
    ax1.axvspan(120, 140, color='red', alpha=0.15, label="Cisne Negro (Flash Crash)")
    
    ax1.set_title("Evolución Dinámica: Microestructura de 100 Activos", color='white')
    ax1.set_ylabel("Valor Relativo", color='white')
    ax1.legend()
    ax1.tick_params(colors='white')
    
    # --- Gráfico Inferior: Sensor de Estrés IOB ---
    ax2.set_facecolor('#111111')
    ax2.plot(psi_c_history, color='cyan', label=r"Sensor Geométrico $\Psi_c(t)$")
    ax2.axhline(6.0, color='magenta', linestyle=':', label="Umbral de Alerta Sistémica")
    
    # Marcar los puntos críticos
    crisis_indices = np.where(crisis_flags)[0]
    ax2.scatter(crisis_indices, psi_c_history[crisis_indices], color='red', s=20, zorder=5)
    
    ax2.set_title("Telemetría IOB: Deformación del Espacio de Liquidez", color='white')
    ax2.set_xlabel("Tiempo Discreto (Ticks)", color='white')
    ax2.set_ylabel(r"Aceleración de Estrés ($\Psi_c$)", color='white')
    ax2.legend()
    ax2.tick_params(colors='white')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_flash_crash()