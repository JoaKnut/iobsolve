import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def plot_hovmoller(space_data: np.ndarray, 
                   times: np.ndarray, 
                   title: str = "Diagrama de Hovmöller: Dinámica Espacio-Temporal",
                   alert_time: Optional[float] = None) -> None:
    """
    Renderiza un diagrama de Hovmöller para la evolución temporal en alta dimensión.

    Parameters
    ----------
    space_data : np.ndarray
        Matriz de forma (n_time_steps, n_spatial_dimensions).
    times : np.ndarray
        Array 1D de coordenadas temporales.
    title : str, optional
        Título del gráfico.
    alert_time : float, optional
        Marca de tiempo para superponer un indicador vertical de alerta del IOB.
    """
    if space_data.shape[0] != times.shape[0]:
        raise ValueError("Las filas de space_data deben coincidir con la longitud de times.")
        
    fig, ax = plt.subplots(figsize=(12, 6))
    extent_tuple = (float(times[0]), float(times[-1]), 0.0, float(space_data.shape[1]))
    
    # Transposición para alinear Espacio (Y) vs Tiempo (X)
    im = ax.imshow(
        space_data.T, 
        aspect='auto', 
        origin='lower',
        extent=extent_tuple,
        cmap='twilight_shifted'
    )
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Tiempo (t)')
    ax.set_ylabel('Índice Espacial (i)')
    
    if alert_time is not None:
        ax.axvline(x=alert_time, color='magenta', linestyle='--', linewidth=2, label='Pre-Alerta IOB')
        ax.scatter(
            [alert_time], [space_data.shape[1] // 2], 
            color='magenta', marker='*', s=200, edgecolor='black', zorder=5
        )
        ax.legend(loc='upper right')
        
    fig.colorbar(im, ax=ax, label='Amplitud de Estado')
    plt.tight_layout()
    plt.show()

def plot_early_warning_telemetry(times: np.ndarray, 
                                 psi_c: np.ndarray, 
                                 threshold: float,
                                 system_energy: Optional[np.ndarray] = None) -> None:
    """
    Grafica el Funcional de Estrés Topológico (Psi_c) junto a observables macroscópicos.

    Parameters
    ----------
    times : np.ndarray
        Pasos de integración temporal.
    psi_c : np.ndarray
        Evolución temporal de la aceleración de estrés topológico.
    threshold : float
        Límite crítico para la ruptura sistémica.
    system_energy : np.ndarray, optional
        Norma L2 o energía cinética del sistema para evidenciar la latencia predictiva.
    """
    fig, ax1 = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#111111')
    ax1.set_facecolor('#111111')
    
    ax1.plot(times, psi_c, color='cyan', label=r'Sensor IOB $\Psi_c(t)$')
    ax1.axhline(y=threshold, color='red', linestyle=':', label='Umbral Crítico')
    
    ax1.set_xlabel('Tiempo (t)', color='white')
    ax1.set_ylabel(r'Aceleración de Estrés ($\Psi_c$)', color='cyan')
    
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', labelcolor='cyan', colors='white')
    
    for spine in ax1.spines.values():
        spine.set_color('#444444')

    if system_energy is not None:
        ax2 = ax1.twinx()
        ax2.plot(times, system_energy, color='orange', alpha=0.7, label='Energía del Sistema (L2)')
        ax2.set_ylabel('Observable Macroscópico', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange', colors='white')
        for spine in ax2.spines.values():
            spine.set_color('#444444')
            
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), facecolor='#222222', edgecolor='none', labelcolor='white')
    plt.title("Telemetría de Integridad Topológica", color='white', fontweight='bold')
    plt.tight_layout()
    plt.show()