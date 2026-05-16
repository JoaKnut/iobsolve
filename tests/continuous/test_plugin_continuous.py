r"""
Suite de Pruebas Unitarias para el Dominio Continuo.

Verifica la consistencia analítica del IOB Continuo, garantizando 
la conservación de la medida euclidiana y la correcta evaluación del 
residuo hiper-trascendental en campos diferenciables. Certifica que 
el Teorema del Flujo de Integridad aísla las raíces asintóticas sin 
recurrir a la matriz Jacobiana.

References
----------
.. [1] Knuttzen, J. (2026). "Formalismo de Integridad de Bisagra: Aislamiento 
       Topológico de Singularidades y Control de Bifurcaciones en Variedades Continuas".
"""

import torch
from iobsolve.core.space import EuclideanManifold
from iobsolve.continuous.hinge import ContinuousIntegrityOperator
from iobsolve.plugins.continuous.dynamics import Lorenz96System
import math
from iobsolve.continuous.flow_theorem import FlowTheoremLocator
from iobsolve.plugins.continuous.singularities import TranscendentalManifold

def test_lorenz_continuous_operator():
    r"""
    Verifica que el Operador de Integridad Continuo procese correctamente 
    el campo vectorial del Atractor de Lorenz-96 sin arrojar errores dimensionales.

    Notes
    -----
    Garantía 1: Sensibilidad Geométrica en Regímenes Caóticos.
    Asegura que frente a un flujo dinámico altamente inestable (F=8.0), el operador 
    detecte invariablemente una curvatura escalar \mathcal{H}(x) > 0.0, acotada 
    estrictamente en el intervalo normalizado [0, 1].
    """
    num_variables = 40
    steps = 50 
    dt = 0.01
    
    manifold = EuclideanManifold(shape=(num_variables,), grid_spacing=1.0)
    lorenz = Lorenz96System(forcing_constant=8.0)
    operator = ContinuousIntegrityOperator()

    # Precisión de 64 bits garantizada para el tensor de estado dinámico
    state = torch.ones(num_variables, dtype=torch.float64) * 8.0
    state[19] += 0.01

    history_stress = torch.zeros((steps, num_variables), dtype=torch.float64)

    for t in range(steps):
        vector_field = lorenz(t * dt, state)
        stress = operator.compute_stress(
            state_tensor=vector_field, 
            manifold=manifold, 
            normalize=True
        )
        history_stress[t, :] = stress
        state = state + vector_field * dt

    # Aserciones Matemáticas
    assert history_stress.shape == (steps, num_variables), "El historial de estrés perdió isomorfismo dimensional."
    
    # Dado que estamos en un régimen caótico (F=8.0) y hay perturbación, 
    # garantizamos que el operador detectó curvatura (estrés relativo > 0)
    max_stress_recorded = torch.max(history_stress).item()
    assert max_stress_recorded > 0.0, "El IOB Continuo computó estrés nulo en un campo dinámico caótico."
    
    # Al estar normalizado, no debe superar 1.0
    assert max_stress_recorded <= 1.0, "Fallo en la normalización topológica (estrés > 1.0)."

def test_flow_theorem_singularities():
    r"""
    Verifica que el motor QuadTree acoplado con IOB-FFT pueda aislar
    las raíces de un sistema hiper-trascendental con múltiples singularidades,
    eludiendo la evaluación tradicional del Jacobiano.

    Notes
    -----
    Garantía 2: Convergencia de Bisección Asintótica.
    Valida la extracción de atractores puntuales (raíces en n\pi) comprobando 
    que los hipercubos terminales \Omega_k resultantes de la bisección de fase 
    encapsulan analíticamente las coordenadas topológicas esperadas.
    """
    system = TranscendentalManifold()
    
    # Dominio inicial maestro: x entre [-4, 4], y entre [-1, 1]
    # Contiene al menos tres raíces analíticas: (-pi, 0), (0, 0) y (pi, 0)
    initial_domain = ((-4.0, 4.0), (-1.0, 1.0))
    
    # Instanciamos el localizador del Teorema del Flujo
    locator = FlowTheoremLocator(
        system_equation=system,
        grid_resolution=16,          # Resolución de la malla para la FFT en cada hipercubo
        spectral_threshold=0.05,     # Umbral de dispersión espectral
        device="cpu"                 # En tests lo mantenemos en CPU
    )
    
    # Ejecutamos la bisección espacial recursiva (QuadTree)
    # Una profundidad de 5 dividirá el espacio hasta en 4^5 = 1024 sub-regiones
    singular_hypercubes = locator.locate_roots(
        initial_domain=initial_domain, 
        time_t=0.0, 
        max_depth=5
    )
    
    # 1. Aserción de Existencia: El motor debe haber encontrado singularidades
    assert len(singular_hypercubes) > 0, "El FlowTheoremLocator no encontró ninguna raíz en un espacio altamente singular."
    
    # 2. Aserción Analítica: Verificamos si la raíz central exacta en (0, 0) 
    # quedó atrapada dentro de los límites de alguno de los hipercubos terminales.
    root_0_found = False
    root_pi_found = False
    
    for (x_min, x_max), (y_min, y_max) in singular_hypercubes:
        if x_min <= 0.0 <= x_max and y_min <= 0.0 <= y_max:
            root_0_found = True
        if x_min <= math.pi <= x_max and y_min <= 0.0 <= y_max:
            root_pi_found = True
            
    assert root_0_found, "Singularidad Eludida: La raíz central en (0.0, 0.0) no fue aislada."
    assert root_pi_found, r"Singularidad Eludida: La raíz en (\pi, 0) no fue aislada."