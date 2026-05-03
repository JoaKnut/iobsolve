import logging
import numpy as np
import time

# Importamos las herramientas de nuestro plugin matemático
from iobsolve.plugins.math.singularities import (
    ComplexAnalyticSystem, 
    IOBQuadTreeLocator, 
    NewtonRefiner
)

# Configuración limpia de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("iobsolve.tests.math")

def transcendental_function(z: np.ndarray) -> np.ndarray:
    """
    Función de prueba: f(z) = exp(z^5 - 1) - 1
    Tiene exactamente 5 raíces complejas puras distribuidas en un círculo de radio 1.
    """
    return np.exp(z**5 - 1) - 1

def derivative_function(z: np.ndarray) -> np.ndarray:
    """
    Derivada analítica opcional: f'(z) = 5z^4 * exp(z^5 - 1)
    """
    return 5 * (z**4) * np.exp(z**5 - 1)

def test_root_finding_and_refinement():
    logger.info("=== INICIANDO TEST IOB-QUADTREE & NEWTON REFINER ===")
    
    # 1. Definimos el sistema
    system = ComplexAnalyticSystem(transcendental_function)
    
    # Opcional: Inyectamos el Jacobiano (la derivada analítica) para acelerar Newton
    # Si lo comentas, el refiner usará diferencias finitas (Caja Negra) y funcionará igual.
    system.evaluate_jacobian = lambda state, direction: derivative_function(state) * direction

    # 2. Configuramos el IOB-QuadTree
    # L_metric pequeño para no superponer curvaturas
    # max_depth=6 nos dará cuadrantes lo suficientemente pequeños para ser buenas semillas
    locator = IOBQuadTreeLocator(L_metric=0.01, max_depth=6)

    # 3. Localización Topológica (La fase de "fuerza bruta inteligente")
    dominio = (-2.0, 2.0, -2.0, 2.0) # Buscamos en un cuadrado de 4x4
    
    logger.info(f"Escaneando el perímetro del dominio {dominio} mediante el Teorema del Flujo...")
    start_time = time.perf_counter()
    
    semillas_aproximadas = locator.locate(system, dominio)
    
    # Filtramos duplicados cercanos (el QuadTree a veces devuelve bordes compartidos)
    semillas_unicas = []
    for s in semillas_aproximadas:
        if not any(np.abs(s - u) < 0.1 for u in semillas_unicas):
            semillas_unicas.append(s)

    quadtree_time = (time.perf_counter() - start_time) * 1000
    logger.info(f"QuadTree encontró {len(semillas_unicas)} cuencas de atracción en {quadtree_time:.2f} ms.")

    for i, seed in enumerate(semillas_unicas):
        logger.info(f"  Semilla Cruda {i+1}: {seed.real:.6f} + {seed.imag:.6f}j")

    # 4. Refinamiento Newtoniano (Precisión de Hardware de 16 decimales)
    logger.info("\nIniciando Refinamiento Newtoniano (Convergencia Cuadrática)...")
    start_time = time.perf_counter()
    
    raices_perfectas = NewtonRefiner.refine(system, semillas_unicas, tol=1e-15)
    
    newton_time = (time.perf_counter() - start_time) * 1000
    logger.info(f"Refinamiento completado en {newton_time:.2f} ms.\n")

    # 5. Resultados y Validación de Error
    logger.info("=== RESULTADOS FINALES (15 Decimales) ===")
    for i, root in enumerate(raices_perfectas):
        # Evaluamos f(z) en la raíz hallada para ver el error real
        error = np.abs(system.evaluate(np.array([root]))[0])
        logger.info(f"  Raíz {i+1}: {root.real:+.15f} {root.imag:+.15f}j  (Error: {error:.2e})")

if __name__ == "__main__":
    test_root_finding_and_refinement()