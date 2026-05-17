import torch
from iobsolve.plugins.continuous.dynamics import Lorenz96System

# =====================================================================
# PRUEBAS DEL PLUGIN DINÁMICO (LORENZ-96)
# =====================================================================

def test_lorenz96_equilibrio_analitico():
    """
    Verdad Base: En el atractor de Lorenz-96, si todos los estados x_i
    toman el valor de la constante de forzamiento F, la derivada del 
    sistema es exactamente 0 (Punto Fijo / Equilibrio).
    """
    F = 8.0
    lorenz = Lorenz96System(forcing_constant=F)
    
    # Creamos un tensor de 40 dimensiones (estándar climático) inicializado en F
    estado_equilibrio = torch.full((40,), F, dtype=torch.float64)
    
    # Evaluamos el flujo
    flujo_derivativo = lorenz(t=0.0, state_tensor=estado_equilibrio)
    
    # ASERCIÓN: El flujo debe ser exactamente un vector de ceros
    vector_nulo = torch.zeros_like(estado_equilibrio)
    assert torch.allclose(flujo_derivativo, vector_nulo, atol=1e-12), "Fallo matemático: El estado de equilibrio analítico no retorna 0."

def test_lorenz96_preservacion_tensorial():
    """
    Verifica que la vectorización basada en `torch.roll` mantenga intacta 
    la dimensionalidad del tensor, sin importar si inyectamos un vector simple 
    o un batch multidimensional.
    """
    lorenz = Lorenz96System(forcing_constant=5.0)
    
    # Caso 1: Vector 1D (ej. 1 sistema de 10 dimensiones)
    estado_1d = torch.randn(10, dtype=torch.float64)
    flujo_1d = lorenz(0.0, estado_1d)
    assert flujo_1d.shape == (10,), "La dimensión colapsó en la evaluación 1D."
    
    # Caso 2: Batch 2D (ej. 100 simulaciones simultáneas de 40 dimensiones)
    estado_2d = torch.randn(100, 40, dtype=torch.float64)
    flujo_2d = lorenz(0.0, estado_2d)
    assert flujo_2d.shape == (100, 40), "Fallo en la vectorización por lotes (Batch processing)."