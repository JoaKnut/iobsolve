import pytest
import warnings
import matplotlib

# Deshabilitar interfaz gráfica para toda la suite de pruebas
matplotlib.use('Agg')

@pytest.fixture(autouse=True)
def silence_pytorch_warnings():
    """Silencia warnings irrelevantes de invariantes dispersos en toda la suite."""
    warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")