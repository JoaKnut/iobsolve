import torch
import matplotlib
# Configurar matplotlib en modo "headless" estricto para que no busque una pantalla
matplotlib.use('Agg') 

import pytest
from iobsolve.io.visualizers import plot_roots_spectrum, plot_shield_surgery, plot_audit_variance

def test_plot_roots_spectrum_generation(tmp_path):
    """Verifica que el motor continuo pueda renderizar el mapa de hipercubos."""
    salida_png = tmp_path / "raices.png"
    
    # Simulamos raíces encontradas (lista de dominios límite)
    roots_mock = [
        ((-1.0, 0.0), (1.0, 2.0)),
        ((3.0, 4.0), (-2.0, -1.0))
    ]
    radio_busqueda = 5.0
    
    plot_roots_spectrum(roots_mock, radio_busqueda, str(salida_png))
    
    assert salida_png.exists(), "El archivo PNG no fue generado."
    assert salida_png.stat().st_size > 1024, "El archivo PNG parece estar vacío o corrupto."

def test_plot_shield_surgery_generation(tmp_path):
    """Verifica que el grafo discreto se renderice resaltando el nodo extirpado."""
    salida_png = tmp_path / "cirugia.png"
    N = 10
    
    # Grafo estrella simple
    adj = torch.zeros((N, N))
    adj[0, 1:] = 1.0
    adj[1:, 0] = 1.0
    adj_sparse = adj.to_sparse_coo()
    
    # Simulamos que el nodo 0 fue alertado/extirpado
    alertas_mock = torch.zeros(N, dtype=torch.bool)
    alertas_mock[0] = True
    
    plot_shield_surgery(adj_sparse, alertas_mock, str(salida_png))
    
    assert salida_png.exists()
    assert salida_png.stat().st_size > 1024

def test_plot_audit_variance_generation(tmp_path):
    """Verifica el renderizado de la reducción de dimensionalidad (PCA) del espacio latente."""
    salida_png = tmp_path / "auditoria_ia.png"
    
    # Tensor de embeddings de prueba (Batch: 64, Dim: 128)
    embeddings_mock = torch.randn(64, 128)
    is_collapsing = False
    
    plot_audit_variance(embeddings_mock, is_collapsing, str(salida_png))
    
    assert salida_png.exists()
    assert salida_png.stat().st_size > 1024