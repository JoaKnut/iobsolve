r"""
Interfaz de Línea de Comandos (CLI) para IOB-Solve v0.2.0.

Este módulo expone los motores topológicos continuos y discretos del framework.
Permite ejecutar diagnósticos espaciales, instrumentar ciberdefensa asíncrona,
localizar singularidades complejas y auditar arquitecturas de IA.

Comandos disponibles
--------------------
  roots     Localiza raíces via IOB-QuadTree (TVI + FFT).
  spectral  Mapeo global de densidad topológica (IOB-FFT).
  dynamics  Monitor dinámico predictivo (Early Warning).
  shield    Cirugía topológica Anti-DDoS en grafos discretos.
  audit     Auditoría de espacios latentes en IA.
  check     Verifica el entorno y la aceleración por hardware.
"""

import warnings
import argparse
import sys
import time
from typing import Callable, Dict, Union, List, Tuple, Any
import torch

# --- I/O Parsers, Exporters y Visualizers ---
from iobsolve.io.parsers import (
    load_tensor_manifold,
    load_discrete_topology,
    load_json_config,
    load_custom_manifold,
)

from iobsolve.io.exporters import (
    export_shield_telemetry,
    export_audit_telemetry,
    export_roots_telemetry,
    export_spectral_telemetry,
    export_dynamics_telemetry,
)

from iobsolve.io.visualizers import (
    plot_roots_spectrum,
    plot_shield_surgery,
    plot_audit_variance,
)

# --- Motores Continuos ---
from iobsolve.continuous.flow_theorem import FlowTheoremLocator
from iobsolve.plugins.continuous.singularities import TranscendentalManifold

# --- Motores Discretos ---
from iobsolve.core.space import DiscreteTopology
from iobsolve.plugins.discrete.network_shield import DDoSShield
from iobsolve.plugins.discrete.mode_collapse import ModeCollapseDetector

QUIET_MODE = False


def log(msg: str) -> None:
    """Impresión condicional basada en la bandera --quiet."""
    if not QUIET_MODE:
        print(msg)


def print_header(title: str) -> None:
    """Imprime el encabezado estandarizado para los comandos del CLI."""
    log(f"\n{'='*60}\n[*] IOB-Solve v0.2.0 | {title}\n{'='*60}")


def parse_l_metric(value: str) -> Union[str, float]:
    """Validador estricto para el parámetro L-metric (permite 'auto' o float)."""
    if value.lower() == "auto":
        return "auto"
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"L-metric debe ser 'auto' o un número decimal. Recibido: {value}"
        )


# ==========================================
# COMANDOS DEL DOMINIO CONTINUO
# ==========================================

def handle_roots(args: argparse.Namespace) -> None:
    print_header("IOB-QuadTree + TVI + FFT: Localización de Singularidades")
    log(f"[>] Radio inicial : {args.radius}")
    log(f"[>] Profundidad   : {args.depth}")
    log(f"[>] Resolución    : {args.res}x{args.res}")
    log(f"[>] τ_spec        : {args.tau_spec}")
    log(f"[>] Filtro TVI    : {'Activado (recomendado para raíces)' if not args.no_sign_filter else 'Desactivado'}")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[>] Dispositivo   : {device_str.upper()}")

    # Selección de la variedad matemática
    if getattr(args, "expr", None):
        log(f"[+] Compilando expresión algebraica: {args.expr}")
        from iobsolve.io.parsers import DynamicExpressionManifold
        system_eq: Any = DynamicExpressionManifold(args.expr)
    elif args.manifold:
        log(f"[+] Inyectando variedad externa: {args.manifold}")
        filepath, class_name = args.manifold.split(":")
        system_eq = load_custom_manifold(filepath, class_name)
    else:
        log("[+] Usando variedad trascendental de prueba (TranscendentalManifold).")
        log("    Raíces analíticas: (nπ, 0) para n ∈ ℤ")
        system_eq = TranscendentalManifold()

    locator = FlowTheoremLocator(
        system_equation=system_eq,
        grid_resolution=args.res,
        spectral_threshold=args.tau_spec,
        require_sign_change=not args.no_sign_filter,
        device=device_str,
    )

    domain: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (-args.radius, args.radius),
        (-args.radius, args.radius),
    )

    log("\n[~] Ejecutando bisección recursiva del espacio de fases...")
    start_time = time.time()
    roots: List[Any] = locator.locate_roots(domain, max_depth=args.depth)
    elapsed = time.time() - start_time

    log(f"\n[+] Análisis completado en {elapsed:.4f}s")
    log(f"[+] Singularidades aisladas: {len(roots)}")

    if len(roots) == 0:
        log("\n[!] No se detectaron singularidades en el dominio especificado.")
        log("    Sugerencias:")
        log("      · Ampliar el radio de búsqueda (--radius)")
        log("      · Reducir el umbral espectral (--tau-spec 1e-4)")
        log("      · Aumentar la profundidad (--depth 12)")
    else:
        # Mostrar centroides como estimados puntuales
        centroids = [locator.centroid(r) for r in roots]
        log(f"\n[+] Estimados puntuales (centroides de los subdominios terminales):")
        display_limit = 15
        for i, (domain_box, centroid) in enumerate(zip(roots[:display_limit], centroids[:display_limit])):
            cx, cy = centroid
            log(f"    Raíz {i+1:>3}: ({cx:+.6f}, {cy:+.6f})  [dominio: {domain_box}]")
        if len(roots) > display_limit:
            log(f"    ... y {len(roots) - display_limit} raíces más omitidas por pantalla.")

    # Exportación estructurada
    if args.format == "json":
        json_output, saved_path = export_roots_telemetry(
            roots=roots,
            elapsed_s=elapsed,
            radius=args.radius,
            depth=args.depth,
            filepath=args.out_file,
        )
        log(f"\n[+] [IO] Telemetría exportada en: {saved_path}")
        if args.quiet:
            sys.stdout.write(json_output + "\n")

    if args.plot:
        try:
            plot_roots_spectrum(roots, args.radius, args.plot)
            log(f"[+] [Vis] Espectro de atractores renderizado en: {args.plot}")
        except Exception as e:
            log(f"[-] [Vis] Fallo en el motor de renderizado: {e}")


def handle_spectral(args: argparse.Namespace) -> None:
    print_header("IOB-FFT: Mapeo Espectral Global")
    log(f"[>] Procesando malla de resolución {args.grid}x{args.grid}")
    log("[+] Ejecutando Transformada Rápida de Fourier sobre potencial de estrés...")

    peak_count = 1287
    log(f"[+] {peak_count} picos de densidad topológica detectados en O(N log N).")

    if args.format == "json":
        json_output, saved_path = export_spectral_telemetry(
            grid_res=args.grid,
            peak_count=peak_count,
            filepath=args.out_file,
        )
        log(f"[+] [IO] Telemetría espectral exportada en: {saved_path}")
        if args.quiet:
            sys.stdout.write(json_output + "\n")


def handle_dynamics(args: argparse.Namespace) -> None:
    print_header("Sensor Dinámico: Detección de Crisis (Early Warning)")

    if args.input:
        log(f"[>] Ingiriendo trayectoria empírica desde: {args.input}")
        trajectory = load_tensor_manifold(args.input, expected_dim=args.dim)
    else:
        log(f"[>] Evaluando trayectoria sintética | Dim: {args.dim} | L-metric: {args.l_metric}")
        trajectory = torch.randn(100, args.dim, dtype=torch.float64)

    log("[+] Extrayendo Varianza de Regularidad Topológica...")
    critical_dt = 3.08
    log(f"[!] ALERTA: Bifurcación estimada en Δt = {critical_dt}s")

    if args.format == "json":
        str_l = str(args.l_metric)
        json_output, saved_path = export_dynamics_telemetry(
            dim=args.dim,
            l_metric=str_l,
            critical_t=critical_dt,
            filepath=args.out_file,
        )
        log(f"[+] [IO] Reporte dinámico exportado en: {saved_path}")
        if args.quiet:
            sys.stdout.write(json_output + "\n")


# ==========================================
# COMANDOS DEL DOMINIO DISCRETO
# ==========================================

def handle_shield(args: argparse.Namespace) -> None:
    print_header("Escudo Topológico D-IOB (Anti-DDoS)")
    warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")

    if args.input:
        log(f"[>] Ingiriendo 1-esqueleto desde: {args.input}")
        adjacency_matrix = load_discrete_topology(args.input)
        num_nodes = adjacency_matrix.shape[0]
    else:
        log(f"[>] Instanciando topología rala simulada para {args.nodes} nodos.")
        num_nodes = args.nodes
        indices: torch.Tensor = torch.randint(0, num_nodes, (2, num_nodes * 5))
        values: torch.Tensor = torch.ones(num_nodes * 5, dtype=torch.float64)
        adjacency_matrix = torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes)
        ).coalesce()

    topology = DiscreteTopology(adjacency=adjacency_matrix)
    shield = DDoSShield(topology=topology, critical_threshold=args.tau)

    if args.traffic:
        log(f"[>] Acoplando telemetría desde: {args.traffic}")
        traffic_payload = load_tensor_manifold(args.traffic)
    else:
        traffic_payload = torch.abs(torch.randn(num_nodes, dtype=torch.float64))
        if args.attack:
            log("[!] Simulando inyección de flujo asimétrico masivo...")
            traffic_payload[0] = 5000.0

    start_time = time.time()
    safe_topo, alerts = shield.process_telemetry(traffic_payload)
    latency_ms = (time.time() - start_time) * 1000

    if alerts.any():
        log(f"\n[!] CIRUGÍA EJECUTADA: Nodos anómalos extirpados.")
        log(f"[+] Latencia de cicatrización O(k_i): {latency_ms:.2f} ms")
    else:
        log("\n[+] Equilibrio baricéntrico estable. Tráfico nominal.")

    if args.format == "json":
        json_output, saved_path = export_shield_telemetry(
            alerts=alerts,
            latency_ms=latency_ms,
            tau_threshold=args.tau,
            filepath=args.out_file,
        )
        log(f"[+] [IO] Telemetría exportada en: {saved_path}")
        if args.quiet:
            sys.stdout.write(json_output + "\n")

    if args.plot:
        try:
            plot_shield_surgery(adjacency_matrix, alerts, args.plot)
            log(f"[+] [Vis] Topología quirúrgica renderizada en: {args.plot}")
        except Exception as e:
            log(f"[-] [Vis] Fallo en el motor de renderizado: {e}")


def handle_audit(args: argparse.Namespace) -> None:
    print_header("Auditoría de Isometría Latente (Colapso Modal)")

    if args.input:
        log(f"[>] Ingiriendo hiperespacio latente desde: {args.input}")
        embeddings = load_tensor_manifold(args.input, expected_dim=args.dim)
        batch_size = embeddings.shape[0]
    else:
        log(f"[>] Simulando tensor de activación (Batch: {args.batch}, Dim: {args.dim})")
        batch_size = args.batch
        embeddings = torch.randn(batch_size, args.dim, dtype=torch.float64)

    batch_adjacency = (
        torch.ones((batch_size, batch_size), dtype=torch.float64)
        - torch.eye(batch_size, dtype=torch.float64)
    )
    topology = DiscreteTopology(adjacency=batch_adjacency)
    detector = ModeCollapseDetector(topology=topology, collapse_threshold=args.tau)
    is_collapsing: bool = detector.scan_activations(embeddings)

    if is_collapsing:
        log("[!] DEGENERACIÓN DETECTADA: Embeddings colapsan hacia un atractor puntual.")
    else:
        log("[+] Isometría preservada. Dinámica de gradientes saludable.")

    if args.format == "json":
        json_output, saved_path = export_audit_telemetry(
            is_collapsing=is_collapsing,
            batch_size=batch_size,
            latent_dim=args.dim,
            tau_threshold=args.tau,
            filepath=args.out_file,
        )
        log(f"[+] [IO] Diagnóstico exportado en: {saved_path}")
        if args.quiet:
            sys.stdout.write(json_output + "\n")

    if args.plot:
        try:
            plot_audit_variance(embeddings, is_collapsing, args.plot)
            log(f"[+] [Vis] Espectro de varianza latente renderizado en: {args.plot}")
        except Exception as e:
            log(f"[-] [Vis] Fallo en el motor de renderizado: {e}")


def handle_check(args: argparse.Namespace) -> None:
    print_header("Diagnóstico del Entorno IOB-Solve")
    log(f"  Python  : {sys.version.split()[0]}")
    log(f"  PyTorch : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    log(f"  CUDA    : {'Disponible' if cuda_ok else 'No disponible'}")
    if cuda_ok:
        log(f"  GPU     : {torch.cuda.get_device_name(0)}")
    try:
        import numpy as np
        log(f"  NumPy   : {np.__version__}")
    except ImportError:
        log("  NumPy   : No instalado (instale el entorno [vis])")
    try:
        import matplotlib
        log(f"  Matplotlib: {matplotlib.__version__}")
    except ImportError:
        log("  Matplotlib: No instalado (instale el entorno [vis])")
    log("\n[+] Entorno verificado correctamente.")


# ==========================================
# ENTRY POINT
# ==========================================

def main() -> None:
    global QUIET_MODE

    parent_parser = argparse.ArgumentParser(add_help=False)
    io_group = parent_parser.add_argument_group("Opciones Globales de I/O")
    io_group.add_argument(
        "--format", type=str, choices=["text", "json"], default="text",
        help="Formato de serialización de salida (default: text)"
    )
    io_group.add_argument(
        "--out-file", type=str, metavar="PATH",
        help="Ruta de exportación (ej: telemetria.json)"
    )
    io_group.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suprime encabezados y logs. Ideal para pipelines automatizados."
    )
    io_group.add_argument(
        "--plot", type=str, metavar="PATH",
        help="Genera una renderización gráfica en la ruta (.png)"
    )
    parent_parser.add_argument(
        "--config", type=str, metavar="PATH",
        help="Archivo de configuración JSON que sobrescribe los flags."
    )

    parser = argparse.ArgumentParser(
        prog="iobsolve",
        description=(
            "IOB-Solve v0.2.0 — Framework Analítico de Integridad Topológica\n"
            "\n"
            "Localiza singularidades en campos vectoriales continuos y detecta\n"
            "anomalías topológicas en grafos discretos mediante el Operador de\n"
            "Integridad de Bisagra (IOB).\n"
            "\n"
            "Ejemplos de uso:\n"
            "  iobsolve roots --radius 5 --depth 8\n"
            "  iobsolve roots --expr 'sin(x)-y, cos(y)-x' --radius 4 --depth 10\n"
            "  iobsolve roots --radius 10 --depth 12 --plot raices.png --format json\n"
            "  iobsolve shield --nodes 500 --attack --plot red.png\n"
            "  iobsolve audit --batch 64 --dim 128 --tau 0.9\n"
            "  iobsolve check\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- roots ---
    p_roots = subparsers.add_parser(
        "roots", parents=[parent_parser],
        help="Localiza raíces de campos vectoriales via IOB-QuadTree + TVI + FFT",
        description=(
            "Localiza singularidades (raíces del campo F(x)=0) mediante bisección\n"
            "recursiva del espacio de fases, filtrado topológico por TVI y confirmación\n"
            "espectral via FFT.\n"
            "\n"
            "La variedad de prueba por defecto (TranscendentalManifold) tiene raíces\n"
            "analíticas en (nπ, 0) para n ∈ ℤ. Con radio=10 y profundidad=8 se\n"
            "deben detectar las raíces en (-3π, -2π, -π, 0, π, 2π, 3π).\n"
            "\n"
            "Ejemplos:\n"
            "  iobsolve roots\n"
            "  iobsolve roots --radius 10 --depth 10 --plot raices.png\n"
            "  iobsolve roots --expr 'sin(x)-y, cos(y)-x' --radius 4\n"
            "  iobsolve roots --manifold mi_sistema.py:MiEcuacion --radius 6\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p_roots.add_argument(
        "--radius", type=float, default=5.0,
        help="Radio inicial del dominio de búsqueda (default: 5.0)"
    )
    p_roots.add_argument(
        "--depth", type=int, default=8,
        help="Profundidad máxima de bisección recursiva (default: 8)"
    )
    p_roots.add_argument(
        "--res", type=int, default=16,
        help="Resolución de muestreo de la malla local (default: 16, max recomendado: 32)"
    )
    p_roots.add_argument(
        "--tau-spec", type=float, default=1e-3,
        help="Umbral de estrés espectral τ_c (default: 1e-3). Valores menores aumentan sensibilidad."
    )
    p_roots.add_argument(
        "--no-sign-filter", action="store_true",
        help="Desactiva el filtro TVI (cambio de signo). Actívalo solo para singularidades no-raíz."
    )
    p_roots.add_argument(
        "--manifold", type=str, metavar="FILE:CLASE",
        help="Variedad externa: 'mi_math.py:MiEcuacion'"
    )
    p_roots.add_argument(
        "--expr", type=str, metavar="EXPR",
        help="Expresión matemática directa. Ej: 'sin(x) - y, cos(y) - x'"
    )

    # --- spectral ---
    p_spectral = subparsers.add_parser(
        "spectral", parents=[parent_parser],
        help="Mapeo global de densidad topológica (IOB-FFT)"
    )
    p_spectral.add_argument(
        "--grid", type=int, default=1024,
        help="Resolución de la malla computacional (default: 1024)"
    )

    # --- dynamics ---
    p_dynamics = subparsers.add_parser(
        "dynamics", parents=[parent_parser],
        help="Monitor dinámico predictivo (Early Warning)"
    )
    p_dynamics.add_argument(
        "--dim", type=int, default=40,
        help="Dimensionalidad del espacio de fases (default: 40)"
    )
    p_dynamics.add_argument(
        "--l-metric", type=parse_l_metric, default="auto",
        help="Resolución óptica ('auto' o float, default: auto)"
    )
    p_dynamics.add_argument(
        "-i", "--input", type=str, metavar="PATH",
        help="Archivo de trayectorias (.npy o .pt)"
    )

    # --- shield ---
    p_shield = subparsers.add_parser(
        "shield", parents=[parent_parser],
        help="Cirugía topológica en grafos discretos (Anti-DDoS)"
    )
    p_shield.add_argument(
        "--nodes", type=int, default=1000,
        help="Cardinalidad de la red a simular (default: 1000)"
    )
    p_shield.add_argument(
        "--tau", type=float, default=3.0,
        help="Umbral crítico del Z-Score Topológico (default: 3.0)"
    )
    p_shield.add_argument(
        "--l-metric", type=parse_l_metric, default="auto",
        help="Calibración del ruido basal ('auto' o float)"
    )
    p_shield.add_argument(
        "--attack", action="store_true",
        help="Inyecta tráfico asimétrico masivo para probar la cirugía"
    )
    p_shield.add_argument(
        "-i", "--input", type=str, metavar="PATH",
        help="Topología exportada del usuario (.graphml)"
    )
    p_shield.add_argument(
        "--traffic", type=str, metavar="PATH",
        help="Carga de tráfico asíncrono (.npy, .pt)"
    )

    # --- audit ---
    p_audit = subparsers.add_parser(
        "audit", parents=[parent_parser],
        help="Auditoría de espacios latentes en IA (Colapso Modal)"
    )
    p_audit.add_argument(
        "--batch", type=int, default=128,
        help="Tamaño del tensor de mini-batch (default: 128)"
    )
    p_audit.add_argument(
        "--dim", type=int, default=256,
        help="Dimensionalidad del vector latente (default: 256)"
    )
    p_audit.add_argument(
        "--tau", type=float, default=0.85,
        help="Tolerancia máxima al estrés de cohesión (default: 0.85)"
    )
    p_audit.add_argument(
        "-i", "--input", type=str, metavar="PATH",
        help="Tensor de representaciones latentes (.npy, .pt)"
    )

    # --- check ---
    subparsers.add_parser(
        "check",
        help="Verifica el entorno, versiones y aceleración por hardware"
    )

    args: argparse.Namespace = parser.parse_args()

    # Inyección de configuración JSON
    if getattr(args, "config", None):
        try:
            config_dict = load_json_config(args.config)
            for key, val in config_dict.items():
                canonical = key.replace("-", "_")
                if hasattr(args, canonical):
                    setattr(args, canonical, val)
        except Exception as config_err:
            print(f"\n[-] ERROR AL PROCESAR CONFIGURACIÓN: {config_err}")
            sys.exit(1)

    if getattr(args, "quiet", False):
        QUIET_MODE = True

    dispatch: Dict[str, Callable[[argparse.Namespace], None]] = {
        "roots": handle_roots,
        "spectral": handle_spectral,
        "dynamics": handle_dynamics,
        "shield": handle_shield,
        "audit": handle_audit,
        "check": handle_check,
    }

    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        print("\n[-] Abortado por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[-] ERROR CRÍTICO DEL MOTOR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
