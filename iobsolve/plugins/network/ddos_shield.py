import time
import asyncio
import logging
from typing import Dict, Tuple, Union
from dataclasses import dataclass

from iobsolve.core.operator import TopologicalCrisisPredictor

logger = logging.getLogger(__name__)

@dataclass
class NetworkState:
    """Snapshot del estado de la red en el instante t."""
    active_requests: int
    avg_latency: float

class IPMetrics:
    """
    Rastreador de métricas por IP.
    
    Se mantiene intencionalmente ligero para minimizar el overhead de memoria 
    en diccionarios de perfiles masivos.
    """
    __slots__ = ['request_count', 'total_latency', 'last_seen'] # Optimización de memoria RAM

    def __init__(self):
        self.request_count: int = 0
        self.total_latency: float = 0.0
        self.last_seen: float = time.time()

    def add_latency(self, latency_ms: float) -> None:
        self.request_count += 1
        self.total_latency += latency_ms
        self.last_seen = time.time()

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.request_count if self.request_count > 0 else 0.0

class TopologicalShield:
    """
    Motor de análisis tensorial para topología de red.
    
    Evalúa la matriz de covarianza implícita entre peticiones y latencia 
    para detectar estrés geométrico (Flash Crowds o ataques DDoS volumétricos/asimétricos).
    """
    def __init__(self, alert_threshold: float = 15.0, history_size: int = 20, l_metric: Union[float, str] = "auto"):
        self.predictor = TopologicalCrisisPredictor(alert_threshold, history_size)
        
        self.auto_tune = (l_metric == "auto")
        self.l_metric: float = 1.0 if self.auto_tune else float(l_metric)
        
        if self.auto_tune:
            from iobsolve.core.calibration import MetricCalibrator
            self.calibrator = MetricCalibrator(target_baseline_qd=1.0, calibration_steps=30)
            
        self.state_history: list[NetworkState] = []
        self.max_history = history_size

    def evaluate_network_integrity(self, current_state: NetworkState) -> Tuple[bool, float]:
        self.state_history.append(current_state)
        
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
            
        if len(self.state_history) < 3:
            return False, 0.0
            
        t = len(self.state_history) - 1
        s_minus, s_center, s_plus = self.state_history[t - 2], self.state_history[t - 1], self.state_history[t]
        
        # Laplaciano discreto 1D para deformación
        res_req = s_plus.active_requests + s_minus.active_requests - 2 * s_center.active_requests
        res_lat = s_plus.avg_latency + s_minus.avg_latency - 2 * s_center.avg_latency
        norm_res_sq = (res_req ** 2) + (res_lat ** 2)
        
        if self.auto_tune:
            if not self.calibrator.is_calibrated:
                self.calibrator.push_residual(norm_res_sq)
                return False, 0.0
            self.l_metric = self.calibrator.optimal_L
            
        q_d = norm_res_sq / (2 * (self.l_metric ** 4))
        return self.predictor.push_stress_state(q_d)

class IOBASGIMiddleware:
    """
    Middleware ASGI Avanzado con Cirugía Topológica.
    
    Protege aplicaciones ASGI (FastAPI, Starlette) inyectando el sensor IOB 
    en el pipeline de peticiones. Aísla infractores basándose en asimetría métrica.

    Parameters
    ----------
    app : ASGIApp
        La aplicación ASGI subyacente.
    alert_threshold : float, optional
        Nivel crítico de Psi_c para disparar bloqueos (default: 12.0).
    quarantine_sec : int, optional
        Tiempo de exilio en segundos para IPs identificadas (default: 120).
    """
    def __init__(self, 
                 app, 
                 alert_threshold: float = 12.0,
                 history_size: int = 20,
                 l_metric: Union[float, str] = "auto", 
                 quarantine_sec: int = 120,
                 eval_interval_sec: float = 1.0):
                 
        self.app = app
        self.quarantine_sec = quarantine_sec
        self.eval_interval_sec = eval_interval_sec
        
        self.shield = TopologicalShield(alert_threshold, history_size, l_metric)
        
        # Estado en memoria
        # FIXME: Vulnerable a agotamiento de RAM bajo ataque DDoS con IP Spoofing masivo.
        # Considerar implementar un LRU Cache o Redis para `ip_profiles` en producción.
        self.quarantined_ips: Dict[str, float] = {}
        self.ip_profiles: Dict[str, IPMetrics] = {}
        
        self.active_requests: int = 0
        self.total_latency: float = 0.0
        self.latency_samples: int = 0
        
        # Sincronización
        self._metrics_lock = asyncio.Lock()
        self._telemetry_task_started = False

    async def _execute_topological_surgery(self) -> None:
        """Aísla IPs asimétricas durante un pico de estrés."""
        async with self._metrics_lock:
            now = time.time()
            # Eviction policy: purgamos IPs inactivas hace más de 60s
            self.ip_profiles = {ip: m for ip, m in self.ip_profiles.items() if now - m.last_seen < 60}
            
            if not self.ip_profiles:
                return
                
            baseline_latency = 100.0  # TODO: Hacer dinámico basado en un SMA global
            
            for ip, metrics in self.ip_profiles.items():
                if metrics.avg_latency > (baseline_latency * 3) and metrics.request_count >= 2:
                    if ip not in self.quarantined_ips:
                        logger.error(f"[CIRUGÍA] IP Aislada: {ip} | Lat: {metrics.avg_latency:.1f}ms")
                        self.quarantined_ips[ip] = now + self.quarantine_sec

    async def _telemetry_loop(self) -> None:
        """Demonio Lagrangiano que evalúa la métrica en background sin bloquear peticiones."""
        while True:
            await asyncio.sleep(self.eval_interval_sec)
            
            async with self._metrics_lock:
                avg_lat = self.total_latency / self.latency_samples if self.latency_samples > 0 else 0.0
                state = NetworkState(self.active_requests, avg_lat)
                
                # Reset suave para evitar inercia desmedida
                if self.latency_samples > 100:
                    self.total_latency, self.latency_samples = avg_lat, 1
                    
            is_crisis, psi_c = self.shield.evaluate_network_integrity(state)
            
            if is_crisis:
                logger.warning(f"[IOB] ALERTA DE SISTEMA (Psi_c={psi_c:.2f}).")
                await self._execute_topological_surgery()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        if not self._telemetry_task_started:
            asyncio.create_task(self._telemetry_loop())
            self._telemetry_task_started = True

        client_host = scope.get("client", ["127.0.0.1"])[0] if scope.get("client") else "127.0.0.1"
        headers = dict(scope.get("headers", []))
        client_ip = headers[b"x-forwarded-for"].decode("utf-8").split(",")[0].strip() if b"x-forwarded-for" in headers else client_host

        # Firewall de complejidad O(1)
        if client_ip in self.quarantined_ips:
            if time.time() < self.quarantined_ips[client_ip]:
                return await self._send_429(send, "Topological Integrity Protection: Geometric stress isolation.")
            del self.quarantined_ips[client_ip]
            
        async with self._metrics_lock:
            self.active_requests += 1
            if client_ip not in self.ip_profiles:
                self.ip_profiles[client_ip] = IPMetrics()
                
        start_time = time.perf_counter()
        
        try:
            await self.app(scope, receive, send)
        finally:
            process_time = (time.perf_counter() - start_time) * 1000
            async with self._metrics_lock:
                self.total_latency += process_time
                self.latency_samples += 1
                self.active_requests = max(0, self.active_requests - 1)
                self.ip_profiles[client_ip].add_latency(process_time)

    async def _send_429(self, send, message: str) -> None:
        """Envía respuesta estandarizada de Rate Limit superado."""
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"retry-after", str(self.quarantine_sec).encode("utf-8"))
            ],
        })
        await send({
            "type": "http.response.body",
            "body": f'{{"error": "{message}"}}'.encode("utf-8"),
        })