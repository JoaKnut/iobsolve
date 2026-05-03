import time
import asyncio
import logging
import uvicorn
import httpx
from fastapi import FastAPI, Request
from iobsolve.plugins.network.ddos_shield import IOBASGIMiddleware

# Configuración de logs limpia
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("iobsolve.tests.ddos")

# =====================================================================
# 1. EL SERVIDOR VÍCTIMA (FastAPI protegido por IOB)
# =====================================================================

app = FastAPI(title="Servidor Protegido por IOB")

app.add_middleware(
    IOBASGIMiddleware,
    alert_threshold=5.0,
    l_metric=1.0,
    quarantine_sec=30,
    eval_interval_sec=0.5
)

active_connections = 0

@app.get("/api/data")
async def get_data(request: Request):
    global active_connections
    active_connections += 1
    
    # SOLUCIÓN ERROR LINEA 48: Verificación defensiva del host
    client_host = request.client.host if request.client else "127.0.0.1"
    client_ip = request.headers.get("x-forwarded-for", client_host)
    
    is_attacker = client_ip == "192.168.1.99"
    
    try:
        if is_attacker:
            delay = 0.5 + (active_connections * 0.1) 
        else:
            delay = 0.05 + (active_connections * 0.01)
            
        await asyncio.sleep(delay)
        return {"status": "OK", "ip": client_ip, "delay_ms": round(delay * 1000, 2)}
    finally:
        active_connections -= 1

# =====================================================================
# 2. GENERADORES DE TRÁFICO
# =====================================================================

async def simulate_legitimate_traffic(base_url: str):
    async with httpx.AsyncClient(base_url=base_url) as client:
        ips = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
        while True:
            for ip in ips:
                try:
                    await client.get("/api/data", headers={"x-forwarded-for": ip})
                except Exception:
                    pass
            await asyncio.sleep(0.2)

async def launch_stealth_attack(base_url: str, attacker_ip: str):
    logger.warning(f"🚀 INICIANDO ATAQUE SIGILOSO DESDE IP: {attacker_ip}")
    
    async with httpx.AsyncClient(base_url=base_url) as client:
        attack_count = 0
        while True:
            attack_count += 1
            start = time.perf_counter()
            try:
                response = await client.get("/api/data", headers={"x-forwarded-for": attacker_ip})
                latency = (time.perf_counter() - start) * 1000
                
                # SOLUCIÓN ERROR LINEA 104: Uso de logger.info con prefijo
                if response.status_code == 429:
                    logger.info(f"[DEFENSA] ✅ Bloqueado: Ataque {attack_count} contenido por IOB.")
                else:
                    logger.info(f"[IMPACTO] 🗡️ Latencia: {latency:.1f}ms | Status: {response.status_code}")
            
            except Exception as e:
                 logger.error(f"Error de red: {e}")
            
            await asyncio.sleep(0.3) 

# =====================================================================
# 3. ORQUESTADOR DE LA PRUEBA
# =====================================================================

async def run_test_scenario():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error")
    server = uvicorn.Server(config)
    
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(1)
    
    base_url = "http://127.0.0.1:8000"
    logger.info("=== SERVIDOR IOB ONLINE ===")
    
    legit_task = asyncio.create_task(simulate_legitimate_traffic(base_url))
    await asyncio.sleep(5)
    
    attack_task = asyncio.create_task(launch_stealth_attack(base_url, "192.168.1.99"))
    await asyncio.sleep(15)
    
    logger.info("=== SIMULACIÓN FINALIZADA ===")
    server.should_exit = True
    legit_task.cancel()
    attack_task.cancel()
    await server_task

if __name__ == "__main__":
    try:
        asyncio.run(run_test_scenario())
    except KeyboardInterrupt:
        pass