#!/usr/bin/env python3
"""
Benchmark Realista: 10,000 vehículos - 3 horas de simulación

Simula el procesamiento paralelo de emisiones e itinerarios para:
- 10,000 vehículos
- 3 horas (10,800 segundos) de simulación
- Comparación: 1 vs 2 vs 4 procesos
"""

import os
import sys
import time
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent


def print_header(text):
    print(f"\n{'='*70}")
    print(f" {text}")
    print(f"{'='*70}")


def print_section(text):
    print(f"\n{'-'*60}")
    print(f" {text}")
    print(f"{'-'*60}")


# =====================================================
# CÁLCULO DE EMISIONES (Modelo HBEFA simplificado)
# =====================================================

def calculate_emissions_batch(vehicle_states):
    """
    Calcula emisiones para un batch de estados de vehículos.
    Modelo basado en HBEFA con factores de velocidad y aceleración.
    """
    results = []
    
    for state in vehicle_states:
        speed = state["speed"]  # m/s
        accel = state["acceleration"]  # m/s²
        distance = state["distance"]  # m
        waiting = state["waiting_time"]  # s
        
        # Clasificación de velocidad
        if speed < 0.1:  # Parado
            speed_class = "idle"
        elif speed < 8.33:  # < 30 km/h
            speed_class = "low"
        elif speed < 22.22:  # < 80 km/h
            speed_class = "medium"
        else:
            speed_class = "high"
        
        # Factores de emisión base (g/km para CO2, L/km para fuel)
        emission_factors = {
            "idle": {"co2": 2.5, "co": 0.008, "nox": 0.0005, "fuel": 0.0008},  # g/s o L/s
            "low": {"co2": 180, "co": 1.2, "nox": 0.15, "fuel": 0.075},
            "medium": {"co2": 150, "co": 0.5, "nox": 0.12, "fuel": 0.062},
            "high": {"co2": 170, "co": 0.8, "nox": 0.25, "fuel": 0.070}
        }
        
        factors = emission_factors[speed_class]
        
        # Factor de aceleración (más aceleración = más emisiones)
        if accel > 0:
            accel_factor = 1.0 + 0.15 * min(accel, 3.0)
        elif accel < -1:
            accel_factor = 0.9  # Frenado regenerativo (menos emisiones)
        else:
            accel_factor = 1.0
        
        # Cálculo de emisiones
        distance_km = distance / 1000.0
        
        if speed_class == "idle":
            # Emisiones por tiempo (idle)
            idle_time = max(1.0, waiting)
            co2 = factors["co2"] * idle_time * accel_factor
            co = factors["co"] * idle_time
            nox = factors["nox"] * idle_time
            fuel = factors["fuel"] * idle_time
        else:
            # Emisiones por distancia
            co2 = factors["co2"] * distance_km * accel_factor
            co = factors["co"] * distance_km * accel_factor
            nox = factors["nox"] * distance_km * accel_factor
            fuel = factors["fuel"] * distance_km * accel_factor
        
        # Simulación de cálculos adicionales (representando modelo HBEFA completo)
        for _ in range(10):
            temp_factor = 1.0 + 0.02 * np.sin(speed * 0.1)
            co2 *= temp_factor
            _ = np.exp(-speed * 0.01) * np.log1p(abs(accel) + 1)
        
        results.append({
            "vehicle_id": state["vehicle_id"],
            "time_step": state["time_step"],
            "co2": co2,
            "co": co,
            "nox": nox,
            "fuel": fuel
        })
    
    return results


def calculate_route_batch(args):
    """
    Calcula rutas para un batch de solicitudes.
    """
    graph, requests = args
    results = []
    
    import heapq
    
    for req in requests:
        origin = req["origin"]
        destination = req["destination"]
        
        # Dijkstra
        pq = [(0, origin, [])]
        visited = set()
        found = False
        
        while pq and not found:
            cost, node, path = heapq.heappop(pq)
            
            if node == destination:
                total_dist = sum(graph[e]["length"] for e in path) if path else 0
                results.append({
                    "request_id": req["request_id"],
                    "route": path,
                    "distance": total_dist,
                    "time": cost,
                    "success": True
                })
                found = True
                break
            
            if node in visited:
                continue
            visited.add(node)
            
            if node in graph.get("_adj", {}):
                for next_node, edge_data in graph["_adj"][node].items():
                    if next_node not in visited:
                        edge_cost = edge_data["cost"]
                        heapq.heappush(pq, (cost + edge_cost, next_node, path + [edge_data["id"]]))
        
        if not found:
            results.append({
                "request_id": req["request_id"],
                "route": [],
                "distance": 0,
                "time": 0,
                "success": False
            })
    
    return results


def generate_simulation_data(num_vehicles, simulation_hours, sample_interval=10):
    """
    Genera datos de simulación para el benchmark.
    
    Args:
        num_vehicles: Número de vehículos
        simulation_hours: Duración en horas
        sample_interval: Intervalo de muestreo en segundos
    """
    simulation_seconds = int(simulation_hours * 3600)
    num_samples = simulation_seconds // sample_interval
    
    print(f"\n  Generando datos de simulación...")
    print(f"    - Vehículos: {num_vehicles:,}")
    print(f"    - Duración: {simulation_hours} horas ({simulation_seconds:,} segundos)")
    print(f"    - Intervalo de muestreo: {sample_interval}s")
    print(f"    - Muestras por vehículo: {num_samples:,}")
    print(f"    - Total de estados: {num_vehicles * num_samples:,}")
    
    np.random.seed(42)
    
    all_states = []
    
    # Generar estados por timestep para simular comportamiento realista
    for t in range(0, simulation_seconds, sample_interval):
        # Número de vehículos activos varía (simula entrada/salida)
        # Pico de tráfico entre hora 1 y hora 2
        hour = t / 3600
        if 1.0 <= hour <= 2.0:
            active_fraction = 1.0  # 100% de vehículos
        elif 0.5 <= hour < 1.0 or 2.0 < hour <= 2.5:
            active_fraction = 0.7  # 70%
        else:
            active_fraction = 0.4  # 40%
        
        num_active = int(num_vehicles * active_fraction)
        
        for v in range(num_active):
            # Velocidad depende de congestión (más vehículos = menor velocidad)
            congestion_factor = num_active / num_vehicles
            base_speed = np.random.uniform(5, 30)
            speed = base_speed * (1.0 - 0.5 * congestion_factor)
            
            state = {
                "vehicle_id": f"veh_{v}",
                "time_step": float(t),
                "speed": max(0, speed + np.random.normal(0, 2)),
                "acceleration": np.random.uniform(-2, 2),
                "distance": max(0, speed * sample_interval + np.random.normal(0, 5)),
                "waiting_time": np.random.exponential(2) if speed < 1 else 0
            }
            all_states.append(state)
    
    print(f"    - Estados generados: {len(all_states):,}")
    
    return all_states


def generate_network_and_routes(grid_size, num_requests):
    """
    Genera red de carreteras y solicitudes de rutas.
    """
    print(f"\n  Generando red y solicitudes de ruta...")
    print(f"    - Tamaño de red: {grid_size}x{grid_size}")
    print(f"    - Solicitudes de ruta: {num_requests:,}")
    
    np.random.seed(42)
    
    # Crear grafo con adjacencias
    graph = {"_adj": {}}
    edges = {}
    
    for i in range(grid_size):
        for j in range(grid_size):
            node = f"n_{i}_{j}"
            if node not in graph["_adj"]:
                graph["_adj"][node] = {}
            
            # Edge horizontal
            if j < grid_size - 1:
                next_node = f"n_{i}_{j+1}"
                edge_id = f"h_{i}_{j}"
                length = np.random.uniform(200, 800)
                speed = np.random.uniform(10, 25)
                cost = length / speed
                
                graph["_adj"][node][next_node] = {
                    "id": edge_id, 
                    "cost": cost, 
                    "length": length
                }
                edges[edge_id] = {"length": length, "cost": cost}
            
            # Edge vertical
            if i < grid_size - 1:
                next_node = f"n_{i+1}_{j}"
                edge_id = f"v_{i}_{j}"
                length = np.random.uniform(200, 800)
                speed = np.random.uniform(10, 25)
                cost = length / speed
                
                graph["_adj"][node][next_node] = {
                    "id": edge_id, 
                    "cost": cost, 
                    "length": length
                }
                edges[edge_id] = {"length": length, "cost": cost}
    
    graph.update(edges)
    
    # Generar solicitudes
    requests = []
    for i in range(num_requests):
        o_i, o_j = np.random.randint(0, grid_size), np.random.randint(0, grid_size)
        d_i, d_j = np.random.randint(0, grid_size), np.random.randint(0, grid_size)
        
        requests.append({
            "request_id": f"req_{i}",
            "origin": f"n_{o_i}_{o_j}",
            "destination": f"n_{d_i}_{d_j}"
        })
    
    print(f"    - Edges creados: {len(edges):,}")
    
    return graph, requests


def benchmark_emissions(all_states, process_counts, batch_size=5000):
    """
    Benchmark del cálculo de emisiones.
    """
    total_states = len(all_states)
    
    print_section(f"BENCHMARK EMISIONES ({total_states:,} estados)")
    
    # Crear batches
    batches = []
    for i in range(0, len(all_states), batch_size):
        batches.append(all_states[i:i + batch_size])
    
    print(f"\n  Batches: {len(batches)} de ~{batch_size:,} estados cada uno")
    
    results = {}
    baseline_time = None
    
    for num_proc in process_counts:
        print(f"\n  → {num_proc} proceso(s)...", end=" ", flush=True)
        
        if num_proc == 1:
            # Secuencial
            start = time.time()
            all_results = []
            for batch in batches:
                all_results.extend(calculate_emissions_batch(batch))
            elapsed = time.time() - start
        else:
            # Paralelo
            start = time.time()
            with Pool(processes=num_proc) as pool:
                batch_results = pool.map(calculate_emissions_batch, batches)
            all_results = [r for batch in batch_results for r in batch]
            elapsed = time.time() - start
        
        if num_proc == 1:
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed if elapsed > 0 else 1.0
        
        efficiency = speedup / num_proc
        throughput = total_states / elapsed
        
        results[num_proc] = {
            "time": elapsed,
            "speedup": speedup,
            "efficiency": efficiency,
            "throughput": throughput
        }
        
        print(f"Tiempo: {elapsed:.2f}s | Speedup: {speedup:.2f}x | Eficiencia: {efficiency:.0%}")
    
    # Calcular totales
    total_co2 = sum(r["co2"] for r in all_results) / 1000  # kg
    total_fuel = sum(r["fuel"] for r in all_results)  # L
    
    print(f"\n  Emisiones totales:")
    print(f"    - CO2: {total_co2:,.1f} kg ({total_co2/1000:.2f} toneladas)")
    print(f"    - Combustible: {total_fuel:,.1f} litros")
    
    return results


def benchmark_routing(graph, requests, process_counts, batch_size=100):
    """
    Benchmark del cálculo de rutas.
    """
    num_requests = len(requests)
    
    print_section(f"BENCHMARK ITINERARIOS ({num_requests:,} solicitudes)")
    
    # Crear batches de solicitudes
    request_batches = []
    for i in range(0, len(requests), batch_size):
        request_batches.append(requests[i:i + batch_size])
    
    print(f"\n  Batches: {len(request_batches)} de ~{batch_size} solicitudes")
    
    results = {}
    baseline_time = None
    
    for num_proc in process_counts:
        print(f"\n  → {num_proc} proceso(s)...", end=" ", flush=True)
        
        if num_proc == 1:
            start = time.time()
            all_results = []
            for batch in request_batches:
                all_results.extend(calculate_route_batch((graph, batch)))
            elapsed = time.time() - start
        else:
            start = time.time()
            args_list = [(graph, batch) for batch in request_batches]
            with Pool(processes=num_proc) as pool:
                batch_results = pool.map(calculate_route_batch, args_list)
            all_results = [r for batch in batch_results for r in batch]
            elapsed = time.time() - start
        
        if num_proc == 1:
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed if elapsed > 0 else 1.0
        
        efficiency = speedup / num_proc
        successful = sum(1 for r in all_results if r["success"])
        
        results[num_proc] = {
            "time": elapsed,
            "speedup": speedup,
            "efficiency": efficiency,
            "successful": successful
        }
        
        print(f"Tiempo: {elapsed:.2f}s | Speedup: {speedup:.2f}x | Rutas OK: {successful}/{num_requests}")
    
    return results


def plot_results(emission_results, routing_results, process_counts):
    """Generar gráficas de resultados"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        procs = list(process_counts)
        
        # Emisiones
        ax1 = axes[0]
        speedups_e = [emission_results[p]["speedup"] for p in procs]
        times_e = [emission_results[p]["time"] for p in procs]
        
        x = np.arange(len(procs))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, speedups_e, width, label='Speedup Real', color='#3498db')
        ax1.plot(x, procs, 'o--', color='#e74c3c', label='Speedup Ideal', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Número de Procesos', fontsize=12)
        ax1.set_ylabel('Speedup', fontsize=12)
        ax1.set_title('Speedup - Cálculo de Emisiones\n(10,000 vehículos, 3 horas)', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(procs)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Añadir valores y tiempos
        for i, (bar, s, t) in enumerate(zip(bars1, speedups_e, times_e)):
            ax1.annotate(f'{s:.2f}x\n({t:.1f}s)', 
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10)
        
        # Routing
        ax2 = axes[1]
        speedups_r = [routing_results[p]["speedup"] for p in procs]
        times_r = [routing_results[p]["time"] for p in procs]
        
        bars2 = ax2.bar(x - width/2, speedups_r, width, label='Speedup Real', color='#2ecc71')
        ax2.plot(x, procs, 'o--', color='#e74c3c', label='Speedup Ideal', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Número de Procesos', fontsize=12)
        ax2.set_ylabel('Speedup', fontsize=12)
        ax2.set_title('Speedup - Cálculo de Itinerarios\n(1,000 solicitudes)', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(procs)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        for i, (bar, s, t) in enumerate(zip(bars2, speedups_r, times_r)):
            ax2.annotate(f'{s:.2f}x\n({t:.2f}s)', 
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        output_path = PROJECT_ROOT / "results" / "speedup_10k_vehicles_3h.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  📊 Gráfica guardada: {output_path}")
        
    except ImportError:
        print("\n  (matplotlib no disponible)")


def main():
    print("\n" + "█"*70)
    print(" BENCHMARK: 10,000 VEHÍCULOS - 3 HORAS DE SIMULACIÓN")
    print(" Comparación de rendimiento: 1 vs 2 vs 4 procesos")
    print("█"*70)
    
    # Configuración
    PROCESS_COUNTS = [1, 2, 4]
    NUM_VEHICLES = 10000
    SIMULATION_HOURS = 3
    SAMPLE_INTERVAL = 10  # Muestrear cada 10 segundos
    
    # Para routing
    GRID_SIZE = 50  # Red 50x50
    NUM_ROUTE_REQUESTS = 1000
    
    print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  CONFIGURACIÓN DEL BENCHMARK                                 ║")
    print(f"  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  CPUs disponibles:     {cpu_count():<37}║")
    print(f"  ║  Procesos a comparar:  {str(PROCESS_COUNTS):<37}║")
    print(f"  ║  Vehículos:            {NUM_VEHICLES:,}{'':<30}║")
    print(f"  ║  Duración simulación:  {SIMULATION_HOURS} horas{'':<31}║")
    print(f"  ║  Intervalo muestreo:   {SAMPLE_INTERVAL} segundos{'':<27}║")
    print(f"  ║  Red para rutas:       {GRID_SIZE}x{GRID_SIZE}{'':<33}║")
    print(f"  ║  Solicitudes de ruta:  {NUM_ROUTE_REQUESTS:,}{'':<31}║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")
    
    # Generar datos
    all_states = generate_simulation_data(NUM_VEHICLES, SIMULATION_HOURS, SAMPLE_INTERVAL)
    graph, requests = generate_network_and_routes(GRID_SIZE, NUM_ROUTE_REQUESTS)
    
    # Ejecutar benchmarks
    emission_results = benchmark_emissions(all_states, PROCESS_COUNTS)
    routing_results = benchmark_routing(graph, requests, PROCESS_COUNTS)
    
    # Resumen final
    print_header("RESUMEN DE RESULTADOS")
    
    print("""
  ┌────────────────────────────────────────────────────────────────────┐
  │                    SPEEDUP OBTENIDO                                │
  ├──────────────┬───────────────────────┬─────────────────────────────┤
  │   Procesos   │      EMISIONES        │        ITINERARIOS          │
  │              │  Speedup | Eficiencia │    Speedup  |  Eficiencia   │
  ├──────────────┼───────────────────────┼─────────────────────────────┤""")
    
    for p in PROCESS_COUNTS:
        e = emission_results[p]
        r = routing_results[p]
        print(f"  │      {p}       │   {e['speedup']:.2f}x  |    {e['efficiency']:.0%}     │     {r['speedup']:.2f}x   |     {r['efficiency']:.0%}       │")
    
    print("""  └──────────────┴───────────────────────┴─────────────────────────────┘""")
    
    # Generar gráfica
    plot_results(emission_results, routing_results, PROCESS_COUNTS)
    
    # Análisis
    e4 = emission_results[4]
    r4 = routing_results[4]
    
    print(f"""
  ╔════════════════════════════════════════════════════════════════════╗
  ║                         CONCLUSIONES                               ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║                                                                    ║
  ║  EMISIONES (con 4 procesos):                                       ║
  ║    • Speedup: {e4['speedup']:.2f}x                                               ║
  ║    • Eficiencia: {e4['efficiency']:.0%}                                            ║
  ║    • Throughput: {e4['throughput']:,.0f} estados/segundo                      ║
  ║                                                                    ║
  ║  ITINERARIOS (con 4 procesos):                                     ║
  ║    • Speedup: {r4['speedup']:.2f}x                                               ║
  ║    • Eficiencia: {r4['efficiency']:.0%}                                            ║
  ║    • Rutas exitosas: {r4['successful']}/{NUM_ROUTE_REQUESTS}                                   ║
  ║                                                                    ║
  ╚════════════════════════════════════════════════════════════════════╝
""")
    
    print("="*70)
    print(" ✓ BENCHMARK COMPLETADO")
    print("="*70 + "\n")
    
    return emission_results, routing_results


if __name__ == "__main__":
    main()


