#!/usr/bin/env python3
"""
Benchmark de Comparación: 1 vs 2 vs 4 procesos

Ejecuta el cálculo paralelo de emisiones e itinerarios con diferentes
números de procesos para analizar el speedup.
"""

import os
import sys
import time
import warnings
from pathlib import Path

# Silenciar warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Configurar paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Importar módulos sin mensajes
import io
from contextlib import redirect_stdout, redirect_stderr

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    from modules.emissions import ParallelEmissionProcessor
    from modules.routing import NetworkGraph, Edge, ParallelRouteProcessor


def print_header(text):
    print(f"\n{'='*70}")
    print(f" {text}")
    print(f"{'='*70}")


def print_section(text):
    print(f"\n{'-'*50}")
    print(f" {text}")
    print(f"{'-'*50}")


def benchmark_emissions(num_vehicles: int, num_timesteps: int, process_counts: list):
    """
    Benchmark del cálculo de emisiones con diferentes números de procesos.
    """
    print_section(f"BENCHMARK EMISIONES ({num_vehicles} vehículos × {num_timesteps} timesteps)")
    
    total_states = num_vehicles * num_timesteps
    print(f"\n  Estados totales a procesar: {total_states:,}")
    
    # Generar datos
    np.random.seed(42)
    
    vehicle_states = []
    for t in range(num_timesteps):
        for v in range(num_vehicles):
            state = {
                "vehicle_id": f"veh_{v}",
                "time_step": float(t),
                "speed": np.random.uniform(5, 30),
                "acceleration": np.random.uniform(-3, 3),
                "position": [np.random.uniform(0, 10000), np.random.uniform(0, 10000)],
                "edge_id": f"edge_{np.random.randint(0, 100)}",
                "distance": np.random.uniform(5, 30),
                "waiting_time": np.random.uniform(0, 10)
            }
            vehicle_states.append(state)
    
    results = {}
    baseline_time = None
    
    for num_proc in process_counts:
        print(f"\n  → {num_proc} proceso(s)...", end=" ", flush=True)
        
        # Crear procesador
        processor = ParallelEmissionProcessor(
            num_processes=num_proc,
            batch_size=500  # Batch más grande para mejor paralelismo
        )
        
        # Ejecutar múltiples veces para promediar
        times = []
        for _ in range(3):
            processor.reset_stats()
            start = time.time()
            emission_results = processor.process_emissions(vehicle_states)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        if num_proc == 1:
            baseline_time = avg_time
            speedup = 1.0
        else:
            speedup = baseline_time / avg_time if avg_time > 0 else 1.0
        
        efficiency = speedup / num_proc
        
        results[num_proc] = {
            "time": avg_time,
            "std": std_time,
            "speedup": speedup,
            "efficiency": efficiency,
            "throughput": total_states / avg_time
        }
        
        print(f"Tiempo: {avg_time:.3f}s (±{std_time:.3f}), Speedup: {speedup:.2f}x")
    
    # Calcular emisiones totales (del último resultado)
    total_co2 = sum(r["co2"] for r in emission_results)
    total_fuel = sum(r["fuel"] for r in emission_results)
    
    print(f"\n  Emisiones totales calculadas:")
    print(f"    - CO2: {total_co2/1000:.2f} kg")
    print(f"    - Combustible: {total_fuel:.2f} L")
    
    return results


def benchmark_routing(num_edges: int, num_requests: int, process_counts: list):
    """
    Benchmark del cálculo de rutas con diferentes números de procesos.
    """
    print_section(f"BENCHMARK ITINERARIOS ({num_edges} edges, {num_requests} solicitudes)")
    
    # Crear red más grande
    network = NetworkGraph()
    grid_size = int(np.sqrt(num_edges / 2)) + 1
    
    print(f"\n  Creando red grid {grid_size}x{grid_size}...")
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Edge horizontal
            if j < grid_size - 1:
                edge = Edge(
                    edge_id=f"e_h_{i}_{j}",
                    from_node=f"n_{i}_{j}",
                    to_node=f"n_{i}_{j+1}",
                    length=np.random.uniform(200, 800),
                    speed_limit=np.random.uniform(10, 25),
                    num_lanes=2
                )
                network.add_edge(edge)
            
            # Edge vertical
            if i < grid_size - 1:
                edge = Edge(
                    edge_id=f"e_v_{i}_{j}",
                    from_node=f"n_{i}_{j}",
                    to_node=f"n_{i+1}_{j}",
                    length=np.random.uniform(200, 800),
                    speed_limit=np.random.uniform(10, 25),
                    num_lanes=2
                )
                network.add_edge(edge)
    
    print(f"  Edges creados: {len(network.edges)}")
    
    # Generar solicitudes de rutas
    np.random.seed(42)
    route_requests = []
    
    for i in range(num_requests):
        o_i = np.random.randint(0, grid_size - 1)
        o_j = np.random.randint(0, grid_size - 1)
        d_i = np.random.randint(0, grid_size - 1)
        d_j = np.random.randint(0, grid_size - 1)
        
        route_requests.append({
            "request_id": f"req_{i}",
            "vehicle_id": f"veh_{i % 100}",
            "origin_edge": f"e_h_{o_i}_{o_j}",
            "destination_edge": f"e_h_{d_i}_{d_j}",
            "departure_time": float(i),
            "criteria": "time"
        })
    
    results = {}
    baseline_time = None
    
    for num_proc in process_counts:
        print(f"\n  → {num_proc} proceso(s)...", end=" ", flush=True)
        
        # Crear procesador
        processor = ParallelRouteProcessor(
            network=network,
            num_processes=num_proc,
            batch_size=50
        )
        
        # Ejecutar múltiples veces
        times = []
        for _ in range(3):
            processor.reset_stats()
            start = time.time()
            route_results = processor.process_routes(route_requests)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        if num_proc == 1:
            baseline_time = avg_time
            speedup = 1.0
        else:
            speedup = baseline_time / avg_time if avg_time > 0 else 1.0
        
        efficiency = speedup / num_proc
        successful = sum(1 for r in route_results if r["success"])
        
        results[num_proc] = {
            "time": avg_time,
            "std": std_time,
            "speedup": speedup,
            "efficiency": efficiency,
            "successful": successful,
            "throughput": num_requests / avg_time
        }
        
        print(f"Tiempo: {avg_time:.3f}s (±{std_time:.3f}), Speedup: {speedup:.2f}x, Rutas OK: {successful}/{num_requests}")
    
    return results


def print_comparison_table(emission_results, routing_results, process_counts):
    """Imprimir tabla comparativa de resultados"""
    print_header("TABLA COMPARATIVA DE RESULTADOS")
    
    # Tabla de emisiones
    print("\n  EMISIONES:")
    print(f"  {'Procesos':<12} {'Tiempo (s)':<15} {'Speedup':<12} {'Eficiencia':<12} {'Throughput':<15}")
    print(f"  {'-'*66}")
    
    for p in process_counts:
        r = emission_results[p]
        print(f"  {p:<12} {r['time']:.3f} ± {r['std']:.3f}{'':>3} {r['speedup']:.2f}x{'':>6} {r['efficiency']:.1%}{'':>6} {r['throughput']:.0f} est/s")
    
    # Tabla de routing
    print("\n  ITINERARIOS:")
    print(f"  {'Procesos':<12} {'Tiempo (s)':<15} {'Speedup':<12} {'Eficiencia':<12} {'Throughput':<15}")
    print(f"  {'-'*66}")
    
    for p in process_counts:
        r = routing_results[p]
        print(f"  {p:<12} {r['time']:.3f} ± {r['std']:.3f}{'':>3} {r['speedup']:.2f}x{'':>6} {r['efficiency']:.1%}{'':>6} {r['throughput']:.0f} rutas/s")


def plot_results(emission_results, routing_results, process_counts):
    """Generar gráfica de speedup"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Speedup de emisiones
        ax1 = axes[0]
        procs = list(emission_results.keys())
        speedups = [emission_results[p]["speedup"] for p in procs]
        
        ax1.plot(procs, speedups, 'o-', linewidth=2, markersize=8, color='#3498db', label='Real')
        ax1.plot(procs, procs, '--', color='gray', alpha=0.5, label='Ideal')
        ax1.set_xlabel('Número de Procesos')
        ax1.set_ylabel('Speedup')
        ax1.set_title('Speedup - Cálculo de Emisiones')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(procs)
        
        # Speedup de routing
        ax2 = axes[1]
        speedups = [routing_results[p]["speedup"] for p in procs]
        
        ax2.plot(procs, speedups, 'o-', linewidth=2, markersize=8, color='#2ecc71', label='Real')
        ax2.plot(procs, procs, '--', color='gray', alpha=0.5, label='Ideal')
        ax2.set_xlabel('Número de Procesos')
        ax2.set_ylabel('Speedup')
        ax2.set_title('Speedup - Cálculo de Itinerarios')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(procs)
        
        plt.tight_layout()
        
        # Guardar
        output_path = PROJECT_ROOT / "results" / "speedup_comparison.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  Gráfica guardada en: {output_path}")
        
    except ImportError:
        print("\n  (matplotlib no disponible, omitiendo gráfica)")


def main():
    print("\n" + "█"*70)
    print(" BENCHMARK: COMPARACIÓN 1 vs 2 vs 4 PROCESOS")
    print(" Cálculo Paralelo de Emisiones e Itinerarios")
    print("█"*70)
    
    # Configuración
    PROCESS_COUNTS = [1, 2, 4]
    
    # Para emisiones: más datos = mejor speedup
    NUM_VEHICLES = 100        # 100 vehículos
    NUM_TIMESTEPS = 500       # 500 pasos de tiempo
    # Total: 50,000 estados
    
    # Para routing: red más grande y más solicitudes
    NUM_EDGES = 500           # ~500 edges
    NUM_ROUTE_REQUESTS = 200  # 200 solicitudes de ruta
    
    print(f"\n  Configuración del benchmark:")
    print(f"    - Procesos a comparar: {PROCESS_COUNTS}")
    print(f"    - Estados de vehículos: {NUM_VEHICLES * NUM_TIMESTEPS:,}")
    print(f"    - Solicitudes de ruta: {NUM_ROUTE_REQUESTS}")
    
    # Benchmark de emisiones
    emission_results = benchmark_emissions(
        NUM_VEHICLES, 
        NUM_TIMESTEPS, 
        PROCESS_COUNTS
    )
    
    # Benchmark de routing
    routing_results = benchmark_routing(
        NUM_EDGES,
        NUM_ROUTE_REQUESTS,
        PROCESS_COUNTS
    )
    
    # Tabla comparativa
    print_comparison_table(emission_results, routing_results, PROCESS_COUNTS)
    
    # Generar gráfica
    plot_results(emission_results, routing_results, PROCESS_COUNTS)
    
    # Resumen final
    print_header("CONCLUSIONES")
    
    e_speedup_4 = emission_results[4]["speedup"]
    r_speedup_4 = routing_results[4]["speedup"]
    
    print(f"""
  Con 4 procesos vs 1 proceso:
  
    EMISIONES:
      - Speedup: {e_speedup_4:.2f}x
      - Eficiencia: {emission_results[4]['efficiency']:.1%}
      - Throughput: {emission_results[4]['throughput']:.0f} estados/segundo
    
    ITINERARIOS:
      - Speedup: {r_speedup_4:.2f}x  
      - Eficiencia: {routing_results[4]['efficiency']:.1%}
      - Throughput: {routing_results[4]['throughput']:.0f} rutas/segundo
  
  Nota: El speedup depende del volumen de datos.
  Con más vehículos/estados, el speedup mejora significativamente.
""")
    
    print("="*70)
    print(" ✓ BENCHMARK COMPLETADO")
    print("="*70 + "\n")
    
    return emission_results, routing_results


if __name__ == "__main__":
    main()


