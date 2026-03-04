#!/usr/bin/env python3
"""
Test de Simulación Paralela con SUMO

Este script ejecuta una simulación de prueba con 10 vehículos
utilizando 4 procesos paralelos para el cálculo de emisiones e itinerarios.

Uso:
    python test_parallel_simulation.py

Requisitos:
    - SUMO instalado con SUMO_HOME configurado
    - pip install -r requirements.txt
"""

import os
import sys
import time
from pathlib import Path

# Configurar paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Verificar SUMO_HOME
SUMO_HOME = os.environ.get('SUMO_HOME')
if not SUMO_HOME:
    print("="*60)
    print(" ERROR: Variable SUMO_HOME no configurada")
    print("="*60)
    print("\nPor favor, configura SUMO_HOME antes de ejecutar:")
    print("  export SUMO_HOME=/path/to/sumo")
    print("\nO en Mac con Homebrew:")
    print("  export SUMO_HOME=/opt/homebrew/share/sumo")
    print("\nEjecutando simulación sin SUMO (modo demo)...")
    print("="*60 + "\n")
    SUMO_AVAILABLE = False
else:
    tools = os.path.join(SUMO_HOME, 'tools')
    sys.path.append(tools)
    SUMO_AVAILABLE = True

from modules.emissions import ParallelEmissionProcessor, EmissionCalculator, VehicleState
from modules.routing import NetworkGraph, Edge, ParallelRouteProcessor, RouteRequest

# Intentar importar routing mejorado
try:
    from modules.sumo_routing import SUMONetworkParser, SUMORouter, ParallelSUMORouter
    SUMO_ROUTING_AVAILABLE = True
except ImportError:
    SUMO_ROUTING_AVAILABLE = False


def run_demo_parallel_processing():
    """
    Ejecuta demostración del procesamiento paralelo SIN necesidad de SUMO.
    Simula estados de vehículos y calcula emisiones/rutas en paralelo.
    """
    print("\n" + "="*70)
    print(" DEMO: Procesamiento Paralelo de Emisiones e Itinerarios")
    print("="*70)
    
    NUM_PROCESSES = 4
    NUM_VEHICLES = 10
    NUM_TIMESTEPS = 100
    
    print(f"\nConfiguración:")
    print(f"  - Procesos paralelos: {NUM_PROCESSES}")
    print(f"  - Vehículos: {NUM_VEHICLES}")
    print(f"  - Timesteps simulados: {NUM_TIMESTEPS}")
    
    # =====================================================
    # PARTE 1: Cálculo Paralelo de Emisiones
    # =====================================================
    print("\n" + "-"*50)
    print(" PARTE 1: Cálculo Paralelo de Emisiones")
    print("-"*50)
    
    # Crear procesador de emisiones
    emission_processor = ParallelEmissionProcessor(
        num_processes=NUM_PROCESSES,
        batch_size=50
    )
    
    # Generar estados de vehículos simulados
    import numpy as np
    np.random.seed(42)
    
    vehicle_states = []
    for t in range(NUM_TIMESTEPS):
        for v in range(NUM_VEHICLES):
            state = {
                "vehicle_id": f"veh_{v}",
                "time_step": float(t),
                "speed": np.random.uniform(5, 25),  # m/s
                "acceleration": np.random.uniform(-2, 2),  # m/s²
                "position": [np.random.uniform(0, 5000), np.random.uniform(0, 5000)],
                "edge_id": f"edge_{np.random.randint(0, 50)}",
                "distance": np.random.uniform(5, 25),  # metros por step
                "waiting_time": np.random.uniform(0, 5)
            }
            vehicle_states.append(state)
    
    print(f"\nEstados generados: {len(vehicle_states)}")
    
    # Procesar con 1 proceso (baseline)
    print("\n→ Procesando con 1 proceso (baseline)...")
    emission_processor.num_processes = 1
    emission_processor.reset_stats()
    
    start_1p = time.time()
    results_1p = emission_processor.process_emissions(vehicle_states)
    time_1p = time.time() - start_1p
    
    print(f"  Tiempo: {time_1p:.3f}s")
    print(f"  Resultados: {len(results_1p)} cálculos de emisiones")
    
    # Procesar con N procesos
    print(f"\n→ Procesando con {NUM_PROCESSES} procesos...")
    emission_processor.num_processes = NUM_PROCESSES
    emission_processor.reset_stats()
    
    start_np = time.time()
    results_np = emission_processor.process_emissions(vehicle_states)
    time_np = time.time() - start_np
    
    speedup = time_1p / time_np if time_np > 0 else 1.0
    efficiency = speedup / NUM_PROCESSES
    
    print(f"  Tiempo: {time_np:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Eficiencia: {efficiency:.1%}")
    
    # Mostrar totales de emisiones
    total_co2 = sum(r["co2"] for r in results_np)
    total_fuel = sum(r["fuel"] for r in results_np)
    
    print(f"\n  Emisiones totales:")
    print(f"    - CO2: {total_co2/1000:.2f} kg")
    print(f"    - Combustible: {total_fuel:.2f} L")
    
    # =====================================================
    # PARTE 2: Cálculo Paralelo de Rutas
    # =====================================================
    print("\n" + "-"*50)
    print(" PARTE 2: Cálculo Paralelo de Itinerarios")
    print("-"*50)
    
    # Crear red de prueba (grid 10x10)
    network = NetworkGraph()
    
    # Crear edges para una red tipo grid
    print("\n→ Creando red de prueba (grid 10x10)...")
    for i in range(10):
        for j in range(10):
            node_id = f"n_{i}_{j}"
            
            # Edge horizontal (→)
            if j < 9:
                edge = Edge(
                    edge_id=f"e_h_{i}_{j}",
                    from_node=f"n_{i}_{j}",
                    to_node=f"n_{i}_{j+1}",
                    length=500,  # 500m
                    speed_limit=13.89,  # 50 km/h
                    num_lanes=2
                )
                network.add_edge(edge)
            
            # Edge vertical (↓)
            if i < 9:
                edge = Edge(
                    edge_id=f"e_v_{i}_{j}",
                    from_node=f"n_{i}_{j}",
                    to_node=f"n_{i+1}_{j}",
                    length=500,
                    speed_limit=13.89,
                    num_lanes=2
                )
                network.add_edge(edge)
    
    print(f"  Edges creados: {len(network.edges)}")
    
    # Crear procesador de rutas paralelo
    route_processor = ParallelRouteProcessor(
        network=network,
        num_processes=NUM_PROCESSES,
        batch_size=20
    )
    
    # Generar solicitudes de rutas
    route_requests = []
    for i in range(50):  # 50 solicitudes de ruta
        # Origen y destino aleatorios
        o_i, o_j = np.random.randint(0, 9), np.random.randint(0, 8)
        d_i, d_j = np.random.randint(0, 9), np.random.randint(0, 8)
        
        route_requests.append({
            "request_id": f"req_{i}",
            "vehicle_id": f"veh_{i % NUM_VEHICLES}",
            "origin_edge": f"e_h_{o_i}_{o_j}",
            "destination_edge": f"e_h_{d_i}_{d_j}",
            "departure_time": float(i * 10),
            "criteria": "time"
        })
    
    print(f"\n  Solicitudes de ruta generadas: {len(route_requests)}")
    
    # Procesar con 1 proceso
    print("\n→ Calculando rutas con 1 proceso...")
    route_processor.num_processes = 1
    route_processor.reset_stats()
    
    start_1p = time.time()
    routes_1p = route_processor.process_routes(route_requests)
    time_1p = time.time() - start_1p
    
    successful_1p = sum(1 for r in routes_1p if r["success"])
    print(f"  Tiempo: {time_1p:.3f}s")
    print(f"  Rutas exitosas: {successful_1p}/{len(route_requests)}")
    
    # Procesar con N procesos
    print(f"\n→ Calculando rutas con {NUM_PROCESSES} procesos...")
    route_processor.num_processes = NUM_PROCESSES
    route_processor.reset_stats()
    
    start_np = time.time()
    routes_np = route_processor.process_routes(route_requests)
    time_np = time.time() - start_np
    
    speedup_route = time_1p / time_np if time_np > 0 else 1.0
    
    successful_np = sum(1 for r in routes_np if r["success"])
    avg_distance = np.mean([r["total_distance"] for r in routes_np if r["success"]])
    avg_time = np.mean([r["estimated_travel_time"] for r in routes_np if r["success"]])
    
    print(f"  Tiempo: {time_np:.3f}s")
    print(f"  Speedup: {speedup_route:.2f}x")
    print(f"  Rutas exitosas: {successful_np}/{len(route_requests)}")
    print(f"  Distancia promedio: {avg_distance:.0f}m")
    print(f"  Tiempo estimado promedio: {avg_time:.0f}s")
    
    # =====================================================
    # RESUMEN
    # =====================================================
    print("\n" + "="*70)
    print(" RESUMEN DE RESULTADOS")
    print("="*70)
    print(f"\n{'Componente':<25} {'1 Proceso':<15} {'{} Procesos'.format(NUM_PROCESSES):<15} {'Speedup':<10}")
    print("-"*65)
    print(f"{'Emisiones':<25} {time_1p:.3f}s{'':<10} {time_np:.3f}s{'':<10} {speedup:.2f}x")
    print(f"{'Itinerarios':<25} {time_1p:.3f}s{'':<10} {time_np:.3f}s{'':<10} {speedup_route:.2f}x")
    print("-"*65)
    
    return {
        "emission_speedup": speedup,
        "routing_speedup": speedup_route,
        "total_co2_kg": total_co2 / 1000,
        "total_fuel_l": total_fuel
    }


def run_sumo_simulation():
    """
    Ejecuta simulación completa con SUMO (requiere SUMO instalado).
    """
    if not SUMO_AVAILABLE:
        print("\nSUMO no disponible. Ejecutando demo sin SUMO...")
        return run_demo_parallel_processing()
    
    print("\n" + "="*70)
    print(" SIMULACIÓN PARALELA CON SUMO + TraCI")
    print("="*70)
    
    try:
        import traci
        import sumolib
    except ImportError:
        print("\nError: No se puede importar traci/sumolib")
        print("Ejecutando demo sin SUMO...")
        return run_demo_parallel_processing()
    
    from modules.simulation import ParallelSUMOSimulator
    
    NUM_PROCESSES = 4
    
    # Paths
    network_dir = PROJECT_ROOT / "networks"
    net_file = network_dir / "modified.net.xml"
    config_file = network_dir / "test_simulation.sumocfg"
    
    if not net_file.exists():
        print(f"\nError: No se encuentra {net_file}")
        return run_demo_parallel_processing()
    
    print(f"\nConfiguración:")
    print(f"  - Red: {net_file.name}")
    print(f"  - Vehículos: 10")
    print(f"  - Procesos: {NUM_PROCESSES}")
    print(f"  - Tiempo simulación: 300s")
    
    # Crear simulador
    print("\n→ Inicializando simulador paralelo...")
    simulator = ParallelSUMOSimulator(
        num_processes=NUM_PROCESSES,
        emission_batch_size=50,
        routing_batch_size=20,
        use_async_processing=True
    )
    
    # Cargar red
    print("→ Cargando red SUMO...")
    simulator.load_network(str(net_file))
    
    # Comando SUMO
    sumo_binary = os.path.join(SUMO_HOME, 'bin', 'sumo')
    
    sumo_cmd = [
        sumo_binary,
        "-c", str(config_file),
        "--step-length", "1.0",
        "--no-warnings", "true",
        "--no-step-log", "true"
    ]
    
    # =====================================================
    # BASELINE (1 proceso)
    # =====================================================
    print("\n" + "-"*50)
    print(" Ejecutando BASELINE (1 proceso)...")
    print("-"*50)
    
    simulator.num_processes = 1
    if simulator.emission_processor:
        simulator.emission_processor.num_processes = 1
    
    try:
        result_1p = simulator.run_simulation(
            sumo_cmd=sumo_cmd,
            end_time=300,
            enable_rerouting=False,
            collect_emissions=True
        )
        time_1p = result_1p.total_time
        print(f"\n  Tiempo total: {time_1p:.2f}s")
        print(f"  Vehículos: {result_1p.total_vehicles}")
        print(f"  Viajes completados: {result_1p.completed_trips}")
        
    except Exception as e:
        print(f"\nError en simulación baseline: {e}")
        print("Ejecutando demo sin SUMO...")
        return run_demo_parallel_processing()
    
    # =====================================================
    # PARALELO (N procesos)
    # =====================================================
    print("\n" + "-"*50)
    print(f" Ejecutando con {NUM_PROCESSES} PROCESOS...")
    print("-"*50)
    
    simulator.num_processes = NUM_PROCESSES
    if simulator.emission_processor:
        simulator.emission_processor.num_processes = NUM_PROCESSES
    
    try:
        result_np = simulator.run_simulation(
            sumo_cmd=sumo_cmd,
            end_time=300,
            enable_rerouting=True,
            collect_emissions=True
        )
        time_np = result_np.total_time
        
        speedup = time_1p / time_np if time_np > 0 else 1.0
        efficiency = speedup / NUM_PROCESSES
        
        print(f"\n  Tiempo total: {time_np:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Eficiencia: {efficiency:.1%}")
        
    except Exception as e:
        print(f"\nError en simulación paralela: {e}")
        return run_demo_parallel_processing()
    
    # =====================================================
    # RESUMEN
    # =====================================================
    print("\n" + "="*70)
    print(" RESUMEN SIMULACIÓN SUMO")
    print("="*70)
    print(f"\n{'Métrica':<30} {'Valor':<20}")
    print("-"*50)
    print(f"{'Tiempo 1 proceso':<30} {time_1p:.2f}s")
    print(f"{'Tiempo {} procesos'.format(NUM_PROCESSES):<30} {time_np:.2f}s")
    print(f"{'Speedup':<30} {speedup:.2f}x")
    print(f"{'Eficiencia':<30} {efficiency:.1%}")
    print(f"{'Vehículos simulados':<30} {result_np.total_vehicles}")
    print(f"{'Viajes completados':<30} {result_np.completed_trips}")
    
    if result_np.total_emissions:
        print(f"\n{'Emisiones Totales:':<30}")
        print(f"  {'CO2':<26} {result_np.total_emissions.get('co2', 0)/1000:.2f} kg")
        print(f"  {'Combustible':<26} {result_np.total_emissions.get('fuel', 0):.2f} L")
    
    return {
        "baseline_time": time_1p,
        "parallel_time": time_np,
        "speedup": speedup,
        "efficiency": efficiency
    }


def test_sumo_routing():
    """
    Test del routing mejorado con datos del .net.xml
    """
    print("\n" + "="*70)
    print(" TEST: Routing con datos del .net.xml")
    print("="*70)
    
    net_file = PROJECT_ROOT / "networks" / "modified.net.xml"
    
    if not net_file.exists():
        print(f"\nError: No se encuentra {net_file}")
        return
    
    if not SUMO_ROUTING_AVAILABLE:
        print("\nMódulo sumo_routing no disponible.")
        print("Verifica que sumolib esté instalado.")
        return
    
    print(f"\n→ Parseando red: {net_file.name}...")
    
    try:
        network = SUMONetworkParser(str(net_file))
        print(f"  Edges cargados: {len(network.edges)}")
        print(f"  Junctions cargadas: {len(network.junctions)}")
        
        # Mostrar algunos edges
        print(f"\n  Primeros 5 edges:")
        for i, (edge_id, edge) in enumerate(list(network.edges.items())[:5]):
            print(f"    - {edge_id}: {edge.length:.0f}m, {edge.speed*3.6:.0f}km/h")
        
        # Crear router
        print("\n→ Creando router...")
        router = SUMORouter(network)
        
        # Probar algunas rutas
        edges = list(network.edges.keys())
        if len(edges) > 10:
            test_pairs = [
                (edges[0], edges[5]),
                (edges[10], edges[20]) if len(edges) > 20 else (edges[1], edges[3]),
            ]
            
            print("\n→ Calculando rutas de prueba...")
            for origin, dest in test_pairs:
                route, cost = router.find_route_dijkstra(origin, dest)
                if route:
                    print(f"\n  {origin} → {dest}:")
                    print(f"    Edges en ruta: {len(route)}")
                    print(f"    Costo (tiempo): {cost:.1f}s")
                else:
                    print(f"\n  {origin} → {dest}: Sin ruta encontrada")
        
        # Test paralelo
        print("\n→ Test de routing paralelo...")
        parallel_router = ParallelSUMORouter(
            network_parser=network,
            num_processes=4,
            algorithm="dijkstra"
        )
        
        # Generar pares O-D
        import numpy as np
        np.random.seed(42)
        
        valid_edges = [e for e in edges if not e.startswith(':')][:100]
        od_pairs = [
            (np.random.choice(valid_edges), np.random.choice(valid_edges))
            for _ in range(20)
        ]
        
        start = time.time()
        results = parallel_router.calculate_batch(od_pairs)
        elapsed = time.time() - start
        
        successful = sum(1 for r, c in results if r)
        print(f"\n  Pares O-D: {len(od_pairs)}")
        print(f"  Rutas exitosas: {successful}")
        print(f"  Tiempo: {elapsed:.3f}s")
        print(f"  Throughput: {len(od_pairs)/elapsed:.1f} rutas/s")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Función principal"""
    print("\n" + "█"*70)
    print(" TEST DE SIMULACIÓN PARALELA SUMO")
    print(" 10 vehículos · 4 procesos · Emisiones + Itinerarios")
    print("█"*70)
    
    # Test 1: Routing con .net.xml
    test_sumo_routing()
    
    # Test 2: Simulación (con o sin SUMO)
    if SUMO_AVAILABLE:
        results = run_sumo_simulation()
    else:
        results = run_demo_parallel_processing()
    
    print("\n" + "="*70)
    print(" ✓ TEST COMPLETADO")
    print("="*70)
    
    if results:
        print(f"\nResultados guardados. Speedup obtenido: {results.get('speedup', results.get('emission_speedup', 'N/A'))}x")
    
    return results


if __name__ == "__main__":
    main()


