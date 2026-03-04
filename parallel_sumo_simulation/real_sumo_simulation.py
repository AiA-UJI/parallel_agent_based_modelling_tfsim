#!/usr/bin/env python3
"""
Simulación Real con SUMO + Cálculo Paralelo de Emisiones

Este script ejecuta una simulación SUMO real usando TraCI y cada 60 segundos
lanza el cálculo de emisiones en paralelo para comparar 1, 2 y 4 procesos.

Requisitos:
    - SUMO instalado (https://sumo.dlr.de/docs/Installing/index.html)
    - Variable SUMO_HOME configurada
    
    En Mac con Homebrew:
        brew install sumo
        export SUMO_HOME=/opt/homebrew/share/sumo
    
    En Linux:
        sudo apt install sumo sumo-tools
        export SUMO_HOME=/usr/share/sumo
"""

import os
import sys
import time
import random
import subprocess
import tempfile
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple
import numpy as np

# Configuración de paths
PROJECT_ROOT = Path(__file__).parent
NETWORKS_DIR = PROJECT_ROOT / "networks"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Verificar SUMO
SUMO_HOME = os.environ.get('SUMO_HOME')
SUMO_AVAILABLE = False

if SUMO_HOME:
    tools_path = os.path.join(SUMO_HOME, 'tools')
    if os.path.exists(tools_path):
        sys.path.append(tools_path)
        try:
            import traci
            import sumolib
            SUMO_AVAILABLE = True
            print(f"✓ SUMO encontrado en: {SUMO_HOME}")
        except ImportError:
            print("✗ No se pudo importar traci/sumolib")
else:
    print("✗ Variable SUMO_HOME no configurada")
    print("\nPara instalar SUMO:")
    print("  Mac:   brew install sumo && export SUMO_HOME=/opt/homebrew/share/sumo")
    print("  Linux: sudo apt install sumo sumo-tools && export SUMO_HOME=/usr/share/sumo")


# =====================================================
# FUNCIONES DE CÁLCULO DE EMISIONES
# =====================================================

def calculate_emissions_batch(vehicle_states: List[Dict]) -> List[Dict]:
    """
    Calcula emisiones para un batch de estados de vehículos.
    Modelo basado en HBEFA (Handbook Emission Factors for Road Transport).
    """
    results = []
    
    for state in vehicle_states:
        speed = state.get("speed", 0)  # m/s
        accel = state.get("acceleration", 0)  # m/s²
        waiting = state.get("waiting_time", 0)  # s
        
        # Distancia recorrida en el intervalo (asumiendo 1s por defecto)
        distance = speed * state.get("interval", 1.0)  # m
        
        # Clasificación de velocidad para factores de emisión
        if speed < 0.1:
            speed_class = "idle"
        elif speed < 8.33:  # < 30 km/h
            speed_class = "low"
        elif speed < 22.22:  # < 80 km/h
            speed_class = "medium"
        else:
            speed_class = "high"
        
        # Factores de emisión HBEFA (simplificados)
        emission_factors = {
            "idle": {"co2": 2.5, "co": 0.008, "hc": 0.001, "nox": 0.0005, "pmx": 0.00001, "fuel": 0.0008},
            "low": {"co2": 180, "co": 1.2, "hc": 0.08, "nox": 0.15, "pmx": 0.005, "fuel": 0.075},
            "medium": {"co2": 150, "co": 0.5, "hc": 0.03, "nox": 0.12, "pmx": 0.003, "fuel": 0.062},
            "high": {"co2": 170, "co": 0.8, "hc": 0.05, "nox": 0.25, "pmx": 0.004, "fuel": 0.070}
        }
        
        factors = emission_factors[speed_class]
        
        # Factor de aceleración
        if accel > 0:
            accel_factor = 1.0 + 0.15 * min(accel, 3.0)
        elif accel < -1:
            accel_factor = 0.85
        else:
            accel_factor = 1.0
        
        # Cálculo de emisiones
        distance_km = distance / 1000.0
        
        if speed_class == "idle":
            idle_time = max(1.0, waiting)
            co2 = factors["co2"] * idle_time * accel_factor
            co = factors["co"] * idle_time
            hc = factors["hc"] * idle_time
            nox = factors["nox"] * idle_time
            pmx = factors["pmx"] * idle_time
            fuel = factors["fuel"] * idle_time
        else:
            co2 = factors["co2"] * distance_km * accel_factor
            co = factors["co"] * distance_km * accel_factor
            hc = factors["hc"] * distance_km * accel_factor
            nox = factors["nox"] * distance_km * accel_factor
            pmx = factors["pmx"] * distance_km * accel_factor
            fuel = factors["fuel"] * distance_km * accel_factor
        
        # Cálculos adicionales para simular modelo HBEFA completo
        for _ in range(10):
            temp = 1.0 + 0.01 * np.sin(speed * 0.1)
            co2 *= temp
        
        results.append({
            "vehicle_id": state["vehicle_id"],
            "time": state["time"],
            "co2": co2,
            "co": co,
            "hc": hc,
            "nox": nox,
            "pmx": pmx,
            "fuel": fuel
        })
    
    return results


class ParallelEmissionCalculator:
    """Calculador de emisiones con soporte para procesamiento paralelo."""
    
    def __init__(self, num_processes: int = 4, batch_size: int = 1000):
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.total_time = 0.0
        self.total_processed = 0
    
    def calculate(self, vehicle_states: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Calcula emisiones para todos los estados.
        
        Returns:
            Tuple of (results, elapsed_time)
        """
        if not vehicle_states:
            return [], 0.0
        
        # Crear batches
        batches = []
        for i in range(0, len(vehicle_states), self.batch_size):
            batches.append(vehicle_states[i:i + self.batch_size])
        
        start_time = time.time()
        
        if self.num_processes == 1:
            # Secuencial
            all_results = []
            for batch in batches:
                all_results.extend(calculate_emissions_batch(batch))
        else:
            # Paralelo
            with Pool(processes=self.num_processes) as pool:
                batch_results = pool.map(calculate_emissions_batch, batches)
            all_results = [r for batch in batch_results for r in batch]
        
        elapsed = time.time() - start_time
        
        self.total_time += elapsed
        self.total_processed += len(vehicle_states)
        
        return all_results, elapsed


# =====================================================
# GENERADOR DE DEMANDA
# =====================================================

def generate_random_trips(net_file: str, output_file: str, num_vehicles: int, 
                          begin: float = 0, end: float = 10800, seed: int = 42):
    """
    Genera trips aleatorios usando randomTrips.py de SUMO.
    """
    random_trips = os.path.join(SUMO_HOME, 'tools', 'randomTrips.py')
    
    if not os.path.exists(random_trips):
        print(f"  ⚠ randomTrips.py no encontrado, generando trips manualmente...")
        return generate_trips_manual(net_file, output_file, num_vehicles, begin, end, seed)
    
    period = (end - begin) / num_vehicles
    
    cmd = [
        sys.executable, random_trips,
        '-n', net_file,
        '-o', output_file,
        '-b', str(begin),
        '-e', str(end),
        '-p', str(period),
        '--seed', str(seed),
        '--validate',
        '--route-file', output_file.replace('.trips.xml', '.rou.xml')
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"  ✓ Generados {num_vehicles} trips")
            return True
        else:
            print(f"  ⚠ Error generando trips: {result.stderr}")
            return generate_trips_manual(net_file, output_file, num_vehicles, begin, end, seed)
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        return generate_trips_manual(net_file, output_file, num_vehicles, begin, end, seed)


def generate_trips_manual(net_file: str, output_file: str, num_vehicles: int,
                          begin: float, end: float, seed: int):
    """
    Genera trips manualmente sin usar randomTrips.py
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Obtener edges de la red
    try:
        net = sumolib.net.readNet(net_file)
        edges = [e.getID() for e in net.getEdges() if not e.getID().startswith(':')]
    except:
        # Fallback: extraer edges del XML
        import xml.etree.ElementTree as ET
        tree = ET.parse(net_file)
        root = tree.getroot()
        edges = [e.get('id') for e in root.findall('.//edge') 
                 if e.get('id') and not e.get('id').startswith(':')]
    
    if len(edges) < 2:
        print("  ✗ No se encontraron edges suficientes en la red")
        return False
    
    # Generar trips
    trips = []
    for i in range(num_vehicles):
        depart = begin + (end - begin) * (i / num_vehicles)
        from_edge = random.choice(edges)
        to_edge = random.choice([e for e in edges if e != from_edge])
        
        trips.append(f'    <trip id="veh_{i}" depart="{depart:.2f}" from="{from_edge}" to="{to_edge}"/>')
    
    # Escribir archivo
    content = '''<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="50"/>
''' + '\n'.join(trips) + '''
</routes>
'''
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"  ✓ Generados {num_vehicles} trips manualmente")
    return True


# =====================================================
# SIMULACIÓN CON SUMO REAL
# =====================================================

def run_sumo_simulation(
    net_file: str,
    route_file: str,
    simulation_time: float = 10800,  # 3 horas
    emission_interval: int = 60,     # Calcular emisiones cada 60s
    process_counts: List[int] = [1, 2, 4],
    use_gui: bool = False
):
    """
    Ejecuta simulación SUMO real con cálculo paralelo de emisiones.
    
    Args:
        net_file: Archivo de red SUMO
        route_file: Archivo de rutas/trips
        simulation_time: Duración de simulación en segundos
        emission_interval: Intervalo para cálculo de emisiones
        process_counts: Lista de números de procesos a comparar
        use_gui: Usar SUMO-GUI en lugar de SUMO headless
    """
    if not SUMO_AVAILABLE:
        print("\n✗ SUMO no disponible. Ejecutando simulación simulada...")
        return run_simulated_sumo(simulation_time, emission_interval, process_counts)
    
    print("\n" + "="*70)
    print(" SIMULACIÓN SUMO REAL CON CÁLCULO PARALELO DE EMISIONES")
    print("="*70)
    
    # Configurar comando SUMO
    sumo_binary = os.path.join(SUMO_HOME, 'bin', 'sumo-gui' if use_gui else 'sumo')
    
    if not os.path.exists(sumo_binary):
        sumo_binary = 'sumo-gui' if use_gui else 'sumo'
    
    sumo_cmd = [
        sumo_binary,
        '-n', net_file,
        '-r', route_file,
        '--step-length', '1.0',
        '--no-warnings', 'true',
        '--no-step-log', 'true',
        '--time-to-teleport', '-1'
    ]
    
    results_by_process = {p: {"times": [], "states_processed": 0} for p in process_counts}
    
    for num_proc in process_counts:
        print(f"\n{'─'*60}")
        print(f" Ejecutando con {num_proc} proceso(s)...")
        print(f"{'─'*60}")
        
        # Iniciar SUMO
        traci.start(sumo_cmd)
        
        calculator = ParallelEmissionCalculator(num_processes=num_proc)
        
        current_time = 0
        step = 0
        collected_states = []
        all_emission_results = []
        total_emission_time = 0
        prev_speeds = {}
        
        print(f"\n  Simulando {simulation_time/3600:.1f} horas...")
        
        while current_time < simulation_time:
            # Avanzar simulación
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            step += 1
            
            # Obtener vehículos activos
            vehicle_ids = traci.vehicle.getIDList()
            
            # Recolectar estados
            for veh_id in vehicle_ids:
                try:
                    speed = traci.vehicle.getSpeed(veh_id)
                    prev_speed = prev_speeds.get(veh_id, speed)
                    accel = speed - prev_speed  # Aproximación
                    waiting = traci.vehicle.getWaitingTime(veh_id)
                    
                    collected_states.append({
                        "vehicle_id": veh_id,
                        "time": current_time,
                        "speed": speed,
                        "acceleration": accel,
                        "waiting_time": waiting,
                        "interval": 1.0
                    })
                    
                    prev_speeds[veh_id] = speed
                except:
                    pass
            
            # Calcular emisiones cada emission_interval segundos
            if step % emission_interval == 0 and collected_states:
                emit_results, emit_time = calculator.calculate(collected_states)
                all_emission_results.extend(emit_results)
                total_emission_time += emit_time
                results_by_process[num_proc]["times"].append(emit_time)
                results_by_process[num_proc]["states_processed"] += len(collected_states)
                
                # Progreso
                progress = current_time / simulation_time * 100
                print(f"\r  Tiempo: {current_time:.0f}s ({progress:.0f}%) | "
                      f"Vehículos: {len(vehicle_ids)} | "
                      f"Estados: {len(collected_states)} | "
                      f"Emisiones: {emit_time:.3f}s", end="", flush=True)
                
                collected_states = []
            
            # Limpiar vehículos que ya no existen
            current_ids = set(vehicle_ids)
            prev_speeds = {k: v for k, v in prev_speeds.items() if k in current_ids}
        
        # Procesar estados restantes
        if collected_states:
            emit_results, emit_time = calculator.calculate(collected_states)
            all_emission_results.extend(emit_results)
            total_emission_time += emit_time
        
        # Cerrar SUMO
        traci.close()
        
        # Calcular totales
        total_co2 = sum(r["co2"] for r in all_emission_results) / 1000  # kg
        total_fuel = sum(r["fuel"] for r in all_emission_results)
        
        print(f"\n\n  Resultados con {num_proc} proceso(s):")
        print(f"    - Tiempo total emisiones: {total_emission_time:.2f}s")
        print(f"    - Estados procesados: {results_by_process[num_proc]['states_processed']:,}")
        print(f"    - CO2 total: {total_co2:,.1f} kg")
        print(f"    - Combustible: {total_fuel:,.1f} L")
    
    # Comparación final
    print_comparison(results_by_process, process_counts)
    
    return results_by_process


# =====================================================
# SIMULACIÓN SIMULADA (SIN SUMO)
# =====================================================

def run_simulated_sumo(
    simulation_time: float = 10800,
    emission_interval: int = 60,
    process_counts: List[int] = [1, 2, 4],
    num_vehicles: int = 10000
):
    """
    Simula el comportamiento de una simulación SUMO para demostrar
    el procesamiento paralelo cuando SUMO no está disponible.
    """
    print("\n" + "="*70)
    print(" SIMULACIÓN SIMULADA (SUMO NO DISPONIBLE)")
    print(" Demostrando procesamiento paralelo de emisiones")
    print("="*70)
    
    print(f"\n  Configuración:")
    print(f"    - Tiempo simulación: {simulation_time/3600:.1f} horas")
    print(f"    - Intervalo emisiones: {emission_interval}s")
    print(f"    - Vehículos máximos: {num_vehicles:,}")
    print(f"    - Procesos a comparar: {process_counts}")
    
    np.random.seed(42)
    
    results_by_process = {p: {"times": [], "states_processed": 0, "total_co2": 0} for p in process_counts}
    
    for num_proc in process_counts:
        print(f"\n{'─'*60}")
        print(f" Ejecutando con {num_proc} proceso(s)...")
        print(f"{'─'*60}")
        
        calculator = ParallelEmissionCalculator(num_processes=num_proc, batch_size=2000)
        all_emission_results = []
        total_emission_time = 0
        
        # Simular cada intervalo
        num_intervals = int(simulation_time / emission_interval)
        
        for interval in range(num_intervals):
            current_time = interval * emission_interval
            
            # Número de vehículos activos (varía con el tiempo)
            hour = current_time / 3600
            if 0.5 <= hour <= 2.5:
                active_fraction = 0.8 + 0.2 * np.sin((hour - 0.5) * np.pi / 2)
            else:
                active_fraction = 0.4
            
            num_active = int(num_vehicles * active_fraction)
            
            # Generar estados de vehículos para este intervalo
            states = []
            for v in range(num_active):
                # Velocidad depende de congestión
                congestion = num_active / num_vehicles
                base_speed = np.random.uniform(5, 30)
                speed = base_speed * (1.0 - 0.5 * congestion)
                
                states.append({
                    "vehicle_id": f"veh_{v}",
                    "time": current_time,
                    "speed": max(0, speed + np.random.normal(0, 3)),
                    "acceleration": np.random.uniform(-2, 2),
                    "waiting_time": np.random.exponential(2) if speed < 2 else 0,
                    "interval": emission_interval
                })
            
            # Calcular emisiones
            emit_results, emit_time = calculator.calculate(states)
            all_emission_results.extend(emit_results)
            total_emission_time += emit_time
            results_by_process[num_proc]["times"].append(emit_time)
            results_by_process[num_proc]["states_processed"] += len(states)
            
            # Progreso
            progress = (interval + 1) / num_intervals * 100
            if interval % 10 == 0:
                print(f"\r  Progreso: {progress:.0f}% | Tiempo sim: {current_time:.0f}s | "
                      f"Vehículos: {num_active:,} | Emisiones: {emit_time:.3f}s", end="", flush=True)
        
        # Totales
        total_co2 = sum(r["co2"] for r in all_emission_results) / 1000
        total_fuel = sum(r["fuel"] for r in all_emission_results)
        results_by_process[num_proc]["total_co2"] = total_co2
        
        print(f"\n\n  Resultados con {num_proc} proceso(s):")
        print(f"    - Tiempo total emisiones: {total_emission_time:.2f}s")
        print(f"    - Estados procesados: {results_by_process[num_proc]['states_processed']:,}")
        print(f"    - CO2 total: {total_co2:,.1f} kg ({total_co2/1000:.2f} toneladas)")
        print(f"    - Combustible: {total_fuel:,.1f} litros")
    
    # Comparación
    print_comparison(results_by_process, process_counts)
    plot_results(results_by_process, process_counts)
    
    return results_by_process


def print_comparison(results: Dict, process_counts: List[int]):
    """Imprime tabla comparativa de resultados."""
    print("\n" + "="*70)
    print(" COMPARACIÓN DE RENDIMIENTO")
    print("="*70)
    
    baseline_time = sum(results[1]["times"])
    
    print(f"""
  ┌─────────────┬──────────────┬───────────┬────────────┬──────────────┐
  │  Procesos   │ Tiempo Total │  Speedup  │ Eficiencia │   Estados    │
  ├─────────────┼──────────────┼───────────┼────────────┼──────────────┤""")
    
    for p in process_counts:
        total_time = sum(results[p]["times"])
        speedup = baseline_time / total_time if total_time > 0 else 1.0
        efficiency = speedup / p
        states = results[p]["states_processed"]
        
        print(f"  │     {p:<7} │   {total_time:>7.2f}s  │   {speedup:>5.2f}x  │    {efficiency:>5.0%}   │  {states:>10,} │")
    
    print(f"  └─────────────┴──────────────┴───────────┴────────────┴──────────────┘")
    
    # Speedup obtenido
    s2 = baseline_time / sum(results[2]["times"]) if 2 in results else 0
    s4 = baseline_time / sum(results[4]["times"]) if 4 in results else 0
    
    print(f"""
  ╔════════════════════════════════════════════════════════════════════╗
  ║                      SPEEDUP OBTENIDO                              ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║   Con 2 procesos:  {s2:.2f}x más rápido                              ║
  ║   Con 4 procesos:  {s4:.2f}x más rápido                              ║
  ╚════════════════════════════════════════════════════════════════════╝
""")


def plot_results(results: Dict, process_counts: List[int]):
    """Genera gráfica de resultados."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Tiempos por intervalo
        ax1 = axes[0]
        for p in process_counts:
            times = results[p]["times"]
            ax1.plot(range(len(times)), times, label=f'{p} proceso(s)', marker='o', markersize=2)
        
        ax1.set_xlabel('Intervalo (cada 60s)', fontsize=11)
        ax1.set_ylabel('Tiempo de cálculo (s)', fontsize=11)
        ax1.set_title('Tiempo de Cálculo de Emisiones por Intervalo', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Speedup
        ax2 = axes[1]
        baseline = sum(results[1]["times"])
        speedups = [baseline / sum(results[p]["times"]) for p in process_counts]
        
        bars = ax2.bar(process_counts, speedups, color=['#3498db', '#2ecc71', '#e74c3c'][:len(process_counts)])
        ax2.plot(process_counts, process_counts, 'o--', color='gray', label='Ideal', linewidth=2)
        
        ax2.set_xlabel('Número de Procesos', fontsize=11)
        ax2.set_ylabel('Speedup', fontsize=11)
        ax2.set_title('Speedup del Cálculo Paralelo de Emisiones', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        for bar, s in zip(bars, speedups):
            ax2.annotate(f'{s:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = RESULTS_DIR / "sumo_parallel_emissions.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Gráfica guardada: {output_path}")
        
    except ImportError:
        print("  (matplotlib no disponible)")


# =====================================================
# MAIN
# =====================================================

def main():
    print("\n" + "█"*70)
    print(" SIMULACIÓN SUMO CON CÁLCULO PARALELO DE EMISIONES")
    print(" Comparación: 1 vs 2 vs 4 procesos (cada 60 segundos)")
    print("█"*70)
    
    # Configuración
    NET_FILE = str(NETWORKS_DIR / "modified.net.xml")
    ROUTE_FILE = str(NETWORKS_DIR / "demand_generated.rou.xml")
    SIMULATION_TIME = 10800  # 3 horas
    EMISSION_INTERVAL = 60   # Cada 60 segundos
    NUM_VEHICLES = 10000
    PROCESS_COUNTS = [1, 2, 4]
    
    print(f"\n  Configuración:")
    print(f"    - Red: {Path(NET_FILE).name}")
    print(f"    - Vehículos: {NUM_VEHICLES:,}")
    print(f"    - Duración: {SIMULATION_TIME/3600:.1f} horas")
    print(f"    - Intervalo cálculo emisiones: {EMISSION_INTERVAL}s")
    print(f"    - CPUs disponibles: {cpu_count()}")
    
    if SUMO_AVAILABLE:
        # Verificar red
        if not os.path.exists(NET_FILE):
            print(f"\n  ✗ Red no encontrada: {NET_FILE}")
            return
        
        # Generar demanda si no existe
        if not os.path.exists(ROUTE_FILE):
            print(f"\n  Generando demanda de tráfico...")
            generate_trips_manual(NET_FILE, ROUTE_FILE, NUM_VEHICLES, 0, SIMULATION_TIME, 42)
        
        # Ejecutar simulación real
        results = run_sumo_simulation(
            NET_FILE, ROUTE_FILE,
            simulation_time=SIMULATION_TIME,
            emission_interval=EMISSION_INTERVAL,
            process_counts=PROCESS_COUNTS
        )
    else:
        # Ejecutar simulación simulada
        results = run_simulated_sumo(
            simulation_time=SIMULATION_TIME,
            emission_interval=EMISSION_INTERVAL,
            process_counts=PROCESS_COUNTS,
            num_vehicles=NUM_VEHICLES
        )
    
    print("\n" + "="*70)
    print(" ✓ SIMULACIÓN COMPLETADA")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    main()


