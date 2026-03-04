#!/usr/bin/env python3
"""
Simulación con Step Length = 1 segundo
Procesa emisiones cada 60 segundos en paralelo
"""

import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict
import numpy as np

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def calculate_emissions_batch(vehicle_states: List[Dict]) -> List[Dict]:
    """Cálculo de emisiones modelo HBEFA."""
    results = []
    
    for state in vehicle_states:
        speed = state.get("speed", 0)
        accel = state.get("acceleration", 0)
        distance = speed * 1.0  # step_length = 1s
        
        # Factores HBEFA
        if speed < 0.1:
            base_co2, base_fuel = 2.5, 0.0008  # g/s, L/s
        elif speed < 8.33:
            base_co2, base_fuel = 180 * distance/1000, 0.075 * distance/1000
        elif speed < 22.22:
            base_co2, base_fuel = 150 * distance/1000, 0.062 * distance/1000
        else:
            base_co2, base_fuel = 170 * distance/1000, 0.070 * distance/1000
        
        # Modelo detallado (50 iteraciones)
        co2, fuel = 0, 0
        for i in range(50):
            temp = 0.95 + 0.05 * (1 - np.exp(-i * 0.1))
            sf = 1.0 + 0.02 * np.sin(speed * 0.1 + i * 0.1)
            af = 1.0 + 0.1 * np.log1p(abs(accel)) if accel > 0 else 1.0
            
            co2 += base_co2 * sf * af * temp / 50
            fuel += base_fuel * sf * af * temp / 50
            _ = np.exp(-speed * 0.01) * np.log1p(abs(accel) + 1)
        
        results.append({
            "vehicle_id": state["vehicle_id"],
            "time": state["time"],
            "co2": co2,
            "fuel": fuel
        })
    
    return results


def run_simulation(
    simulation_time: int = 10800,  # 3 horas en segundos
    step_length: int = 1,          # 1 segundo
    process_interval: int = 60,    # Procesar cada 60 segundos
    num_vehicles: int = 10000,
    process_counts: List[int] = [1, 2, 4]
):
    """
    Simulación con step_length=1 y procesamiento paralelo cada 60s.
    """
    print("\n" + "█"*70)
    print(" SIMULACIÓN: Step Length = 1s, Procesamiento cada 60s")
    print("█"*70)
    
    total_steps = simulation_time // step_length
    steps_per_process = process_interval // step_length
    num_processings = simulation_time // process_interval
    
    print(f"\n  Configuración:")
    print(f"    - Duración: {simulation_time/3600:.1f} horas ({simulation_time:,}s)")
    print(f"    - Step length: {step_length}s")
    print(f"    - Total steps: {total_steps:,}")
    print(f"    - Procesar cada: {process_interval}s ({steps_per_process} steps)")
    print(f"    - Número de procesamientos: {num_processings}")
    print(f"    - Vehículos: {num_vehicles:,}")
    print(f"    - CPUs: {cpu_count()}")
    
    np.random.seed(42)
    results_all = {}
    
    for num_proc in process_counts:
        print(f"\n{'═'*70}")
        print(f" EJECUTANDO CON {num_proc} PROCESO(S)")
        print(f"{'═'*70}")
        
        # Pool persistente
        pool = Pool(processes=num_proc) if num_proc > 1 else None
        
        processing_times = []
        total_states = 0
        all_emissions = []
        
        for proc_idx in range(num_processings):
            start_step = proc_idx * steps_per_process
            end_step = start_step + steps_per_process
            
            # Recolectar estados durante 60 segundos (60 steps)
            collected_states = []
            
            for step in range(start_step, end_step):
                t = step * step_length
                hour = t / 3600
                
                # Vehículos activos (varía con hora pico)
                if 0.5 <= hour <= 2.5:
                    active_frac = 0.7 + 0.3 * np.sin((hour - 0.5) * np.pi / 2)
                else:
                    active_frac = 0.4
                
                num_active = int(num_vehicles * active_frac)
                
                for v in range(num_active):
                    congestion = num_active / num_vehicles
                    speed = max(0, np.random.uniform(5, 30) * (1 - 0.5 * congestion))
                    
                    collected_states.append({
                        "vehicle_id": f"veh_{v}",
                        "time": t,
                        "speed": speed,
                        "acceleration": np.random.uniform(-2, 2)
                    })
            
            # PROCESAR EMISIONES EN PARALELO
            num_states = len(collected_states)
            
            # Crear batches
            batch_size = 5000
            batches = [collected_states[i:i+batch_size] 
                      for i in range(0, len(collected_states), batch_size)]
            
            calc_start = time.time()
            
            if num_proc == 1:
                results = []
                for batch in batches:
                    results.extend(calculate_emissions_batch(batch))
            else:
                batch_results = pool.map(calculate_emissions_batch, batches)
                results = [r for b in batch_results for r in b]
            
            calc_time = time.time() - calc_start
            processing_times.append(calc_time)
            total_states += num_states
            all_emissions.extend(results)
            
            # Progreso cada 10 intervalos
            if proc_idx % 10 == 0:
                progress = (proc_idx + 1) / num_processings * 100
                print(f"\r  Progreso: {progress:5.1f}% | Intervalo {proc_idx+1}/{num_processings} | "
                      f"Estados: {num_states:,} | Tiempo: {calc_time:.3f}s", end="", flush=True)
        
        if pool:
            pool.close()
            pool.join()
        
        # Totales
        total_time = sum(processing_times)
        total_co2 = sum(r["co2"] for r in all_emissions) / 1000
        total_fuel = sum(r["fuel"] for r in all_emissions)
        
        results_all[num_proc] = {
            "times": processing_times,
            "total_time": total_time,
            "total_states": total_states,
            "total_co2": total_co2,
            "total_fuel": total_fuel
        }
        
        print(f"\n\n  Resultados {num_proc} proceso(s):")
        print(f"    - Tiempo total: {total_time:.2f}s")
        print(f"    - Estados: {total_states:,}")
        print(f"    - CO2: {total_co2:,.1f} kg")
        print(f"    - Combustible: {total_fuel:,.1f} L")
    
    # COMPARACIÓN FINAL
    print("\n" + "═"*70)
    print(" COMPARACIÓN FINAL")
    print("═"*70)
    
    baseline = results_all[1]["total_time"]
    
    print(f"""
  ┌─────────────┬──────────────┬───────────┬────────────┬──────────────┐
  │  Procesos   │ Tiempo Total │  Speedup  │ Eficiencia │   Estados    │
  ├─────────────┼──────────────┼───────────┼────────────┼──────────────┤""")
    
    for p in process_counts:
        r = results_all[p]
        speedup = baseline / r["total_time"]
        eff = speedup / p
        print(f"  │     {p:<7} │   {r['total_time']:>7.2f}s  │   {speedup:>5.2f}x  │    {eff:>5.0%}   │  {r['total_states']:>10,} │")
    
    print("  └─────────────┴──────────────┴───────────┴────────────┴──────────────┘")
    
    s2 = baseline / results_all[2]["total_time"]
    s4 = baseline / results_all[4]["total_time"]
    
    print(f"""
  ╔════════════════════════════════════════════════════════════════════╗
  ║                      SPEEDUP OBTENIDO                              ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║   Con 2 procesos:  {s2:.2f}x más rápido                              ║
  ║   Con 4 procesos:  {s4:.2f}x más rápido                              ║
  ╚════════════════════════════════════════════════════════════════════╝
""")
    
    # Gráfica
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Speedup
        ax1 = axes[0]
        speedups = [baseline / results_all[p]["total_time"] for p in process_counts]
        bars = ax1.bar(process_counts, speedups, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax1.plot(process_counts, process_counts, 'o--', color='gray', label='Ideal')
        ax1.set_xlabel('Procesos')
        ax1.set_ylabel('Speedup')
        ax1.set_title('Speedup - Cálculo Paralelo de Emisiones\n(Step=1s, Proceso cada 60s)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        for bar, s in zip(bars, speedups):
            ax1.annotate(f'{s:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Tiempos
        ax2 = axes[1]
        times = [results_all[p]["total_time"] for p in process_counts]
        bars2 = ax2.bar(process_counts, times, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax2.set_xlabel('Procesos')
        ax2.set_ylabel('Tiempo (s)')
        ax2.set_title('Tiempo Total de Procesamiento')
        ax2.grid(True, alpha=0.3)
        for bar, t in zip(bars2, times):
            ax2.annotate(f'{t:.1f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        output = RESULTS_DIR / "speedup_step1_60s.png"
        plt.savefig(output, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 Gráfica: {output}")
        
    except ImportError:
        pass
    
    return results_all


if __name__ == "__main__":
    run_simulation(
        simulation_time=10800,   # 3 horas
        step_length=1,           # 1 segundo
        process_interval=60,     # Cada 60 segundos
        num_vehicles=10000,      # 10,000 vehículos
        process_counts=[1, 2, 4]
    )
    
    print("\n" + "═"*70)
    print(" ✓ COMPLETADO")
    print("═"*70 + "\n")


