#!/usr/bin/env python3
"""
Simulación de Emisiones 3 horas - 10,000 vehículos
Con procesamiento paralelo cada 60 segundos
"""

import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def calc_emisiones(batch):
    """Cálculo HBEFA - 30 iteraciones"""
    results = []
    for s in batch:
        speed, accel = s["speed"], s["accel"]
        
        if speed < 0.1:
            base = 2.5
        elif speed < 8.33:
            base = 180 * speed / 1000
        elif speed < 22.22:
            base = 150 * speed / 1000
        else:
            base = 170 * speed / 1000
        
        co2 = 0.0
        for i in range(30):
            sf = 1.0 + 0.02 * np.sin(speed * 0.1 + i * 0.1)
            af = 1.0 + 0.1 * np.log1p(abs(accel)) if accel > 0 else 1.0
            co2 += base * sf * af / 30
            _ = np.exp(-speed * 0.01) * np.sqrt(abs(accel) + 1)
        
        results.append({"co2": co2})
    return results


def run():
    print("\n" + "█"*65)
    print(" SIMULACIÓN 3 HORAS - 10,000 VEHÍCULOS")
    print(" Cálculo paralelo de emisiones cada 60 segundos")
    print("█"*65)
    
    # Config
    SIM_TIME = 10800      # 3 horas
    INTERVAL = 60         # Procesar cada 60s
    NUM_VEH = 10000
    BATCH_SIZE = 20000
    PROCS = [1, 2, 4]
    
    num_intervals = SIM_TIME // INTERVAL
    
    print(f"\n  Configuración:")
    print(f"    - Simulación: {SIM_TIME//3600}h ({SIM_TIME}s)")
    print(f"    - Intervalo: {INTERVAL}s")
    print(f"    - Intervalos: {num_intervals}")
    print(f"    - Vehículos: {NUM_VEH:,}")
    
    np.random.seed(42)
    
    # Pre-generar datos de cada intervalo
    print("\n  Generando datos de simulación...")
    all_intervals = []
    total_states = 0
    
    for i in range(num_intervals):
        t = i * INTERVAL
        hour = t / 3600
        
        # Patrón de tráfico
        if 0.5 <= hour <= 2.5:
            frac = 0.7 + 0.3 * np.sin((hour - 0.5) * np.pi / 2)
        else:
            frac = 0.4
        
        num_active = int(NUM_VEH * frac)
        
        states = [
            {"speed": max(0, np.random.uniform(3, 30) * (1 - 0.4 * frac)),
             "accel": np.random.uniform(-2, 2)}
            for _ in range(num_active)
        ]
        all_intervals.append(states)
        total_states += len(states)
    
    print(f"  Total estados: {total_states:,}")
    
    results = {}
    
    for np_ in PROCS:
        print(f"\n{'─'*65}")
        print(f" {np_} PROCESO(S)")
        print(f"{'─'*65}")
        
        pool = Pool(np_) if np_ > 1 else None
        times = []
        all_co2 = 0
        
        for idx, states in enumerate(all_intervals):
            batches = [states[i:i+BATCH_SIZE] for i in range(0, len(states), BATCH_SIZE)]
            
            start = time.time()
            if np_ == 1:
                res = []
                for b in batches:
                    res.extend(calc_emisiones(b))
            else:
                batch_res = pool.map(calc_emisiones, batches)
                res = [r for br in batch_res for r in br]
            elapsed = time.time() - start
            
            times.append(elapsed)
            all_co2 += sum(r["co2"] for r in res)
            
            if idx % 30 == 0:
                print(f"\r  Progreso: {(idx+1)/num_intervals*100:5.1f}% | "
                      f"Estados: {len(states):,} | Tiempo: {elapsed:.3f}s", end="")
        
        if pool:
            pool.close()
            pool.join()
        
        total_time = sum(times)
        results[np_] = {"time": total_time, "co2": all_co2/1000, "states": total_states}
        
        print(f"\n\n  Total: {total_time:.2f}s | CO2: {all_co2/1000:,.0f} kg")
    
    # Resultados
    baseline = results[1]["time"]
    
    print("\n" + "="*65)
    print(" RESULTADOS FINALES")
    print("="*65)
    
    print(f"\n  {'Procs':<8} {'Tiempo':<12} {'Speedup':<10} {'Eficiencia'}")
    print(f"  {'-'*45}")
    for p in PROCS:
        r = results[p]
        sp = baseline / r["time"]
        eff = sp / p
        print(f"  {p:<8} {r['time']:.2f}s{'':<5} {sp:.2f}x{'':<5} {eff:.0%}")
    
    s2 = baseline / results[2]["time"]
    s4 = baseline / results[4]["time"]
    
    print(f"""
  ╔═══════════════════════════════════════════════════════════════╗
  ║                    SPEEDUP OBTENIDO                           ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║   2 procesos:  {s2:.2f}x  ({results[1]['time']:.1f}s → {results[2]['time']:.1f}s)                    ║
  ║   4 procesos:  {s4:.2f}x  ({results[1]['time']:.1f}s → {results[4]['time']:.1f}s)                    ║
  ╚═══════════════════════════════════════════════════════════════╝
""")
    
    # Gráfica
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        
        speedups = [baseline / results[p]["time"] for p in PROCS]
        bars = ax.bar(PROCS, speedups, color=['#3498db', '#2ecc71', '#e74c3c'], width=0.6)
        ax.plot(PROCS, PROCS, 'o--', color='gray', label='Ideal', linewidth=2)
        
        ax.set_xlabel('Número de Procesos', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title('Speedup - Cálculo Paralelo de Emisiones\n(10,000 vehículos, 3 horas, proceso cada 60s)', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(PROCS)
        
        for bar, s in zip(bars, speedups):
            ax.annotate(f'{s:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        out = RESULTS_DIR / "speedup_emisiones_3h.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  📊 {out}")
    except:
        pass
    
    print("="*65 + "\n")


if __name__ == "__main__":
    run()

