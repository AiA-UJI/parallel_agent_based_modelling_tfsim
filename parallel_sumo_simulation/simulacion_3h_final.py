#!/usr/bin/env python3
"""
Simulación 3 Horas - 10,000 vehículos
Cálculo paralelo de emisiones cada 60 segundos
Corregido para macOS/Python 3.13
"""

import time
import multiprocessing as mp
from pathlib import Path
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def calc_emisiones(batch):
    """Cálculo HBEFA - 30 iteraciones por vehículo"""
    results = []
    for s in batch:
        speed, accel = s["speed"], s["accel"]
        
        base = 2.5 if speed < 0.1 else (180 if speed < 20 else 150) * speed / 1000
        co2 = 0.0
        
        for i in range(30):
            sf = 1.0 + 0.02 * np.sin(speed * 0.1 + i * 0.1)
            af = 1.0 + 0.1 * np.log1p(abs(accel)) if accel > 0 else 1.0
            co2 += base * sf * af / 30
            _ = np.exp(-speed * 0.01) * np.sqrt(abs(accel) + 1)
        
        results.append({"co2": co2})
    return results


def run_simulation():
    print("\n" + "█"*65)
    print(" SIMULACIÓN 3 HORAS - 10,000 VEHÍCULOS")
    print(" Cálculo paralelo de emisiones cada 60s")
    print("█"*65)
    
    # Config
    SIM_TIME = 10800      # 3 horas
    INTERVAL = 60         # cada 60s
    NUM_VEH = 10000
    BATCH_SIZE = 15000
    PROCS = [1, 2, 4]
    
    num_intervals = SIM_TIME // INTERVAL
    
    print(f"\n  Config: {SIM_TIME//3600}h | {INTERVAL}s intervalo | {NUM_VEH:,} veh")
    
    np.random.seed(42)
    
    # Generar datos
    print("  Generando datos...")
    all_intervals = []
    total_states = 0
    
    for i in range(num_intervals):
        hour = i * INTERVAL / 3600
        frac = 0.7 + 0.3 * np.sin(hour * np.pi / 1.5) if 0.5 <= hour <= 2.5 else 0.4
        n = int(NUM_VEH * frac)
        
        states = [
            {"speed": max(0, np.random.uniform(5, 35) * (1 - 0.3 * frac)),
             "accel": np.random.uniform(-2, 2)}
            for _ in range(n)
        ]
        all_intervals.append(states)
        total_states += n
    
    print(f"  Estados: {total_states:,} | Intervalos: {num_intervals}")
    
    results = {}
    ctx = mp.get_context('fork')  # Para macOS
    
    for np_ in PROCS:
        print(f"\n{'─'*65}")
        print(f" {np_} PROCESO(S)")
        print(f"{'─'*65}")
        
        pool = ctx.Pool(np_) if np_ > 1 else None
        times = []
        total_co2 = 0
        
        for idx, states in enumerate(all_intervals):
            batches = [states[i:i+BATCH_SIZE] for i in range(0, len(states), BATCH_SIZE)]
            
            start = time.time()
            if np_ == 1:
                res = [r for b in batches for r in calc_emisiones(b)]
            else:
                res = [r for br in pool.map(calc_emisiones, batches) for r in br]
            elapsed = time.time() - start
            
            times.append(elapsed)
            total_co2 += sum(r["co2"] for r in res)
            
            if idx % 30 == 0:
                print(f"\r  [{(idx+1)/num_intervals*100:5.1f}%] Estados: {len(states):,} | "
                      f"Tiempo: {elapsed:.3f}s", end="", flush=True)
        
        if pool:
            pool.close()
            pool.join()
        
        total_time = sum(times)
        results[np_] = {"time": total_time, "co2": total_co2/1000}
        print(f"\n  Total: {total_time:.2f}s | CO2: {total_co2/1000:,.0f} kg")
    
    # Resultados
    baseline = results[1]["time"]
    
    print("\n" + "="*65)
    print(" RESULTADOS")
    print("="*65)
    
    print(f"\n  {'Procs':<8} {'Tiempo':<12} {'Speedup':<10} {'Eficiencia'}")
    print(f"  {'-'*45}")
    for p in PROCS:
        sp = baseline / results[p]["time"]
        print(f"  {p:<8} {results[p]['time']:.2f}s{'':<5} {sp:.2f}x{'':<5} {sp/p*100:.0f}%")
    
    s2 = baseline / results[2]["time"]
    s4 = baseline / results[4]["time"]
    
    print(f"""
  ╔═════════════════════════════════════════════════════════════╗
  ║                    SPEEDUP OBTENIDO                         ║
  ╠═════════════════════════════════════════════════════════════╣
  ║   2 procesos:  {s2:.2f}x  ({baseline:.1f}s → {results[2]['time']:.1f}s)                 ║
  ║   4 procesos:  {s4:.2f}x  ({baseline:.1f}s → {results[4]['time']:.1f}s)                 ║
  ╚═════════════════════════════════════════════════════════════╝
""")
    
    # Gráfica
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        
        speedups = [baseline / results[p]["time"] for p in PROCS]
        bars = ax.bar(PROCS, speedups, color=['#3498db', '#2ecc71', '#e74c3c'], width=0.6)
        ax.plot(PROCS, PROCS, 'o--', color='gray', label='Ideal')
        
        ax.set_xlabel('Procesos', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title('Speedup - Simulación 3h, 10k vehículos\nCálculo paralelo cada 60s', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(PROCS)
        
        for bar, s in zip(bars, speedups):
            ax.annotate(f'{s:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        out = RESULTS_DIR / "speedup_3h_final.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  📊 {out}")
    except:
        pass
    
    print("="*65 + "\n")
    return results


if __name__ == '__main__':
    mp.freeze_support()
    run_simulation()

