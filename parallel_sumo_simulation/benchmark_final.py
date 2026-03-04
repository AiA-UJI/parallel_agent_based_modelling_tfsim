#!/usr/bin/env python3
"""
Benchmark Final - Corregido para macOS
"""

import time
import multiprocessing as mp
import numpy as np

# Función GLOBAL (antes de main)
def calc_emisiones(batch):
    """Cálculo HBEFA - 30 iteraciones"""
    results = []
    for s in batch:
        speed, accel = s["speed"], s["accel"]
        
        base = 180 if speed < 20 else 150
        co2 = 0.0
        
        for i in range(30):
            sf = 1.0 + 0.02 * np.sin(speed * 0.1 + i * 0.1)
            af = 1.0 + 0.1 * np.log1p(abs(accel)) if accel > 0 else 1.0
            co2 += base * speed / 1000 * sf * af / 30
            _ = np.exp(-speed * 0.01) * np.sqrt(abs(accel) + 1)
        
        results.append({"co2": co2})
    return results


def main():
    print("\n" + "="*60)
    print(" BENCHMARK FINAL - Emisiones Paralelas")
    print("="*60)
    
    # Config reducida para benchmark rápido
    NUM_STATES = 500000  # 500k estados
    BATCH_SIZE = 25000
    PROCS = [1, 2, 4]
    
    print(f"\n  Estados: {NUM_STATES:,}")
    print(f"  Batch: {BATCH_SIZE:,}")
    
    # Generar datos
    np.random.seed(42)
    print("  Generando datos...")
    
    all_states = [
        {"speed": np.random.uniform(5, 35), "accel": np.random.uniform(-2, 2)}
        for _ in range(NUM_STATES)
    ]
    
    batches = [all_states[i:i+BATCH_SIZE] for i in range(0, NUM_STATES, BATCH_SIZE)]
    print(f"  Batches: {len(batches)}")
    
    results = {}
    baseline = None
    
    for np_ in PROCS:
        print(f"\n  → {np_} proceso(s)...", end=" ", flush=True)
        
        start = time.time()
        
        if np_ == 1:
            res = []
            for b in batches:
                res.extend(calc_emisiones(b))
        else:
            with mp.Pool(np_) as pool:
                batch_res = pool.map(calc_emisiones, batches)
            res = [r for br in batch_res for r in br]
        
        elapsed = time.time() - start
        
        if np_ == 1:
            baseline = elapsed
        
        speedup = baseline / elapsed
        eff = speedup / np_
        
        results[np_] = {"time": elapsed, "speedup": speedup, "eff": eff}
        print(f"Tiempo: {elapsed:.2f}s | Speedup: {speedup:.2f}x | Eff: {eff:.0%}")
    
    # Resumen
    s2 = results[2]["speedup"]
    s4 = results[4]["speedup"]
    
    print(f"""
{'='*60}
 RESULTADOS
{'='*60}

  Procs    Tiempo      Speedup    Eficiencia
  ------------------------------------------
  1        {results[1]['time']:.2f}s      1.00x      100%
  2        {results[2]['time']:.2f}s      {s2:.2f}x      {s2/2*100:.0f}%
  4        {results[4]['time']:.2f}s      {s4:.2f}x      {s4/4*100:.0f}%

  ★ Speedup con 4 procesos: {s4:.2f}x

{'='*60}
""")
    
    # Guardar gráfica
    try:
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        fig, ax = plt.subplots(figsize=(10, 6))
        speedups = [results[p]["speedup"] for p in PROCS]
        bars = ax.bar(PROCS, speedups, color=['#3498db', '#2ecc71', '#e74c3c'], width=0.6)
        ax.plot(PROCS, PROCS, 'o--', color='gray', label='Ideal')
        ax.set_xlabel('Procesos')
        ax.set_ylabel('Speedup')
        ax.set_title('Speedup - Cálculo Paralelo de Emisiones')
        ax.legend()
        ax.grid(True, alpha=0.3)
        for bar, s in zip(bars, speedups):
            ax.annotate(f'{s:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        out = Path(__file__).parent / "results" / "speedup_final.png"
        out.parent.mkdir(exist_ok=True)
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  📊 {out}")
    except:
        pass


if __name__ == '__main__':
    mp.freeze_support()  # Para macOS
    main()
