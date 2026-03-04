#!/usr/bin/env python3
"""
Benchmark Simple de Emisiones Paralelas
Solo cálculo de emisiones con Pool persistente
"""

import time
from multiprocessing import Pool
import numpy as np

# Función GLOBAL para pickle
def calc_emisiones(batch):
    """Cálculo de emisiones HBEFA - 30 iteraciones por vehículo"""
    results = []
    for state in batch:
        speed = state["speed"]
        accel = state["accel"]
        
        # Base según velocidad
        if speed < 0.1:
            base = 2.5
        elif speed < 8.33:
            base = 180 * speed / 1000
        elif speed < 22.22:
            base = 150 * speed / 1000
        else:
            base = 170 * speed / 1000
        
        # Modelo HBEFA - 30 iteraciones
        co2 = 0.0
        for i in range(30):
            sf = 1.0 + 0.02 * np.sin(speed * 0.1 + i * 0.1)
            af = 1.0 + 0.1 * np.log1p(abs(accel)) if accel > 0 else 1.0
            co2 += base * sf * af / 30
            _ = np.exp(-speed * 0.01) * np.sqrt(abs(accel) + 1)
        
        results.append({"id": state["id"], "co2": co2})
    return results


def main():
    print("\n" + "="*60)
    print(" BENCHMARK EMISIONES - Pool Persistente")
    print("="*60)
    
    # Config
    NUM_STATES = 500000  # 500k estados
    BATCH_SIZE = 25000   # 25k por batch
    PROCS = [1, 2, 4]
    
    print(f"\n  Estados: {NUM_STATES:,}")
    print(f"  Batch size: {BATCH_SIZE:,}")
    
    # Generar datos UNA vez
    np.random.seed(42)
    print("\n  Generando datos...")
    
    all_states = [
        {"id": i, "speed": np.random.uniform(0, 30), "accel": np.random.uniform(-2, 2)}
        for i in range(NUM_STATES)
    ]
    
    # Crear batches
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
            with Pool(np_) as pool:
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
    print("\n" + "="*60)
    print(" RESULTADOS")
    print("="*60)
    print(f"\n  {'Procs':<8} {'Tiempo':<12} {'Speedup':<10} {'Eficiencia'}")
    print(f"  {'-'*45}")
    for p in PROCS:
        r = results[p]
        print(f"  {p:<8} {r['time']:.2f}s{'':<6} {r['speedup']:.2f}x{'':<5} {r['eff']:.0%}")
    
    s4 = results[4]["speedup"]
    print(f"\n  ★ Speedup con 4 procesos: {s4:.2f}x")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

