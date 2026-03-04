#!/usr/bin/env python3
"""
Benchmark Rápido - Cálculos Externos Paralelos
1 hora simulación, 5000 vehículos
"""

import time
from multiprocessing import Pool
import numpy as np

def calc_todo(batch):
    """6 cálculos combinados - 100 iteraciones total por vehículo"""
    results = []
    for s in batch:
        speed, accel = s["speed"], s["accel"]
        
        co2, fuel, risk, noise = 0, 0, 0, 0
        
        for i in range(100):
            # Emisiones
            base = 180 if speed < 20 else 150
            co2 += base * speed/1000 * (1 + 0.02*np.sin(i*0.1)) / 100
            
            # Combustible
            fuel += (0.07 + 0.002*speed) * speed/1000 / 100
            
            # Riesgo
            risk += (speed/33)**2 * (1 + abs(accel)/3) / 100
            
            # Ruido
            noise += (30*np.log10(speed*3.6+1) + 20) / 100
            
            _ = np.exp(-speed*0.01) * np.sqrt(abs(accel)+1)
            _ = np.sin(speed*0.1+i) * np.cos(accel)
        
        results.append({"co2": co2, "fuel": fuel, "risk": risk, "noise": noise})
    return results


print("\n" + "="*60)
print(" BENCHMARK RÁPIDO - Cálculos Externos")
print("="*60)

SIM_TIME = 3600  # 1 hora
ACCUM = 300      # 5 min
NUM_VEH = 5000
BATCH = 15000
PROCS = [1, 2, 4]

np.random.seed(42)

# Generar datos
print(f"\n  Simulación: 1h | Vehículos: {NUM_VEH:,}")
print("  Generando datos...")

all_data = []
total = 0
for p in range(SIM_TIME // ACCUM):
    states = []
    for t in range(ACCUM):
        frac = 0.6 + 0.3 * np.sin(p * 0.5)
        n = int(NUM_VEH * frac)
        for v in range(n):
            states.append({
                "speed": np.random.uniform(15, 35),
                "accel": np.random.uniform(-1.5, 1.5)
            })
    all_data.append(states)
    total += len(states)

print(f"  Estados: {total:,}")

results = {}

for np_ in PROCS:
    print(f"\n  → {np_} proceso(s)...", end=" ", flush=True)
    
    pool = Pool(np_) if np_ > 1 else None
    times = []
    
    for states in all_data:
        batches = [states[i:i+BATCH] for i in range(0, len(states), BATCH)]
        
        start = time.time()
        if np_ == 1:
            res = [r for b in batches for r in calc_todo(b)]
        else:
            res = [r for br in pool.map(calc_todo, batches) for r in br]
        times.append(time.time() - start)
    
    if pool:
        pool.close()
        pool.join()
    
    total_time = sum(times)
    results[np_] = total_time
    print(f"Tiempo: {total_time:.2f}s")

baseline = results[1]
s2 = baseline / results[2]
s4 = baseline / results[4]

print(f"""
{'='*60}
 RESULTADOS
{'='*60}

  Procs    Tiempo      Speedup    Eficiencia
  ------------------------------------------
  1        {results[1]:.2f}s      1.00x      100%
  2        {results[2]:.2f}s      {s2:.2f}x      {s2/2*100:.0f}%
  4        {results[4]:.2f}s      {s4:.2f}x      {s4/4*100:.0f}%

  ★ Speedup 4 procesos: {s4:.2f}x

{'='*60}
""")

