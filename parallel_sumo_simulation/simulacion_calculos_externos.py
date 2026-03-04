#!/usr/bin/env python3
"""
Simulación con Cálculos Externos Pesados para Autovías/Nacionales

Cálculos paralelos:
1. Emisiones HBEFA completo (6 contaminantes)
2. Modelo de consumo de combustible
3. Predicción de congestión (modelo de flujo)
4. Análisis de riesgo de accidentes
5. Cálculo de ruido ambiental
6. Costes operativos

Se acumulan datos de 5 minutos (300s) antes de procesar
"""

import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# =====================================================
# CÁLCULOS EXTERNOS PESADOS (funciones globales)
# =====================================================

def calc_emisiones_hbefa(batch):
    """
    Modelo HBEFA completo - 6 contaminantes
    50 iteraciones por vehículo para modelo detallado
    """
    results = []
    for s in batch:
        speed, accel, grade = s["speed"], s["accel"], s.get("grade", 0)
        
        # Potencia específica del vehículo (VSP)
        vsp = speed * (1.1 * accel + 9.81 * grade + 0.132) + 0.000302 * speed**3
        
        # Factores base por velocidad
        if speed < 0.1:
            factors = {"co2": 2.5, "nox": 0.001, "pm": 0.00005, "co": 0.01, "hc": 0.002, "fuel": 0.0008}
        elif speed < 15:
            factors = {"co2": 250, "nox": 0.8, "pm": 0.04, "co": 2.0, "hc": 0.15, "fuel": 0.10}
        elif speed < 25:
            factors = {"co2": 180, "nox": 0.5, "pm": 0.025, "co": 1.0, "hc": 0.08, "fuel": 0.075}
        else:
            factors = {"co2": 200, "nox": 0.7, "pm": 0.03, "co": 1.2, "hc": 0.10, "fuel": 0.082}
        
        emissions = {k: 0.0 for k in factors}
        
        # Modelo iterativo HBEFA
        for i in range(50):
            temp_factor = 0.9 + 0.1 * (1 - np.exp(-i * 0.05))
            vsp_factor = 1.0 + 0.1 * np.tanh(vsp / 10)
            speed_var = 1.0 + 0.02 * np.sin(speed * 0.1 + i * 0.1)
            
            for pol in emissions:
                base = factors[pol] * speed / 1000 if speed > 0.1 else factors[pol]
                emissions[pol] += base * temp_factor * vsp_factor * speed_var / 50
            
            _ = np.exp(-speed * 0.01) * np.log1p(abs(vsp) + 1)
        
        results.append(emissions)
    return results


def calc_consumo_combustible(batch):
    """
    Modelo detallado de consumo de combustible
    Considera: velocidad, aceleración, pendiente, masa, resistencia
    """
    results = []
    for s in batch:
        speed, accel = s["speed"], s["accel"]
        mass = 1500  # kg
        cd = 0.3     # coef aerodinámico
        area = 2.2   # m²
        rho = 1.225  # densidad aire
        cr = 0.01    # coef rodadura
        
        fuel = 0.0
        for i in range(40):
            # Fuerzas
            f_aero = 0.5 * rho * cd * area * (speed + 0.5 * np.sin(i * 0.1))**2
            f_roll = cr * mass * 9.81
            f_accel = mass * accel * (1 + 0.01 * i)
            f_total = f_aero + f_roll + f_accel
            
            # Potencia y consumo
            power = max(0, f_total * speed)
            efficiency = 0.25 + 0.1 * np.tanh(power / 50000)
            fuel += (power / (efficiency * 34.2e6)) / 40  # L
            
            _ = np.sqrt(f_total**2 + power) * np.exp(-speed * 0.001)
        
        results.append({"fuel_l": fuel, "power_kw": power / 1000})
    return results


def calc_prediccion_congestion(batch):
    """
    Modelo de predicción de congestión (flujo vehicular)
    Basado en modelo LWR (Lighthill-Whitham-Richards)
    """
    results = []
    
    # Parámetros de la vía
    v_free = 33.33   # velocidad libre m/s (120 km/h)
    k_jam = 0.15     # densidad de atasco veh/m
    
    for s in batch:
        speed = s["speed"]
        density = s.get("density", 0.05)  # veh/m
        
        congestion = 0.0
        travel_time = 0.0
        
        for i in range(35):
            # Modelo fundamental
            v_model = v_free * (1 - (density / k_jam)**2)
            flow = density * v_model
            
            # Índice de congestión
            cong_idx = 1 - (speed / v_free) if v_free > 0 else 1
            congestion += cong_idx / 35
            
            # Tiempo de viaje estimado
            segment_length = 1000  # m
            tt = segment_length / max(speed, 0.1)
            travel_time += tt / 35
            
            # Propagación de onda
            wave_speed = v_free * (1 - 2 * density / k_jam)
            _ = np.sin(wave_speed * 0.1 + i) * np.cos(flow * 0.001)
            _ = np.exp(-cong_idx) * np.log1p(tt)
        
        results.append({"congestion_idx": congestion, "travel_time": travel_time})
    return results


def calc_riesgo_accidentes(batch):
    """
    Modelo de análisis de riesgo de accidentes
    Basado en: velocidad, densidad, variabilidad de velocidades
    """
    results = []
    
    for s in batch:
        speed = s["speed"]
        accel = s["accel"]
        density = s.get("density", 0.05)
        
        risk = 0.0
        for i in range(30):
            # Factor de velocidad (riesgo aumenta exponencialmente)
            speed_risk = (speed / 33.33)**2.5 if speed > 0 else 0
            
            # Factor de densidad
            density_risk = density / 0.15
            
            # Factor de maniobras bruscas
            maneuver_risk = abs(accel) / 3.0
            
            # Time-to-collision estimado
            ttc = 2.0 / (density * max(speed, 0.1) + 0.1)
            ttc_risk = np.exp(-ttc / 2)
            
            risk += (0.3 * speed_risk + 0.3 * density_risk + 
                    0.2 * maneuver_risk + 0.2 * ttc_risk) / 30
            
            _ = np.sqrt(speed_risk * density_risk) * np.tanh(ttc_risk)
            _ = np.exp(-risk * i * 0.01) * np.log1p(ttc + 1)
        
        severity = "LOW" if risk < 0.3 else "MEDIUM" if risk < 0.6 else "HIGH"
        results.append({"risk_score": risk, "severity": severity})
    return results


def calc_ruido_ambiental(batch):
    """
    Modelo de emisión de ruido (dB)
    Basado en CNOSSOS-EU
    """
    results = []
    
    for s in batch:
        speed = s["speed"]
        accel = s["accel"]
        
        noise_db = 0.0
        for i in range(25):
            speed_kmh = speed * 3.6
            
            # Ruido de rodadura
            l_rolling = 30 * np.log10(speed_kmh + 1) + 20 + 2 * np.sin(i * 0.2)
            
            # Ruido de propulsión
            l_propulsion = 23 + 0.1 * speed_kmh + abs(accel) * 3
            
            # Combinación energética
            l_total = 10 * np.log10(10**(l_rolling/10) + 10**(l_propulsion/10))
            noise_db += l_total / 25
            
            _ = np.exp(-l_total * 0.01) * np.sqrt(speed_kmh + 1)
        
        results.append({"noise_db": noise_db})
    return results


def calc_costes_operativos(batch):
    """
    Modelo de costes operativos del vehículo
    """
    results = []
    
    fuel_price = 1.5  # €/L
    maintenance_per_km = 0.05  # €/km
    depreciation_per_km = 0.08  # €/km
    
    for s in batch:
        speed = s["speed"]
        accel = s["accel"]
        distance = speed * 1.0  # 1 segundo
        
        cost = 0.0
        for i in range(20):
            # Consumo instantáneo
            if speed < 0.1:
                fuel_rate = 0.0008  # L/s idle
            else:
                fuel_rate = (0.07 + 0.002 * speed + 0.01 * abs(accel)) * speed / 1000
            
            fuel_cost = fuel_rate * fuel_price
            maint_cost = distance / 1000 * maintenance_per_km
            depr_cost = distance / 1000 * depreciation_per_km
            
            cost += (fuel_cost + maint_cost + depr_cost) / 20
            
            _ = np.log1p(cost + fuel_cost) * np.sqrt(speed + 1)
        
        results.append({"cost_eur": cost})
    return results


# Función que ejecuta TODOS los cálculos para un batch
def procesar_batch_completo(batch):
    """Ejecuta los 6 cálculos para un batch"""
    r1 = calc_emisiones_hbefa(batch)
    r2 = calc_consumo_combustible(batch)
    r3 = calc_prediccion_congestion(batch)
    r4 = calc_riesgo_accidentes(batch)
    r5 = calc_ruido_ambiental(batch)
    r6 = calc_costes_operativos(batch)
    
    # Combinar resultados
    combined = []
    for i in range(len(batch)):
        combined.append({
            **r1[i], **r2[i], **r3[i], **r4[i], **r5[i], **r6[i]
        })
    return combined


def run():
    print("\n" + "█"*70)
    print(" SIMULACIÓN CON CÁLCULOS EXTERNOS PESADOS")
    print(" Autovías/Nacionales - Sin semáforos")
    print("█"*70)
    
    # Config
    SIM_TIME = 10800      # 3 horas
    ACCUMULATE = 300      # Acumular 5 minutos antes de procesar
    NUM_VEH = 10000
    BATCH_SIZE = 10000
    PROCS = [1, 2, 4]
    
    num_processings = SIM_TIME // ACCUMULATE
    
    print(f"\n  Configuración:")
    print(f"    - Simulación: {SIM_TIME//3600}h")
    print(f"    - Acumular: {ACCUMULATE}s (5 min) antes de procesar")
    print(f"    - Procesamientos: {num_processings}")
    print(f"    - Vehículos: {NUM_VEH:,}")
    print(f"    - Batch size: {BATCH_SIZE:,}")
    print(f"\n  Cálculos por batch:")
    print(f"    1. Emisiones HBEFA (50 iter)")
    print(f"    2. Consumo combustible (40 iter)")
    print(f"    3. Predicción congestión (35 iter)")
    print(f"    4. Riesgo accidentes (30 iter)")
    print(f"    5. Ruido ambiental (25 iter)")
    print(f"    6. Costes operativos (20 iter)")
    
    np.random.seed(42)
    
    # Pre-generar datos
    print("\n  Generando datos...")
    all_data = []
    total_states = 0
    
    for p in range(num_processings):
        interval_states = []
        for t in range(ACCUMULATE):
            sim_time = p * ACCUMULATE + t
            hour = sim_time / 3600
            
            if 0.5 <= hour <= 2.5:
                frac = 0.7 + 0.3 * np.sin((hour - 0.5) * np.pi / 2)
            else:
                frac = 0.4
            
            num_active = int(NUM_VEH * frac)
            density = num_active / (NUM_VEH * 10)  # densidad normalizada
            
            for v in range(num_active):
                speed = max(0, np.random.uniform(20, 35) * (1 - 0.3 * frac))
                interval_states.append({
                    "speed": speed,
                    "accel": np.random.uniform(-1.5, 1.5),
                    "grade": np.random.uniform(-0.03, 0.03),
                    "density": density
                })
        
        all_data.append(interval_states)
        total_states += len(interval_states)
    
    print(f"  Estados totales: {total_states:,}")
    print(f"  Estados por procesamiento: ~{total_states // num_processings:,}")
    
    results = {}
    
    for np_ in PROCS:
        print(f"\n{'═'*70}")
        print(f" {np_} PROCESO(S)")
        print(f"{'═'*70}")
        
        pool = Pool(np_) if np_ > 1 else None
        times = []
        totals = {"co2": 0, "fuel": 0, "noise": 0, "cost": 0, "risk": 0}
        
        for idx, states in enumerate(all_data):
            batches = [states[i:i+BATCH_SIZE] for i in range(0, len(states), BATCH_SIZE)]
            
            start = time.time()
            if np_ == 1:
                res = []
                for b in batches:
                    res.extend(procesar_batch_completo(b))
            else:
                batch_res = pool.map(procesar_batch_completo, batches)
                res = [r for br in batch_res for r in br]
            elapsed = time.time() - start
            
            times.append(elapsed)
            totals["co2"] += sum(r["co2"] for r in res)
            totals["fuel"] += sum(r["fuel_l"] for r in res)
            totals["noise"] += sum(r["noise_db"] for r in res) / len(res)
            totals["cost"] += sum(r["cost_eur"] for r in res)
            totals["risk"] += sum(r["risk_score"] for r in res) / len(res)
            
            throughput = len(states) / elapsed if elapsed > 0 else 0
            print(f"\r  [{(idx+1)/num_processings*100:5.1f}%] Tiempo: {elapsed:.2f}s | "
                  f"Throughput: {throughput:,.0f}/s", end="")
        
        if pool:
            pool.close()
            pool.join()
        
        total_time = sum(times)
        results[np_] = {"time": total_time, "totals": totals}
        
        print(f"\n\n  Tiempo total: {total_time:.2f}s")
        print(f"  CO2: {totals['co2']/1000:,.0f} kg | Fuel: {totals['fuel']:,.0f} L")
    
    # Resultados finales
    baseline = results[1]["time"]
    
    print("\n" + "="*70)
    print(" RESULTADOS FINALES")
    print("="*70)
    
    print(f"\n  {'Procs':<8} {'Tiempo':<12} {'Speedup':<12} {'Eficiencia'}")
    print(f"  {'-'*50}")
    for p in PROCS:
        r = results[p]
        sp = baseline / r["time"]
        eff = sp / p
        print(f"  {p:<8} {r['time']:.2f}s{'':<5} {sp:.2f}x{'':<7} {eff:.0%}")
    
    s2 = baseline / results[2]["time"]
    s4 = baseline / results[4]["time"]
    
    print(f"""
  ╔════════════════════════════════════════════════════════════════════╗
  ║                      SPEEDUP OBTENIDO                              ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║   2 procesos:  {s2:.2f}x  ({baseline:.1f}s → {results[2]['time']:.1f}s)                       ║
  ║   4 procesos:  {s4:.2f}x  ({baseline:.1f}s → {results[4]['time']:.1f}s)                       ║
  ╚════════════════════════════════════════════════════════════════════╝
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
        ax.set_title('Speedup - Cálculos Externos Paralelos\n(Emisiones + Combustible + Congestión + Riesgo + Ruido + Costes)', 
                    fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(PROCS)
        
        for bar, s in zip(bars, speedups):
            ax.annotate(f'{s:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        out = RESULTS_DIR / "speedup_calculos_externos.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  📊 {out}")
    except:
        pass
    
    print("="*70 + "\n")


if __name__ == "__main__":
    run()

