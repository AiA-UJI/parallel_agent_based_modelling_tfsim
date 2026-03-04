# parallel_agent_based_modelling_tfsim

Código del artículo **"Parallel Agent-Based Modeling for Improving Traffic Flow Simulation"** (V. R. Tomás, M. Castillo, I. Monzón Catalán, L. A. García — Universitat Jaume I).

## Resumen

El trabajo aborda la paralelización a nivel de tareas de simulaciones SUMO (Simulation of Urban MObility) para reducir tiempos de ejecución sin modificar el motor de SUMO. El framework reparte el cálculo por vehículo y por segmento (emisiones, rutas dinámicas, estado de agentes) entre varios procesos, obteniendo speedups de hasta ~4× en corredores interurbanos y redes de autopista reales.

## Contenido del repositorio

- **`hgv.py`** — Utilidad para cargar y procesar datos DATEX2 (XML/gzip) de tráfico/velocidad.
- **`parallel_sumo_simulation/`** — Framework de simulación paralela:
  - Módulos: emisiones (HBEFA), routing (Dijkstra/A*, SUMO), simulador TraCI con workers.
  - Scripts: ejecución de simulaciones, generación de demanda, benchmarks y análisis de resultados.

Solo se versiona código Python de este paquete y `hgv.py`; el artículo, datos y entornos virtuales quedan fuera (véase `.gitignore`).

## Requisitos

- Python 3.9+
- SUMO con TraCI y `SUMO_HOME` configurado.

## Uso rápido

Desde la raíz del repo:

```bash
cd parallel_sumo_simulation
python scripts/run_simulation.py --network <red> --processes 4
python scripts/benchmark.py --mode quick
```

Ver documentación detallada en `parallel_sumo_simulation/README.md` si está incluida en el árbol que hayas clonado.
