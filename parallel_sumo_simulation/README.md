# Parallel SUMO Simulation Framework

Framework para ejecutar simulaciones de tráfico SUMO con cálculo paralelo de **emisiones** e **itinerarios**, permitiendo obtener speedup significativo en sistemas multicore.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    SUMO Simulation (TraCI)                       │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Step Loop   │───▶│ Data Collect │───▶│ Batch States │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │              │
└──────────────────────────────────────────────────│──────────────┘
                                                   │
                    ┌──────────────────────────────┴──────────────────────┐
                    │              PARALLEL PROCESSING                     │
                    │                                                      │
                    │  ┌─────────────┐         ┌─────────────┐           │
                    │  │  Worker 1   │         │  Worker N   │           │
                    │  │ Emissions   │   ...   │ Emissions   │           │
                    │  └─────────────┘         └─────────────┘           │
                    │                                                      │
                    │  ┌─────────────┐         ┌─────────────┐           │
                    │  │  Worker 1   │         │  Worker N   │           │
                    │  │  Routing    │   ...   │  Routing    │           │
                    │  └─────────────┘         └─────────────┘           │
                    │                                                      │
                    └──────────────────────────────────────────────────────┘
```

## Componentes Principales

### 1. Módulo de Emisiones (`modules/emissions.py`)
- **EmissionCalculator**: Calcula emisiones por vehículo basado en modelo HBEFA
- **ParallelEmissionProcessor**: Procesa lotes de estados de vehículos en paralelo
- Pollutantes: CO2, CO, HC, NOx, PMx, consumo de combustible

### 2. Módulo de Rutas (`modules/routing.py` y `modules/sumo_routing.py`)
- **RouteCalculator**: Algoritmos Dijkstra y A*
- **SUMORouter**: Usa datos completos del `.net.xml` de SUMO
- **ParallelRouteProcessor**: Cálculo paralelo de rutas
- **DynamicRerouter**: Re-enrutamiento dinámico basado en congestión

### 3. Simulador Principal (`modules/simulation.py`)
- **ParallelSUMOSimulator**: Integra TraCI con procesamiento paralelo
- Soporte para procesamiento asíncrono
- Manejo de accidentes y eventos

## Sobre el Cálculo de Rutas con `.net.xml`

### ¿Qué información usa?

El archivo `.net.xml` de SUMO contiene:

| Dato | Disponible | Uso en Routing |
|------|------------|----------------|
| Topología (edges, nodes) | ✅ | Grafo de la red |
| Longitud de edges | ✅ | Costo por distancia |
| Velocidad límite | ✅ | Tiempo de flujo libre |
| Conexiones (giros) | ✅ | Restricciones de camino |
| Semáforos | ✅ | (Solo programas, no estado) |
| Tráfico actual | ❌ | Requiere TraCI |

### Estrategia de Routing

1. **Inicial**: Tiempos de flujo libre (`tiempo = longitud / velocidad_limite`)
2. **Durante simulación**: Actualización desde TraCI con `edge.getTraveltime()`
3. **Suavizado temporal**: Media ponderada de últimos N valores

### Calidad de los Resultados

| Escenario | Precisión | Notas |
|-----------|-----------|-------|
| Red sin congestión | ⭐⭐⭐⭐⭐ | Excelente |
| Congestión ligera | ⭐⭐⭐⭐ | Buena con actualizaciones TraCI |
| Congestión severa | ⭐⭐⭐ | Requiere re-enrutamiento frecuente |
| Con accidentes | ⭐⭐⭐ | Depende de velocidad de detección |

### Opciones de Routing

```python
# Opción 1: Router interno (rápido, paralelo)
router = SUMORouter(network_parser)
route, cost = router.find_route_astar(from_edge, to_edge)

# Opción 2: DUAROUTER de SUMO (más preciso, más lento)
route, cost = router.find_route_duarouter(from_edge, to_edge)

# Opción 3: Routing paralelo con actualización TraCI
parallel_router = ParallelSUMORouter(network_parser, num_processes=4)
updater = TraCIRouteUpdater(network_parser)

# Durante simulación:
updater.update_from_traci(traci_connection, current_time)
routes = parallel_router.calculate_batch(od_pairs)
```

## Instalación

### Requisitos
- Python 3.9+
- SUMO 1.14+ con TraCI
- Variable de entorno `SUMO_HOME`

### Instalación

```bash
cd parallel_sumo_simulation

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o: venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### 1. Crear Red de Prueba

```bash
# Crear archivos de definición de red
python scripts/run_simulation.py --create-network test_grid

# Generar red con netconvert (SUMO)
cd networks/test_grid
netconvert -n test_grid.nod.xml -e test_grid.edg.xml -o test_grid.net.xml
```

### 2. Generar Demanda de Tráfico

```bash
python scripts/generate_demand.py --network test_grid --level all --time 3600
```

### 3. Ejecutar Simulación

```bash
# Simulación básica
python scripts/run_simulation.py --network test_grid --processes 4 --traffic medium

# Con accidentes
python scripts/run_simulation.py --network test_grid --processes 8 --accidents 1

# Con más opciones
python scripts/run_simulation.py \
    --network test_grid \
    --processes 8 \
    --traffic high \
    --time 3600 \
    --accidents 1
```

### 4. Ejecutar Benchmark

```bash
# Benchmark simulado (sin SUMO real)
python scripts/benchmark.py --mode simulated

# Benchmark rápido
python scripts/benchmark.py --mode quick --machine "Mi Máquina"

# Benchmark completo
python scripts/benchmark.py --mode full --machine "HPC Node" \
    --processes 1 2 4 8 16 32 \
    --repetitions 3
```

### 5. Analizar Resultados

```bash
# Generar datos de ejemplo y gráficas
python scripts/analyze_results.py --create-sample

# Analizar resultados de benchmark
python scripts/analyze_results.py --input results/benchmark_results.csv

# Solo gráficas combinadas
python scripts/analyze_results.py --input results/benchmark_results.csv --plots combined
```

## Uso Programático

```python
from modules import ParallelSUMOSimulator, ParallelEmissionProcessor

# Crear simulador
simulator = ParallelSUMOSimulator(
    num_processes=8,
    emission_batch_size=100,
    use_async_processing=True
)

# Cargar red
simulator.load_network("networks/rotterdam/rotterdam.net.xml")

# Configurar comando SUMO
sumo_cmd = [
    "sumo",
    "-n", "networks/rotterdam/rotterdam.net.xml",
    "-r", "networks/rotterdam/rotterdam_high.rou.xml",
    "--step-length", "1.0"
]

# Ejecutar simulación
result = simulator.run_simulation(
    sumo_cmd=sumo_cmd,
    end_time=3600,
    enable_rerouting=True,
    accident_edges=["edge_123"],  # Opcional
    collect_emissions=True
)

# Ver resultados
print(f"Speedup: {result.speedup:.2f}x")
print(f"Total CO2: {result.total_emissions['co2']/1000:.2f} kg")
```

## Estructura del Proyecto

```
parallel_sumo_simulation/
├── config/
│   └── settings.py         # Configuración global
├── modules/
│   ├── __init__.py
│   ├── emissions.py        # Cálculo paralelo de emisiones
│   ├── routing.py          # Cálculo paralelo de rutas (básico)
│   ├── sumo_routing.py     # Routing mejorado con sumolib
│   ├── simulation.py       # Simulador principal
│   └── data_collector.py   # Recolector de datos TraCI
├── scripts/
│   ├── run_simulation.py   # Script principal
│   ├── benchmark.py        # Benchmarking de speedup
│   ├── analyze_results.py  # Análisis y gráficas
│   └── generate_demand.py  # Generador de demanda
├── networks/               # Redes SUMO
├── results/                # Resultados y gráficas
├── requirements.txt
└── README.md
```

## Formato de Resultados

Los resultados de benchmark se guardan en CSV/Excel con columnas:

| Columna | Descripción |
|---------|-------------|
| scenario | Red (Almenara, Rotterdam) |
| machine | Identificador de máquina |
| processes | Número de procesos paralelos |
| traffic | Nivel de tráfico (Low/Medium/High) |
| accidents | Número de accidentes |
| speedup | Aceleración vs. baseline |
| total_time | Tiempo total de ejecución |
| emission_time | Tiempo en cálculo de emisiones |
| routing_time | Tiempo en cálculo de rutas |

## Optimización del Speedup

Para maximizar el speedup:

1. **Batch size adecuado**: 50-200 estados por lote
2. **Actualización no muy frecuente**: Rutas cada 30-60 segundos
3. **Procesamiento asíncrono**: Usar `use_async_processing=True`
4. **Escalar con tráfico**: Más vehículos = más trabajo paralelizable

## Limitaciones Conocidas

1. **TraCI es secuencial**: El bucle de simulación SUMO no se puede paralelizar
2. **Overhead de comunicación**: Con pocos vehículos, el overhead supera el beneficio
3. **Memoria**: Cada proceso necesita copia del grafo de red
4. **DUAROUTER**: Llamadas externas son lentas (usar solo para validación)

## Referencias

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [TraCI API](https://sumo.dlr.de/docs/TraCI.html)
- [HBEFA Emission Model](https://www.hbefa.net/)


