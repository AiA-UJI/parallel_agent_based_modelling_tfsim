# Parallel SUMO Simulation Framework

Framework for running SUMO traffic simulations with parallel **emissions** and **routing** computation, enabling significant speedup on multicore systems.

## Architecture

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

## Main components

### 1. Emissions module (`modules/emissions.py`)
- **EmissionCalculator**: Computes per-vehicle emissions using the HBEFA model
- **ParallelEmissionProcessor**: Processes batches of vehicle states in parallel
- Pollutants: CO2, CO, HC, NOx, PMx, fuel consumption

### 2. Routing module (`modules/routing.py` and `modules/sumo_routing.py`)
- **RouteCalculator**: Dijkstra and A* algorithms
- **SUMORouter**: Uses full data from SUMO `.net.xml`
- **ParallelRouteProcessor**: Parallel route computation
- **DynamicRerouter**: Dynamic rerouting based on congestion

### 3. Main simulator (`modules/simulation.py`)
- **ParallelSUMOSimulator**: Integrates TraCI with parallel processing
- Asynchronous processing support
- Accident and event handling

## Routing with `.net.xml`

### What data is used?

The SUMO `.net.xml` file contains:

| Data | Available | Use in routing |
|------|-----------|----------------|
| Topology (edges, nodes) | Yes | Network graph |
| Edge length | Yes | Distance cost |
| Speed limit | Yes | Free-flow time |
| Connections (turns) | Yes | Path constraints |
| Traffic lights | Yes | (Programs only, not state) |
| Current traffic | No | Requires TraCI |

### Routing strategy

1. **Initial**: Free-flow times (`time = length / speed_limit`)
2. **During simulation**: Update from TraCI with `edge.getTraveltime()`
3. **Temporal smoothing**: Weighted average of last N values
|

### Routing options

```python
# Option 1: Internal router (fast, parallel)
router = SUMORouter(network_parser)
route, cost = router.find_route_astar(from_edge, to_edge)

# Option 2: SUMO DUAROUTER (more accurate, slower)
route, cost = router.find_route_duarouter(from_edge, to_edge)

# Option 3: Parallel routing with TraCI update
parallel_router = ParallelSUMORouter(network_parser, num_processes=4)
updater = TraCIRouteUpdater(network_parser)

# During simulation:
updater.update_from_traci(traci_connection, current_time)
routes = parallel_router.calculate_batch(od_pairs)
```

## Installation

### Requirements
- Python 3.9+
- SUMO 1.14+ with TraCI
- Environment variable `SUMO_HOME`

### Install

```bash
cd parallel_sumo_simulation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Create test network

```bash
# Create network definition files
python scripts/run_simulation.py --create-network test_grid

# Build network with netconvert (SUMO)
cd networks/test_grid
netconvert -n test_grid.nod.xml -e test_grid.edg.xml -o test_grid.net.xml
```

### 2. Generate traffic demand

```bash
python scripts/generate_demand.py --network test_grid --level all --time 3600
```

### 3. Run simulation

```bash
# Basic simulation
python scripts/run_simulation.py --network test_grid --processes 4 --traffic medium

# With accidents
python scripts/run_simulation.py --network test_grid --processes 8 --accidents 1

# With more options
python scripts/run_simulation.py \
    --network test_grid \
    --processes 8 \
    --traffic high \
    --time 3600 \
    --accidents 1
```

### 4. Run benchmark

```bash
# Simulated benchmark (no real SUMO)
python scripts/benchmark.py --mode simulated

# Quick benchmark
python scripts/benchmark.py --mode quick --machine "My Machine"

# Full benchmark
python scripts/benchmark.py --mode full --machine "HPC Node" \
    --processes 1 2 4 8 16 32 \
    --repetitions 3
```

### 5. Analyze results

```bash
# Generate sample data and plots
python scripts/analyze_results.py --create-sample

# Analyze benchmark results
python scripts/analyze_results.py --input results/benchmark_results.csv

# Combined plots only
python scripts/analyze_results.py --input results/benchmark_results.csv --plots combined
```

## Programmatic usage

```python
from modules import ParallelSUMOSimulator, ParallelEmissionProcessor

# Create simulator
simulator = ParallelSUMOSimulator(
    num_processes=8,
    emission_batch_size=100,
    use_async_processing=True
)

# Load network
simulator.load_network("networks/rotterdam/rotterdam.net.xml")

# Configure SUMO command
sumo_cmd = [
    "sumo",
    "-n", "networks/rotterdam/rotterdam.net.xml",
    "-r", "networks/rotterdam/rotterdam_high.rou.xml",
    "--step-length", "1.0"
]

# Run simulation
result = simulator.run_simulation(
    sumo_cmd=sumo_cmd,
    end_time=3600,
    enable_rerouting=True,
    accident_edges=["edge_123"],  # Optional
    collect_emissions=True
)

# View results
print(f"Speedup: {result.speedup:.2f}x")
print(f"Total CO2: {result.total_emissions['co2']/1000:.2f} kg")
```

## Project structure

```
parallel_sumo_simulation/
├── config/
│   └── settings.py         # Global configuration
├── modules/
│   ├── __init__.py
│   ├── emissions.py        # Parallel emissions computation
│   ├── routing.py          # Parallel routing (basic)
│   ├── sumo_routing.py     # Improved routing with sumolib
│   ├── simulation.py       # Main simulator
│   └── data_collector.py   # TraCI data collector
├── scripts/
│   ├── run_simulation.py   # Main script
│   ├── benchmark.py        # Speedup benchmarking
│   ├── analyze_results.py  # Analysis and plots
│   └── generate_demand.py  # Demand generator
├── networks/               # SUMO networks
├── results/                 # Results and plots
├── requirements.txt
└── README.md
```

## Result format

Benchmark results are saved in CSV/Excel with columns:

| Column | Description |
|--------|-------------|
| scenario | Network (Almenara, Rotterdam) |
| machine | Machine identifier |
| processes | Number of parallel processes |
| traffic | Traffic level (Low/Medium/High) |
| accidents | Number of accidents |
| speedup | Speedup vs. baseline |
| total_time | Total execution time |
| emission_time | Time in emissions computation |
| routing_time | Time in routing computation |

## Speedup optimization

To maximize speedup:

1. **Suitable batch size**: 50–200 states per batch
2. **Not too frequent updates**: Routes every 30–60 seconds
3. **Asynchronous processing**: Use `use_async_processing=True`
4. **Scale with traffic**: More vehicles means more parallelizable work

## References

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [TraCI API](https://sumo.dlr.de/docs/TraCI.html)
- [HBEFA Emission Model](https://www.hbefa.net/)
