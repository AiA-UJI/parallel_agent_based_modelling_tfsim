# parallel_agent_based_modelling_tfsim

Code for the paper **"Parallel Agent-Based Modeling for Improving Traffic Flow Simulation"** (V. R. Tomás, M. Castillo, I. Monzón Catalán, L. A. García — Universitat Jaume I).

## Summary

This work addresses task-level parallelization of SUMO (Simulation of Urban MObility) simulations to reduce runtime without modifying the SUMO engine. The framework distributes vehicle- and segment-level computation (emissions, dynamic routing, agent state) across multiple processes, achieving speedups of up to ~4× on real-world interurban corridors and highway networks.

## Repository contents

- **`hgv.py`** — Utility to load and process DATEX2 (XML/gzip) traffic/speed data.
- **`parallel_sumo_simulation/`** — Parallel simulation framework:
  - Modules: emissions (HBEFA), routing (Dijkstra/A*, SUMO), TraCI simulator with workers.
  - Scripts: run simulations, generate demand, run benchmarks, and analyze results.

Only Python code from this package and `hgv.py` is versioned; the paper, maps, and virtual environments are excluded (see `.gitignore`).

## Requirements

- Python 3.9+
- SUMO with TraCI and `SUMO_HOME` set.

## Quick start

From the repository root:

```bash
cd parallel_sumo_simulation
python scripts/run_simulation.py --network <network> --processes 4
python scripts/benchmark.py --mode quick
```

See `parallel_sumo_simulation/README.md` for detailed documentation when available in your clone.



# Additional Materials

Some of the files used in this article (such as maps, itineraries, and other supporting materials) are not included in this repository due to their large size.

If you need access to these resources, they can be requested by email:

* [vtomas@uji.es](mailto:vtomas@uji.es)
* [imonzon@uji.es](mailto:imonzon@uji.es)

The authors will provide the requested files upon demand for academic or research purposes.
