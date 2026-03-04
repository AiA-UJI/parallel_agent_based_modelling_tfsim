"""
Configuration settings for Parallel SUMO Simulation
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# =====================================================
# PATH CONFIGURATION
# =====================================================

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# SUMO configuration
SUMO_HOME = os.environ.get("SUMO_HOME", "/usr/share/sumo")
SUMO_BINARY = os.path.join(SUMO_HOME, "bin", "sumo")
SUMO_GUI_BINARY = os.path.join(SUMO_HOME, "bin", "sumo-gui")

# Network files directory
NETWORKS_DIR = PROJECT_ROOT / "networks"
RESULTS_DIR = PROJECT_ROOT / "results"


# =====================================================
# SIMULATION PARAMETERS
# =====================================================

@dataclass
class SimulationConfig:
    """Configuration for a single simulation run"""
    
    # Network identification
    network_name: str = "test_network"
    scenario: str = "Almenara"  # or "Rotterdam"
    
    # Simulation time parameters (in seconds)
    begin_time: int = 0
    end_time: int = 3600  # 1 hour
    step_length: float = 1.0  # simulation step in seconds
    
    # Traffic demand parameters
    traffic_level: str = "Medium"  # Low, Medium, High
    vehicles_per_hour: int = 1000
    
    # Accident configuration
    num_accidents: int = 0
    accident_edges: List[str] = field(default_factory=list)
    accident_duration: int = 600  # seconds
    
    # Parallel processing
    num_processes: int = 4
    batch_size: int = 100  # vehicles per batch for parallel emission calculation
    
    # Output configuration
    output_emissions: bool = True
    output_routes: bool = True
    output_statistics: bool = True
    
    # SUMO specific
    seed: int = 42
    use_gui: bool = False
    
    def get_sumo_cmd(self, port: int = 8813) -> List[str]:
        """Generate SUMO command with all options"""
        binary = SUMO_GUI_BINARY if self.use_gui else SUMO_BINARY
        
        cmd = [
            binary,
            "-c", str(NETWORKS_DIR / self.network_name / f"{self.network_name}.sumocfg"),
            "--begin", str(self.begin_time),
            "--end", str(self.end_time),
            "--step-length", str(self.step_length),
            "--seed", str(self.seed),
            "--remote-port", str(port),
            "--no-warnings", "true",
            "--no-step-log", "true",
        ]
        
        return cmd


# =====================================================
# TRAFFIC LEVEL CONFIGURATIONS
# =====================================================

TRAFFIC_CONFIGS = {
    "Low": {
        "vehicles_per_hour": 500,
        "description": "Light traffic conditions"
    },
    "Medium": {
        "vehicles_per_hour": 1500,
        "description": "Normal traffic conditions"
    },
    "High": {
        "vehicles_per_hour": 3000,
        "description": "Heavy traffic / rush hour"
    }
}


# =====================================================
# MACHINE CONFIGURATIONS (for benchmarking)
# =====================================================

MACHINE_CONFIGS = {
    "Machine A": {
        "max_processes": 8,
        "description": "8-core desktop CPU"
    },
    "Machine B": {
        "max_processes": 16,
        "description": "16-core workstation"
    },
    "Machine C": {
        "max_processes": 6,
        "description": "6-core laptop"
    },
    "HPC Node": {
        "max_processes": 32,
        "description": "HPC cluster node"
    }
}


# =====================================================
# EMISSION CALCULATION PARAMETERS
# =====================================================

@dataclass
class EmissionConfig:
    """Configuration for emission calculations"""
    
    # HBEFA emission classes
    emission_class: str = "HBEFA3/PC_G_EU4"
    
    # Pollutants to calculate
    pollutants: List[str] = field(default_factory=lambda: [
        "CO2", "CO", "HC", "NOx", "PMx", "fuel"
    ])
    
    # Calculation intervals
    aggregation_interval: int = 60  # seconds
    
    # Speed thresholds for emission factors
    idle_speed_threshold: float = 0.1  # m/s
    

# =====================================================
# ROUTE CALCULATION PARAMETERS
# =====================================================

@dataclass
class RouteConfig:
    """Configuration for route/itinerary calculations"""
    
    # Routing algorithm
    algorithm: str = "dijkstra"  # dijkstra, astar, ch
    
    # Route optimization criteria
    criteria: str = "time"  # time, distance, emissions
    
    # Dynamic rerouting
    rerouting_enabled: bool = True
    rerouting_period: int = 60  # seconds
    rerouting_probability: float = 0.1
    
    # Parallelization
    parallel_routing: bool = True


# =====================================================
# BENCHMARK CONFIGURATION
# =====================================================

@dataclass
class BenchmarkConfig:
    """Configuration for speedup benchmarking"""
    
    # Number of repetitions for averaging
    num_repetitions: int = 3
    
    # Process counts to test
    process_counts: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    
    # Scenarios to test
    scenarios: List[str] = field(default_factory=lambda: ["Almenara", "Rotterdam"])
    
    # Traffic levels
    traffic_levels: List[str] = field(default_factory=lambda: ["Low", "Medium", "High"])
    
    # Accident configurations
    accident_counts: List[int] = field(default_factory=lambda: [0, 1])
    
    # Warmup runs (discarded)
    warmup_runs: int = 1
    
    # Output format
    output_format: str = "xlsx"  # csv, xlsx, json


