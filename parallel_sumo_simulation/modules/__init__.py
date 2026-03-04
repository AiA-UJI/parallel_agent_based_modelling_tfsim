"""
Parallel SUMO Simulation Modules
"""

from .emissions import EmissionCalculator, ParallelEmissionProcessor
from .routing import RouteCalculator, ParallelRouteProcessor
from .simulation import ParallelSUMOSimulator
from .data_collector import DataCollector

# Enhanced SUMO routing (uses sumolib for better network parsing)
try:
    from .sumo_routing import (
        SUMONetworkParser,
        SUMORouter,
        ParallelSUMORouter,
        TraCIRouteUpdater
    )
    SUMO_ROUTING_AVAILABLE = True
except ImportError:
    SUMO_ROUTING_AVAILABLE = False

__all__ = [
    "EmissionCalculator",
    "ParallelEmissionProcessor",
    "RouteCalculator", 
    "ParallelRouteProcessor",
    "ParallelSUMOSimulator",
    "DataCollector",
    # Enhanced SUMO routing
    "SUMONetworkParser",
    "SUMORouter",
    "ParallelSUMORouter",
    "TraCIRouteUpdater",
    "SUMO_ROUTING_AVAILABLE"
]

