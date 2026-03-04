"""
Parallel Emission Calculation Module

This module handles the calculation of vehicle emissions outside the main
SUMO simulation loop, enabling parallel processing of emission computations.

The emission model is based on HBEFA (Handbook Emission Factors for Road Transport)
and calculates CO2, CO, HC, NOx, PMx, and fuel consumption.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time


# =====================================================
# EMISSION FACTORS (HBEFA-based simplified model)
# =====================================================

# Emission factors per pollutant (g/km at different speed ranges)
# Based on HBEFA3/PC_G_EU4 (Gasoline Passenger Car Euro 4)
EMISSION_FACTORS = {
    "CO2": {
        "idle": 2.56,      # g/s at idle
        "low": 180.0,      # g/km at < 30 km/h
        "medium": 150.0,   # g/km at 30-80 km/h
        "high": 170.0,     # g/km at > 80 km/h
    },
    "CO": {
        "idle": 0.008,
        "low": 1.2,
        "medium": 0.5,
        "high": 0.8,
    },
    "HC": {
        "idle": 0.001,
        "low": 0.08,
        "medium": 0.03,
        "high": 0.05,
    },
    "NOx": {
        "idle": 0.0005,
        "low": 0.15,
        "medium": 0.12,
        "high": 0.25,
    },
    "PMx": {
        "idle": 0.00001,
        "low": 0.005,
        "medium": 0.003,
        "high": 0.004,
    },
    "fuel": {
        "idle": 0.0008,    # L/s at idle
        "low": 0.075,      # L/km at low speed
        "medium": 0.062,   # L/km at medium speed
        "high": 0.070,     # L/km at high speed
    }
}


@dataclass
class VehicleState:
    """State of a vehicle at a specific time step"""
    vehicle_id: str
    time_step: float
    speed: float          # m/s
    acceleration: float   # m/s²
    position: Tuple[float, float]  # x, y coordinates
    edge_id: str
    distance: float       # distance traveled this step (m)
    waiting_time: float   # time waiting (s)
    
    
@dataclass
class EmissionResult:
    """Emission calculation result for a vehicle"""
    vehicle_id: str
    time_step: float
    co2: float      # grams
    co: float       # grams
    hc: float       # grams
    nox: float      # grams
    pmx: float      # grams
    fuel: float     # liters


class EmissionCalculator:
    """
    Calculator for vehicle emissions based on instantaneous vehicle states.
    
    This class can operate independently of TraCI, receiving vehicle state
    data and computing emissions in a thread-safe manner.
    """
    
    def __init__(self, emission_class: str = "HBEFA3/PC_G_EU4"):
        """
        Initialize emission calculator.
        
        Args:
            emission_class: HBEFA emission class identifier
        """
        self.emission_class = emission_class
        self.emission_factors = EMISSION_FACTORS
        
        # Speed thresholds (m/s)
        self.idle_threshold = 0.1
        self.low_speed_threshold = 8.33    # 30 km/h
        self.high_speed_threshold = 22.22  # 80 km/h
        
    def get_speed_category(self, speed: float) -> str:
        """Determine speed category for emission factors"""
        if speed < self.idle_threshold:
            return "idle"
        elif speed < self.low_speed_threshold:
            return "low"
        elif speed < self.high_speed_threshold:
            return "medium"
        else:
            return "high"
    
    def calculate_acceleration_factor(self, acceleration: float) -> float:
        """
        Calculate emission multiplier based on acceleration.
        
        Higher acceleration = more emissions (engine load factor)
        """
        if acceleration <= 0:
            return 1.0
        elif acceleration < 1.0:
            return 1.0 + 0.1 * acceleration
        elif acceleration < 2.0:
            return 1.1 + 0.2 * (acceleration - 1.0)
        else:
            return 1.3 + 0.3 * (acceleration - 2.0)
    
    def calculate_emissions(self, state: VehicleState) -> EmissionResult:
        """
        Calculate emissions for a single vehicle state.
        
        Args:
            state: Vehicle state at current time step
            
        Returns:
            EmissionResult with all pollutant values
        """
        speed_category = self.get_speed_category(state.speed)
        accel_factor = self.calculate_acceleration_factor(state.acceleration)
        
        # Distance in km for this step
        distance_km = state.distance / 1000.0
        
        emissions = {}
        
        for pollutant, factors in self.emission_factors.items():
            factor = factors[speed_category]
            
            if speed_category == "idle":
                # Idle emissions are per second
                # Assume 1 second step if waiting
                idle_time = max(1.0, state.waiting_time) if state.speed < self.idle_threshold else 1.0
                emissions[pollutant] = factor * idle_time * accel_factor
            else:
                # Distance-based emissions (g/km or L/km)
                if distance_km > 0:
                    emissions[pollutant] = factor * distance_km * accel_factor
                else:
                    emissions[pollutant] = 0.0
        
        return EmissionResult(
            vehicle_id=state.vehicle_id,
            time_step=state.time_step,
            co2=emissions["CO2"],
            co=emissions["CO"],
            hc=emissions["HC"],
            nox=emissions["NOx"],
            pmx=emissions["PMx"],
            fuel=emissions["fuel"]
        )
    
    def calculate_batch(self, states: List[VehicleState]) -> List[EmissionResult]:
        """
        Calculate emissions for a batch of vehicle states.
        
        Args:
            states: List of vehicle states
            
        Returns:
            List of emission results
        """
        return [self.calculate_emissions(state) for state in states]


def _calculate_emissions_worker(args: Tuple[List[dict], str]) -> List[dict]:
    """
    Worker function for parallel emission calculation.
    
    Args:
        args: Tuple of (batch of vehicle state dicts, emission class)
        
    Returns:
        List of emission result dicts
    """
    states_data, emission_class = args
    calculator = EmissionCalculator(emission_class)
    
    # Convert dicts to VehicleState objects
    states = [
        VehicleState(
            vehicle_id=s["vehicle_id"],
            time_step=s["time_step"],
            speed=s["speed"],
            acceleration=s["acceleration"],
            position=tuple(s["position"]),
            edge_id=s["edge_id"],
            distance=s["distance"],
            waiting_time=s["waiting_time"]
        )
        for s in states_data
    ]
    
    results = calculator.calculate_batch(states)
    
    # Convert back to dicts for serialization
    return [
        {
            "vehicle_id": r.vehicle_id,
            "time_step": r.time_step,
            "co2": r.co2,
            "co": r.co,
            "hc": r.hc,
            "nox": r.nox,
            "pmx": r.pmx,
            "fuel": r.fuel
        }
        for r in results
    ]


class ParallelEmissionProcessor:
    """
    Parallel processor for batch emission calculations.
    
    This class manages the parallel computation of emissions for large
    numbers of vehicle states, distributing work across multiple processes.
    """
    
    def __init__(
        self, 
        num_processes: Optional[int] = None,
        batch_size: int = 100,
        emission_class: str = "HBEFA3/PC_G_EU4"
    ):
        """
        Initialize parallel emission processor.
        
        Args:
            num_processes: Number of worker processes (None = CPU count)
            batch_size: Number of vehicle states per batch
            emission_class: HBEFA emission class
        """
        self.num_processes = num_processes or cpu_count()
        self.batch_size = batch_size
        self.emission_class = emission_class
        
        # Performance metrics
        self.total_processed = 0
        self.total_time = 0.0
        
    def _create_batches(self, states: List[dict]) -> List[List[dict]]:
        """Split states into batches for parallel processing"""
        batches = []
        for i in range(0, len(states), self.batch_size):
            batches.append(states[i:i + self.batch_size])
        return batches
    
    def process_emissions(self, vehicle_states: List[dict]) -> List[dict]:
        """
        Process emissions for all vehicle states in parallel.
        
        Args:
            vehicle_states: List of vehicle state dictionaries with keys:
                - vehicle_id, time_step, speed, acceleration, position,
                  edge_id, distance, waiting_time
                  
        Returns:
            List of emission result dictionaries
        """
        if not vehicle_states:
            return []
        
        start_time = time.time()
        
        # Create batches
        batches = self._create_batches(vehicle_states)
        
        # Prepare arguments for workers
        worker_args = [(batch, self.emission_class) for batch in batches]
        
        all_results = []
        
        if self.num_processes == 1:
            # Sequential processing for baseline
            for args in worker_args:
                results = _calculate_emissions_worker(args)
                all_results.extend(results)
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                futures = [
                    executor.submit(_calculate_emissions_worker, args)
                    for args in worker_args
                ]
                
                for future in as_completed(futures):
                    results = future.result()
                    all_results.extend(results)
        
        elapsed = time.time() - start_time
        
        # Update metrics
        self.total_processed += len(vehicle_states)
        self.total_time += elapsed
        
        return all_results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            "total_processed": self.total_processed,
            "total_time": self.total_time,
            "throughput": self.total_processed / self.total_time if self.total_time > 0 else 0,
            "num_processes": self.num_processes
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.total_processed = 0
        self.total_time = 0.0


class AggregatedEmissions:
    """
    Aggregator for emission results over time and space.
    """
    
    def __init__(self, aggregation_interval: int = 60):
        """
        Initialize aggregator.
        
        Args:
            aggregation_interval: Time interval for aggregation (seconds)
        """
        self.aggregation_interval = aggregation_interval
        
        # Storage for aggregated data
        self.by_vehicle: Dict[str, Dict[str, float]] = {}
        self.by_edge: Dict[str, Dict[str, float]] = {}
        self.by_time: Dict[int, Dict[str, float]] = {}
        self.total: Dict[str, float] = {
            "co2": 0.0, "co": 0.0, "hc": 0.0, 
            "nox": 0.0, "pmx": 0.0, "fuel": 0.0
        }
        
    def add_results(
        self, 
        results: List[dict], 
        edge_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Add emission results to aggregation.
        
        Args:
            results: List of emission result dicts
            edge_mapping: Optional mapping of vehicle_id to edge_id
        """
        for r in results:
            vehicle_id = r["vehicle_id"]
            time_bucket = int(r["time_step"] // self.aggregation_interval)
            
            pollutants = ["co2", "co", "hc", "nox", "pmx", "fuel"]
            
            # By vehicle
            if vehicle_id not in self.by_vehicle:
                self.by_vehicle[vehicle_id] = {p: 0.0 for p in pollutants}
            
            # By time bucket
            if time_bucket not in self.by_time:
                self.by_time[time_bucket] = {p: 0.0 for p in pollutants}
            
            for p in pollutants:
                value = r[p]
                self.by_vehicle[vehicle_id][p] += value
                self.by_time[time_bucket][p] += value
                self.total[p] += value
                
                # By edge if mapping provided
                if edge_mapping and vehicle_id in edge_mapping:
                    edge_id = edge_mapping[vehicle_id]
                    if edge_id not in self.by_edge:
                        self.by_edge[edge_id] = {p: 0.0 for p in pollutants}
                    self.by_edge[edge_id][p] += value
    
    def get_summary(self) -> Dict:
        """Get summary of all aggregated emissions"""
        return {
            "total": self.total,
            "num_vehicles": len(self.by_vehicle),
            "num_edges": len(self.by_edge),
            "num_time_buckets": len(self.by_time),
            "aggregation_interval": self.aggregation_interval
        }


