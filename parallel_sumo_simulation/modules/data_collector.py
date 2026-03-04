"""
Data Collection Module for SUMO Simulation

Collects vehicle states from TraCI during simulation and prepares
batches for parallel processing of emissions and routes.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import numpy as np


@dataclass
class SimulationStatistics:
    """Statistics collected during simulation"""
    total_vehicles: int = 0
    completed_trips: int = 0
    total_distance: float = 0.0  # meters
    total_travel_time: float = 0.0  # seconds
    total_waiting_time: float = 0.0  # seconds
    total_time_loss: float = 0.0  # seconds
    average_speed: float = 0.0  # m/s
    
    # Timing statistics
    simulation_time: float = 0.0
    emission_calculation_time: float = 0.0
    routing_calculation_time: float = 0.0
    data_collection_time: float = 0.0
    
    # Per-step statistics
    vehicles_per_step: List[int] = field(default_factory=list)
    

class DataCollector:
    """
    Collects and manages vehicle state data from TraCI.
    
    This class serves as the bridge between TraCI and the parallel
    processing modules for emissions and routing.
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        collection_interval: int = 1  # collect every N steps
    ):
        """
        Initialize data collector.
        
        Args:
            batch_size: Target batch size for parallel processing
            collection_interval: How often to collect full data
        """
        self.batch_size = batch_size
        self.collection_interval = collection_interval
        
        # Data storage
        self.vehicle_states: List[dict] = []
        self.vehicle_history: Dict[str, List[dict]] = defaultdict(list)
        
        # Route tracking
        self.vehicle_routes: Dict[str, List[str]] = {}
        self.vehicle_destinations: Dict[str, str] = {}
        
        # Edge traffic data
        self.edge_vehicle_counts: Dict[str, int] = defaultdict(int)
        self.edge_travel_times: Dict[str, List[float]] = defaultdict(list)
        
        # Statistics
        self.stats = SimulationStatistics()
        self.step_count = 0
        
        # Previous state for acceleration calculation
        self._prev_speeds: Dict[str, float] = {}
        self._prev_positions: Dict[str, Tuple[float, float]] = {}
        
    def collect_step(
        self, 
        traci_connection,
        current_time: float
    ) -> Tuple[List[dict], List[str]]:
        """
        Collect data for the current simulation step.
        
        Args:
            traci_connection: Active TraCI connection
            current_time: Current simulation time
            
        Returns:
            Tuple of (vehicle states for this step, departed vehicle IDs)
        """
        collection_start = time.time()
        
        step_states = []
        departed_vehicles = []
        
        # Get all vehicles in simulation
        vehicle_ids = traci_connection.vehicle.getIDList()
        self.stats.vehicles_per_step.append(len(vehicle_ids))
        
        # Get departed and arrived vehicles
        departed = traci_connection.simulation.getDepartedIDList()
        arrived = traci_connection.simulation.getArrivedIDList()
        
        departed_vehicles = list(departed)
        
        # Update statistics
        self.stats.total_vehicles = max(self.stats.total_vehicles, len(vehicle_ids))
        self.stats.completed_trips += len(arrived)
        
        # Collect state for each vehicle
        for veh_id in vehicle_ids:
            try:
                # Get current state from TraCI
                speed = traci_connection.vehicle.getSpeed(veh_id)
                position = traci_connection.vehicle.getPosition(veh_id)
                edge_id = traci_connection.vehicle.getRoadID(veh_id)
                waiting_time = traci_connection.vehicle.getWaitingTime(veh_id)
                
                # Calculate acceleration
                prev_speed = self._prev_speeds.get(veh_id, speed)
                acceleration = (speed - prev_speed) / 1.0  # assuming 1s step
                
                # Calculate distance traveled this step
                prev_pos = self._prev_positions.get(veh_id, position)
                distance = np.sqrt(
                    (position[0] - prev_pos[0]) ** 2 + 
                    (position[1] - prev_pos[1]) ** 2
                )
                
                # Create state dict
                state = {
                    "vehicle_id": veh_id,
                    "time_step": current_time,
                    "speed": speed,
                    "acceleration": acceleration,
                    "position": list(position),
                    "edge_id": edge_id,
                    "distance": distance,
                    "waiting_time": waiting_time
                }
                
                step_states.append(state)
                
                # Update history
                self.vehicle_history[veh_id].append(state)
                
                # Update edge counts
                self.edge_vehicle_counts[edge_id] += 1
                
                # Update previous state
                self._prev_speeds[veh_id] = speed
                self._prev_positions[veh_id] = position
                
                # Update aggregate statistics
                self.stats.total_distance += distance
                self.stats.total_waiting_time += waiting_time
                
            except Exception as e:
                # Vehicle may have left simulation
                continue
        
        # Clean up state for arrived vehicles
        for veh_id in arrived:
            self._prev_speeds.pop(veh_id, None)
            self._prev_positions.pop(veh_id, None)
        
        # Store states
        self.vehicle_states.extend(step_states)
        self.step_count += 1
        
        self.stats.data_collection_time += time.time() - collection_start
        
        return step_states, departed_vehicles
    
    def get_batch_for_emissions(self, clear: bool = True) -> List[dict]:
        """
        Get collected vehicle states for emission calculation.
        
        Args:
            clear: Whether to clear the buffer after retrieval
            
        Returns:
            List of vehicle state dictionaries
        """
        states = self.vehicle_states.copy()
        if clear:
            self.vehicle_states = []
        return states
    
    def get_rerouting_candidates(
        self,
        congestion_threshold: float = 0.5  # vehicles per meter threshold
    ) -> Tuple[Dict[str, str], Dict[str, str], set]:
        """
        Get data needed for rerouting decisions.
        
        Args:
            congestion_threshold: Threshold for considering an edge congested
            
        Returns:
            Tuple of (vehicle positions, vehicle destinations, congested edges)
        """
        # Calculate congested edges
        congested_edges = set()
        for edge_id, count in self.edge_vehicle_counts.items():
            # Simplified congestion detection
            if count > 10:  # More than 10 vehicles counted on this edge
                congested_edges.add(edge_id)
        
        # Get current vehicle positions (latest state)
        vehicle_positions = {}
        for veh_id, history in self.vehicle_history.items():
            if history:
                vehicle_positions[veh_id] = history[-1]["edge_id"]
        
        return vehicle_positions, self.vehicle_destinations, congested_edges
    
    def update_route(self, vehicle_id: str, route: List[str], destination: str = None):
        """Update stored route for a vehicle"""
        self.vehicle_routes[vehicle_id] = route
        if destination:
            self.vehicle_destinations[vehicle_id] = destination
    
    def get_edge_travel_times(self) -> Dict[str, float]:
        """
        Calculate current travel times for edges based on collected data.
        
        Returns:
            Dictionary of edge_id -> average travel time
        """
        travel_times = {}
        
        for edge_id, times in self.edge_travel_times.items():
            if times:
                travel_times[edge_id] = np.mean(times)
        
        return travel_times
    
    def record_edge_traversal(self, edge_id: str, travel_time: float):
        """Record a vehicle's traversal time for an edge"""
        self.edge_travel_times[edge_id].append(travel_time)
    
    def get_statistics(self) -> SimulationStatistics:
        """Get current simulation statistics"""
        # Calculate averages
        if self.stats.vehicles_per_step:
            avg_vehicles = np.mean(self.stats.vehicles_per_step)
            if self.stats.total_distance > 0 and sum(self.stats.vehicles_per_step) > 0:
                self.stats.average_speed = self.stats.total_distance / (
                    len(self.stats.vehicles_per_step) * avg_vehicles
                )
        
        return self.stats
    
    def reset(self):
        """Reset all collected data"""
        self.vehicle_states = []
        self.vehicle_history.clear()
        self.vehicle_routes.clear()
        self.vehicle_destinations.clear()
        self.edge_vehicle_counts.clear()
        self.edge_travel_times.clear()
        self.stats = SimulationStatistics()
        self.step_count = 0
        self._prev_speeds.clear()
        self._prev_positions.clear()


class BatchManager:
    """
    Manages batching of vehicle states for parallel processing.
    """
    
    def __init__(
        self,
        emission_batch_size: int = 100,
        routing_batch_size: int = 50,
        flush_interval: int = 10  # steps
    ):
        """
        Initialize batch manager.
        
        Args:
            emission_batch_size: Batch size for emission calculations
            routing_batch_size: Batch size for routing calculations
            flush_interval: Number of steps before forcing batch flush
        """
        self.emission_batch_size = emission_batch_size
        self.routing_batch_size = routing_batch_size
        self.flush_interval = flush_interval
        
        self._emission_buffer: List[dict] = []
        self._routing_buffer: List[dict] = []
        self._steps_since_flush = 0
        
    def add_states(self, states: List[dict]):
        """Add vehicle states to emission buffer"""
        self._emission_buffer.extend(states)
        self._steps_since_flush += 1
        
    def add_route_requests(self, requests: List[dict]):
        """Add route requests to routing buffer"""
        self._routing_buffer.extend(requests)
    
    def should_process_emissions(self) -> bool:
        """Check if emission batch should be processed"""
        return (
            len(self._emission_buffer) >= self.emission_batch_size or
            self._steps_since_flush >= self.flush_interval
        )
    
    def should_process_routes(self) -> bool:
        """Check if routing batch should be processed"""
        return len(self._routing_buffer) >= self.routing_batch_size
    
    def get_emission_batch(self) -> List[dict]:
        """Get and clear emission buffer"""
        batch = self._emission_buffer
        self._emission_buffer = []
        self._steps_since_flush = 0
        return batch
    
    def get_routing_batch(self) -> List[dict]:
        """Get and clear routing buffer"""
        batch = self._routing_buffer
        self._routing_buffer = []
        return batch
    
    def flush_all(self) -> Tuple[List[dict], List[dict]]:
        """Flush all buffers regardless of size"""
        emission_batch = self.get_emission_batch()
        routing_batch = self.get_routing_batch()
        return emission_batch, routing_batch


