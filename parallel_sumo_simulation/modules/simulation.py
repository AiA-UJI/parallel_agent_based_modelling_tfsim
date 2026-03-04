"""
Parallel SUMO Simulation Module

Main simulation controller that integrates SUMO/TraCI with parallel
processing of emissions and routing calculations.

The key optimization strategy is to:
1. Run SUMO simulation step-by-step via TraCI
2. Collect vehicle states in batches
3. Process emissions and routing in parallel worker pools
4. Apply routing updates back to the simulation
"""

import os
import sys
import time
import socket
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import queue

# Add SUMO tools to path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)

try:
    import traci
    import sumolib
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    print("Warning: TraCI not available. Install SUMO and set SUMO_HOME environment variable.")

from .emissions import ParallelEmissionProcessor, AggregatedEmissions
from .routing import NetworkGraph, Edge, ParallelRouteProcessor, DynamicRerouter
from .data_collector import DataCollector, BatchManager, SimulationStatistics


@dataclass
class SimulationResult:
    """Results from a complete simulation run"""
    # Timing results
    total_time: float = 0.0
    simulation_time: float = 0.0
    emission_time: float = 0.0
    routing_time: float = 0.0
    overhead_time: float = 0.0
    
    # Speedup metrics
    baseline_time: Optional[float] = None
    speedup: float = 1.0
    efficiency: float = 1.0
    
    # Simulation statistics
    total_vehicles: int = 0
    completed_trips: int = 0
    total_steps: int = 0
    
    # Emission results
    total_emissions: Dict[str, float] = field(default_factory=dict)
    
    # Configuration
    num_processes: int = 1
    scenario: str = ""
    traffic_level: str = ""
    num_accidents: int = 0


class ParallelSUMOSimulator:
    """
    Main simulator class that coordinates SUMO simulation with
    parallel emission and routing calculations.
    """
    
    def __init__(
        self,
        num_processes: int = 4,
        emission_batch_size: int = 100,
        routing_batch_size: int = 50,
        use_async_processing: bool = True
    ):
        """
        Initialize parallel SUMO simulator.
        
        Args:
            num_processes: Number of worker processes for parallel tasks
            emission_batch_size: Batch size for emission calculations
            routing_batch_size: Batch size for routing calculations
            use_async_processing: Whether to process asynchronously
        """
        if not TRACI_AVAILABLE:
            raise RuntimeError("TraCI is not available. Please install SUMO.")
        
        self.num_processes = num_processes
        self.use_async = use_async_processing
        
        # Initialize components
        self.data_collector = DataCollector(batch_size=emission_batch_size)
        self.batch_manager = BatchManager(
            emission_batch_size=emission_batch_size,
            routing_batch_size=routing_batch_size
        )
        
        # Parallel processors (initialized when network is loaded)
        self.emission_processor: Optional[ParallelEmissionProcessor] = None
        self.route_processor: Optional[ParallelRouteProcessor] = None
        self.rerouter: Optional[DynamicRerouter] = None
        
        # Network data
        self.network: Optional[NetworkGraph] = None
        self.network_file: Optional[str] = None
        
        # Results storage
        self.emission_results: List[dict] = []
        self.aggregated_emissions = AggregatedEmissions()
        
        # Async processing queues
        if self.use_async:
            self._emission_queue = queue.Queue()
            self._result_queue = queue.Queue()
            self._processing_thread: Optional[threading.Thread] = None
            self._stop_processing = threading.Event()
        
        # TraCI connection
        self._traci_connection = None
        self._port = None
        
    def load_network(self, network_file: str):
        """
        Load SUMO network and prepare for simulation.
        
        Args:
            network_file: Path to SUMO .net.xml file
        """
        self.network_file = network_file
        self.network = NetworkGraph()
        
        # Parse network using sumolib
        if TRACI_AVAILABLE:
            try:
                net = sumolib.net.readNet(network_file)
                
                # Add edges to network graph
                for edge in net.getEdges():
                    edge_obj = Edge(
                        edge_id=edge.getID(),
                        from_node=edge.getFromNode().getID(),
                        to_node=edge.getToNode().getID(),
                        length=edge.getLength(),
                        speed_limit=edge.getSpeed(),
                        num_lanes=edge.getLaneNumber()
                    )
                    self.network.add_edge(edge_obj)
                
                print(f"Loaded network with {len(self.network.edges)} edges")
                
            except Exception as e:
                print(f"Warning: Could not parse network file: {e}")
                print("Network-dependent features will be limited.")
        
        # Initialize processors
        self.emission_processor = ParallelEmissionProcessor(
            num_processes=self.num_processes,
            batch_size=self.batch_manager.emission_batch_size
        )
        
        if self.network and len(self.network.edges) > 0:
            self.route_processor = ParallelRouteProcessor(
                network=self.network,
                num_processes=self.num_processes,
                batch_size=self.batch_manager.routing_batch_size
            )
            self.rerouter = DynamicRerouter(self.route_processor)
    
    def _find_free_port(self) -> int:
        """Find a free port for TraCI connection"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def _start_async_processing(self):
        """Start background thread for async emission processing"""
        if not self.use_async:
            return
            
        def process_worker():
            while not self._stop_processing.is_set():
                try:
                    batch = self._emission_queue.get(timeout=0.1)
                    if batch:
                        results = self.emission_processor.process_emissions(batch)
                        self._result_queue.put(results)
                except queue.Empty:
                    continue
        
        self._processing_thread = threading.Thread(target=process_worker, daemon=True)
        self._processing_thread.start()
    
    def _stop_async_processing(self):
        """Stop background processing thread"""
        if self.use_async and self._processing_thread:
            self._stop_processing.set()
            self._processing_thread.join(timeout=5.0)
            self._stop_processing.clear()
    
    def _collect_async_results(self):
        """Collect results from async processing"""
        if not self.use_async:
            return
            
        while not self._result_queue.empty():
            try:
                results = self._result_queue.get_nowait()
                self.emission_results.extend(results)
                self.aggregated_emissions.add_results(results)
            except queue.Empty:
                break
    
    def run_simulation(
        self,
        sumo_cmd: List[str],
        end_time: float = 3600,
        step_length: float = 1.0,
        enable_rerouting: bool = True,
        rerouting_interval: int = 60,
        accident_edges: Optional[List[str]] = None,
        accident_start_time: float = 300,
        accident_duration: float = 600,
        collect_emissions: bool = True
    ) -> SimulationResult:
        """
        Run the parallel SUMO simulation.
        
        Args:
            sumo_cmd: SUMO command with all options
            end_time: Simulation end time in seconds
            step_length: Simulation step length in seconds
            enable_rerouting: Enable dynamic rerouting
            rerouting_interval: Interval for rerouting checks
            accident_edges: List of edge IDs where accidents occur
            accident_start_time: When accidents start
            accident_duration: How long accidents last
            collect_emissions: Whether to collect and process emissions
            
        Returns:
            SimulationResult with all metrics
        """
        result = SimulationResult(num_processes=self.num_processes)
        
        # Find free port
        self._port = self._find_free_port()
        
        # Modify SUMO command with port
        cmd = sumo_cmd.copy()
        # Update or add port
        if "--remote-port" in cmd:
            idx = cmd.index("--remote-port")
            cmd[idx + 1] = str(self._port)
        else:
            cmd.extend(["--remote-port", str(self._port)])
        
        # Reset collectors
        self.data_collector.reset()
        self.emission_results = []
        self.aggregated_emissions = AggregatedEmissions()
        
        if self.emission_processor:
            self.emission_processor.reset_stats()
        if self.route_processor:
            self.route_processor.reset_stats()
        
        total_start = time.time()
        sim_time = 0.0
        emission_time = 0.0
        routing_time = 0.0
        
        try:
            # Start SUMO
            traci.start(cmd)
            self._traci_connection = traci
            
            # Start async processing if enabled
            if self.use_async and collect_emissions:
                self._start_async_processing()
            
            current_time = 0.0
            step = 0
            
            # Accident state
            accidents_active = False
            accident_vehicles = []
            
            while current_time < end_time:
                step_start = time.time()
                
                # === SIMULATION STEP ===
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                step += 1
                
                sim_time += time.time() - step_start
                
                # === ACCIDENT HANDLING ===
                if accident_edges:
                    # Start accident
                    if (current_time >= accident_start_time and 
                        current_time < accident_start_time + accident_duration and
                        not accidents_active):
                        accidents_active = True
                        for edge_id in accident_edges:
                            try:
                                # Add stopped vehicles as accidents
                                acc_id = f"accident_{edge_id}"
                                traci.vehicle.add(
                                    acc_id, 
                                    routeID="",
                                    typeID="DEFAULT_VEHTYPE",
                                    departLane="best",
                                    departPos="base"
                                )
                                # Try to set on the accident edge
                                traci.vehicle.moveToXY(
                                    acc_id, edge_id, 0, 
                                    *traci.simulation.convert2D(edge_id, 10), 
                                    keepRoute=2
                                )
                                traci.vehicle.setSpeed(acc_id, 0)
                                accident_vehicles.append(acc_id)
                            except Exception:
                                pass
                    
                    # End accident
                    if (current_time >= accident_start_time + accident_duration and 
                        accidents_active):
                        accidents_active = False
                        for acc_id in accident_vehicles:
                            try:
                                traci.vehicle.remove(acc_id)
                            except Exception:
                                pass
                        accident_vehicles = []
                
                # === DATA COLLECTION ===
                collect_start = time.time()
                step_states, departed = self.data_collector.collect_step(
                    traci, current_time
                )
                
                # Store routes for new vehicles
                for veh_id in departed:
                    try:
                        route = list(traci.vehicle.getRoute(veh_id))
                        if route:
                            destination = route[-1]
                            self.data_collector.update_route(veh_id, route, destination)
                    except Exception:
                        pass
                
                # === EMISSION PROCESSING ===
                if collect_emissions:
                    self.batch_manager.add_states(step_states)
                    
                    if self.batch_manager.should_process_emissions():
                        batch = self.batch_manager.get_emission_batch()
                        
                        emit_start = time.time()
                        
                        if self.use_async:
                            # Queue for async processing
                            self._emission_queue.put(batch)
                            self._collect_async_results()
                        else:
                            # Sync processing
                            if batch and self.emission_processor:
                                results = self.emission_processor.process_emissions(batch)
                                self.emission_results.extend(results)
                                self.aggregated_emissions.add_results(results)
                        
                        emission_time += time.time() - emit_start
                
                # === DYNAMIC REROUTING ===
                if (enable_rerouting and 
                    self.rerouter and 
                    step % rerouting_interval == 0):
                    
                    route_start = time.time()
                    
                    # Get rerouting candidates
                    positions, destinations, congested = self.data_collector.get_rerouting_candidates()
                    
                    if positions and destinations:
                        # Calculate new routes
                        new_routes = self.rerouter.check_rerouting(
                            current_time, positions, destinations, congested
                        )
                        
                        # Apply new routes
                        for veh_id, route in new_routes.items():
                            try:
                                if route:
                                    traci.vehicle.setRoute(veh_id, route)
                                    self.data_collector.update_route(veh_id, route)
                            except Exception:
                                pass
                    
                    routing_time += time.time() - route_start
                
                # Progress indicator (every 100 steps)
                if step % 100 == 0:
                    vehicles = len(traci.vehicle.getIDList())
                    print(f"\rStep {step}, Time: {current_time:.0f}s, Vehicles: {vehicles}", 
                          end="", flush=True)
            
            print()  # New line after progress
            
            # === FINAL PROCESSING ===
            # Process any remaining batches
            if collect_emissions:
                remaining_emit, remaining_route = self.batch_manager.flush_all()
                
                if remaining_emit and self.emission_processor:
                    emit_start = time.time()
                    results = self.emission_processor.process_emissions(remaining_emit)
                    self.emission_results.extend(results)
                    self.aggregated_emissions.add_results(results)
                    emission_time += time.time() - emit_start
                
                # Collect any remaining async results
                if self.use_async:
                    time.sleep(0.5)  # Wait for processing to complete
                    self._collect_async_results()
            
        except Exception as e:
            print(f"\nSimulation error: {e}")
            raise
            
        finally:
            # Stop async processing
            self._stop_async_processing()
            
            # Close TraCI
            try:
                traci.close()
            except Exception:
                pass
            
            self._traci_connection = None
        
        # === COMPILE RESULTS ===
        total_time = time.time() - total_start
        overhead_time = total_time - sim_time - emission_time - routing_time
        
        result.total_time = total_time
        result.simulation_time = sim_time
        result.emission_time = emission_time
        result.routing_time = routing_time
        result.overhead_time = overhead_time
        result.total_steps = step
        
        # Get statistics
        stats = self.data_collector.get_statistics()
        result.total_vehicles = stats.total_vehicles
        result.completed_trips = stats.completed_trips
        
        # Emission totals
        result.total_emissions = self.aggregated_emissions.total
        
        return result
    
    def run_baseline(
        self,
        sumo_cmd: List[str],
        end_time: float = 3600
    ) -> float:
        """
        Run a baseline sequential simulation for speedup comparison.
        
        Args:
            sumo_cmd: SUMO command
            end_time: Simulation end time
            
        Returns:
            Total execution time
        """
        # Run with single process
        old_num = self.num_processes
        self.num_processes = 1
        
        if self.emission_processor:
            self.emission_processor.num_processes = 1
        if self.route_processor:
            self.route_processor.num_processes = 1
        
        result = self.run_simulation(
            sumo_cmd, 
            end_time, 
            enable_rerouting=False,
            collect_emissions=True
        )
        
        # Restore
        self.num_processes = old_num
        if self.emission_processor:
            self.emission_processor.num_processes = old_num
        if self.route_processor:
            self.route_processor.num_processes = old_num
        
        return result.total_time
    
    def calculate_speedup(
        self,
        parallel_time: float,
        baseline_time: float
    ) -> Tuple[float, float]:
        """
        Calculate speedup and efficiency.
        
        Args:
            parallel_time: Time with parallel processing
            baseline_time: Time with sequential processing
            
        Returns:
            Tuple of (speedup, efficiency)
        """
        speedup = baseline_time / parallel_time if parallel_time > 0 else 1.0
        efficiency = speedup / self.num_processes
        return speedup, efficiency


def create_simple_network(output_dir: str, name: str = "test_network"):
    """
    Create a simple test network for demonstration.
    
    Args:
        output_dir: Directory to save network files
        name: Network name
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Simple grid network node file
    nodes_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
       xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <node id="1" x="0.0" y="0.0"/>
    <node id="2" x="500.0" y="0.0"/>
    <node id="3" x="1000.0" y="0.0"/>
    <node id="4" x="0.0" y="500.0"/>
    <node id="5" x="500.0" y="500.0"/>
    <node id="6" x="1000.0" y="500.0"/>
    <node id="7" x="0.0" y="1000.0"/>
    <node id="8" x="500.0" y="1000.0"/>
    <node id="9" x="1000.0" y="1000.0"/>
</nodes>
"""
    
    # Edge file
    edges_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
       xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <!-- Horizontal edges -->
    <edge id="1to2" from="1" to="2" numLanes="2" speed="13.89"/>
    <edge id="2to3" from="2" to="3" numLanes="2" speed="13.89"/>
    <edge id="4to5" from="4" to="5" numLanes="2" speed="13.89"/>
    <edge id="5to6" from="5" to="6" numLanes="2" speed="13.89"/>
    <edge id="7to8" from="7" to="8" numLanes="2" speed="13.89"/>
    <edge id="8to9" from="8" to="9" numLanes="2" speed="13.89"/>
    
    <!-- Reverse horizontal -->
    <edge id="2to1" from="2" to="1" numLanes="2" speed="13.89"/>
    <edge id="3to2" from="3" to="2" numLanes="2" speed="13.89"/>
    <edge id="5to4" from="5" to="4" numLanes="2" speed="13.89"/>
    <edge id="6to5" from="6" to="5" numLanes="2" speed="13.89"/>
    <edge id="8to7" from="8" to="7" numLanes="2" speed="13.89"/>
    <edge id="9to8" from="9" to="8" numLanes="2" speed="13.89"/>
    
    <!-- Vertical edges -->
    <edge id="1to4" from="1" to="4" numLanes="2" speed="13.89"/>
    <edge id="4to7" from="4" to="7" numLanes="2" speed="13.89"/>
    <edge id="2to5" from="2" to="5" numLanes="2" speed="13.89"/>
    <edge id="5to8" from="5" to="8" numLanes="2" speed="13.89"/>
    <edge id="3to6" from="3" to="6" numLanes="2" speed="13.89"/>
    <edge id="6to9" from="6" to="9" numLanes="2" speed="13.89"/>
    
    <!-- Reverse vertical -->
    <edge id="4to1" from="4" to="1" numLanes="2" speed="13.89"/>
    <edge id="7to4" from="7" to="4" numLanes="2" speed="13.89"/>
    <edge id="5to2" from="5" to="2" numLanes="2" speed="13.89"/>
    <edge id="8to5" from="8" to="5" numLanes="2" speed="13.89"/>
    <edge id="6to3" from="6" to="3" numLanes="2" speed="13.89"/>
    <edge id="9to6" from="9" to="6" numLanes="2" speed="13.89"/>
</edges>
"""
    
    # Save files
    with open(os.path.join(output_dir, f"{name}.nod.xml"), "w") as f:
        f.write(nodes_xml)
    
    with open(os.path.join(output_dir, f"{name}.edg.xml"), "w") as f:
        f.write(edges_xml)
    
    print(f"Created network definition files in {output_dir}")
    print(f"Run: netconvert -n {name}.nod.xml -e {name}.edg.xml -o {name}.net.xml")
    
    return output_dir


