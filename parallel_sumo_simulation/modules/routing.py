"""
Parallel Route/Itinerary Calculation Module

This module handles the calculation of vehicle routes and itineraries outside
the main SUMO simulation loop, enabling parallel processing of route computations.

Includes:
- Dijkstra and A* shortest path algorithms
- Travel time estimation
- Dynamic rerouting based on traffic conditions
- Parallel batch processing of route requests
"""

import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import time
from collections import defaultdict


@dataclass
class Edge:
    """Representation of a road network edge"""
    edge_id: str
    from_node: str
    to_node: str
    length: float          # meters
    speed_limit: float     # m/s
    num_lanes: int
    current_travel_time: float = 0.0  # seconds (updated dynamically)
    base_travel_time: float = 0.0     # seconds (free-flow)
    
    def __post_init__(self):
        self.base_travel_time = self.length / self.speed_limit if self.speed_limit > 0 else float('inf')
        if self.current_travel_time == 0:
            self.current_travel_time = self.base_travel_time


@dataclass
class RouteRequest:
    """Request for route calculation"""
    request_id: str
    vehicle_id: str
    origin_edge: str
    destination_edge: str
    departure_time: float
    criteria: str = "time"  # time, distance, emissions
    vehicle_class: str = "passenger"


@dataclass
class RouteResult:
    """Result of route calculation"""
    request_id: str
    vehicle_id: str
    route: List[str]           # List of edge IDs
    total_distance: float      # meters
    estimated_travel_time: float  # seconds
    estimated_emissions: float    # grams CO2
    computation_time: float       # seconds (for benchmarking)
    success: bool
    error_message: str = ""


class NetworkGraph:
    """
    Graph representation of the road network for routing.
    
    Supports efficient shortest path calculations with dynamic edge weights.
    """
    
    def __init__(self):
        """Initialize empty network graph"""
        self.edges: Dict[str, Edge] = {}
        self.adjacency: Dict[str, List[str]] = defaultdict(list)  # node -> [edge_ids]
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)  # node -> [incoming edge_ids]
        self.edge_to_node: Dict[str, Tuple[str, str]] = {}  # edge_id -> (from, to)
        
    def add_edge(self, edge: Edge):
        """Add an edge to the graph"""
        self.edges[edge.edge_id] = edge
        self.adjacency[edge.from_node].append(edge.edge_id)
        self.reverse_adjacency[edge.to_node].append(edge.edge_id)
        self.edge_to_node[edge.edge_id] = (edge.from_node, edge.to_node)
        
    def get_neighbors(self, edge_id: str) -> List[str]:
        """Get all edges reachable from the end of given edge"""
        if edge_id not in self.edge_to_node:
            return []
        _, to_node = self.edge_to_node[edge_id]
        return self.adjacency[to_node]
    
    def update_travel_times(self, travel_times: Dict[str, float]):
        """Update current travel times for edges"""
        for edge_id, travel_time in travel_times.items():
            if edge_id in self.edges:
                self.edges[edge_id].current_travel_time = travel_time
                
    def get_edge_cost(self, edge_id: str, criteria: str = "time") -> float:
        """Get the cost of traversing an edge based on criteria"""
        if edge_id not in self.edges:
            return float('inf')
            
        edge = self.edges[edge_id]
        
        if criteria == "time":
            return edge.current_travel_time
        elif criteria == "distance":
            return edge.length
        elif criteria == "emissions":
            # Simplified: emissions proportional to distance and congestion
            congestion_factor = edge.current_travel_time / edge.base_travel_time
            return edge.length * congestion_factor * 0.15  # ~150g CO2/km base
        else:
            return edge.current_travel_time


class RouteCalculator:
    """
    Calculator for vehicle routes using shortest path algorithms.
    
    Supports Dijkstra and A* algorithms with customizable cost functions.
    """
    
    def __init__(
        self, 
        network: NetworkGraph,
        algorithm: str = "dijkstra"
    ):
        """
        Initialize route calculator.
        
        Args:
            network: NetworkGraph object
            algorithm: "dijkstra" or "astar"
        """
        self.network = network
        self.algorithm = algorithm
        
        # Precompute node positions for A* heuristic
        self.node_positions: Dict[str, Tuple[float, float]] = {}
        
    def set_node_positions(self, positions: Dict[str, Tuple[float, float]]):
        """Set node positions for A* heuristic"""
        self.node_positions = positions
        
    def _heuristic(self, from_node: str, to_node: str) -> float:
        """A* heuristic: Euclidean distance / max_speed"""
        if from_node not in self.node_positions or to_node not in self.node_positions:
            return 0  # Fall back to Dijkstra behavior
            
        x1, y1 = self.node_positions[from_node]
        x2, y2 = self.node_positions[to_node]
        
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        max_speed = 33.33  # ~120 km/h assumed max
        
        return distance / max_speed
    
    def find_route(self, request: RouteRequest) -> RouteResult:
        """
        Find optimal route for a request.
        
        Args:
            request: RouteRequest with origin, destination, and criteria
            
        Returns:
            RouteResult with route and statistics
        """
        start_time = time.time()
        
        origin = request.origin_edge
        destination = request.destination_edge
        criteria = request.criteria
        
        # Validate edges exist
        if origin not in self.network.edges:
            return RouteResult(
                request_id=request.request_id,
                vehicle_id=request.vehicle_id,
                route=[],
                total_distance=0,
                estimated_travel_time=0,
                estimated_emissions=0,
                computation_time=time.time() - start_time,
                success=False,
                error_message=f"Origin edge {origin} not found"
            )
            
        if destination not in self.network.edges:
            return RouteResult(
                request_id=request.request_id,
                vehicle_id=request.vehicle_id,
                route=[],
                total_distance=0,
                estimated_travel_time=0,
                estimated_emissions=0,
                computation_time=time.time() - start_time,
                success=False,
                error_message=f"Destination edge {destination} not found"
            )
        
        # Run shortest path algorithm
        if self.algorithm == "astar":
            route = self._astar(origin, destination, criteria)
        else:
            route = self._dijkstra(origin, destination, criteria)
        
        computation_time = time.time() - start_time
        
        if not route:
            return RouteResult(
                request_id=request.request_id,
                vehicle_id=request.vehicle_id,
                route=[],
                total_distance=0,
                estimated_travel_time=0,
                estimated_emissions=0,
                computation_time=computation_time,
                success=False,
                error_message="No route found"
            )
        
        # Calculate route statistics
        total_distance = sum(
            self.network.edges[e].length for e in route if e in self.network.edges
        )
        estimated_time = sum(
            self.network.edges[e].current_travel_time for e in route if e in self.network.edges
        )
        estimated_emissions = total_distance * 0.15  # 150g CO2/km
        
        return RouteResult(
            request_id=request.request_id,
            vehicle_id=request.vehicle_id,
            route=route,
            total_distance=total_distance,
            estimated_travel_time=estimated_time,
            estimated_emissions=estimated_emissions,
            computation_time=computation_time,
            success=True
        )
    
    def _dijkstra(
        self, 
        origin: str, 
        destination: str,
        criteria: str
    ) -> List[str]:
        """
        Dijkstra's shortest path algorithm.
        
        Returns:
            List of edge IDs forming the shortest path
        """
        # Priority queue: (cost, edge_id, path)
        pq = [(0, origin, [origin])]
        visited: Set[str] = set()
        
        while pq:
            cost, current_edge, path = heapq.heappop(pq)
            
            if current_edge == destination:
                return path
                
            if current_edge in visited:
                continue
                
            visited.add(current_edge)
            
            # Explore neighbors
            for next_edge in self.network.get_neighbors(current_edge):
                if next_edge not in visited:
                    edge_cost = self.network.get_edge_cost(next_edge, criteria)
                    new_cost = cost + edge_cost
                    heapq.heappush(pq, (new_cost, next_edge, path + [next_edge]))
        
        return []  # No path found
    
    def _astar(
        self, 
        origin: str, 
        destination: str,
        criteria: str
    ) -> List[str]:
        """
        A* shortest path algorithm.
        
        Returns:
            List of edge IDs forming the shortest path
        """
        # Get destination node for heuristic
        if destination not in self.network.edge_to_node:
            return self._dijkstra(origin, destination, criteria)
        dest_node = self.network.edge_to_node[destination][1]
        
        # Priority queue: (f_score, g_score, edge_id, path)
        pq = [(0, 0, origin, [origin])]
        visited: Set[str] = set()
        
        while pq:
            _, g_score, current_edge, path = heapq.heappop(pq)
            
            if current_edge == destination:
                return path
                
            if current_edge in visited:
                continue
                
            visited.add(current_edge)
            
            # Get current node
            if current_edge not in self.network.edge_to_node:
                continue
            current_node = self.network.edge_to_node[current_edge][1]
            
            # Explore neighbors
            for next_edge in self.network.get_neighbors(current_edge):
                if next_edge not in visited:
                    edge_cost = self.network.get_edge_cost(next_edge, criteria)
                    new_g = g_score + edge_cost
                    
                    # Calculate heuristic
                    if next_edge in self.network.edge_to_node:
                        next_node = self.network.edge_to_node[next_edge][1]
                        h = self._heuristic(next_node, dest_node)
                    else:
                        h = 0
                    
                    f_score = new_g + h
                    heapq.heappush(pq, (f_score, new_g, next_edge, path + [next_edge]))
        
        return []  # No path found
    
    def calculate_batch(self, requests: List[RouteRequest]) -> List[RouteResult]:
        """
        Calculate routes for a batch of requests.
        
        Args:
            requests: List of route requests
            
        Returns:
            List of route results
        """
        return [self.find_route(req) for req in requests]


def _route_calculation_worker(args: Tuple[List[dict], dict, str]) -> List[dict]:
    """
    Worker function for parallel route calculation.
    
    Args:
        args: Tuple of (requests as dicts, network data, algorithm)
        
    Returns:
        List of route result dicts
    """
    requests_data, network_data, algorithm = args
    
    # Reconstruct network
    network = NetworkGraph()
    for edge_data in network_data["edges"]:
        edge = Edge(
            edge_id=edge_data["edge_id"],
            from_node=edge_data["from_node"],
            to_node=edge_data["to_node"],
            length=edge_data["length"],
            speed_limit=edge_data["speed_limit"],
            num_lanes=edge_data["num_lanes"],
            current_travel_time=edge_data.get("current_travel_time", 0)
        )
        network.add_edge(edge)
    
    calculator = RouteCalculator(network, algorithm)
    
    # Set node positions if available
    if "node_positions" in network_data:
        calculator.set_node_positions(network_data["node_positions"])
    
    # Convert requests
    requests = [
        RouteRequest(
            request_id=r["request_id"],
            vehicle_id=r["vehicle_id"],
            origin_edge=r["origin_edge"],
            destination_edge=r["destination_edge"],
            departure_time=r["departure_time"],
            criteria=r.get("criteria", "time")
        )
        for r in requests_data
    ]
    
    results = calculator.calculate_batch(requests)
    
    # Convert to dicts
    return [
        {
            "request_id": r.request_id,
            "vehicle_id": r.vehicle_id,
            "route": r.route,
            "total_distance": r.total_distance,
            "estimated_travel_time": r.estimated_travel_time,
            "estimated_emissions": r.estimated_emissions,
            "computation_time": r.computation_time,
            "success": r.success,
            "error_message": r.error_message
        }
        for r in results
    ]


class ParallelRouteProcessor:
    """
    Parallel processor for batch route calculations.
    
    This class manages the parallel computation of routes for large
    numbers of requests, distributing work across multiple processes.
    """
    
    def __init__(
        self, 
        network: NetworkGraph,
        num_processes: Optional[int] = None,
        batch_size: int = 50,
        algorithm: str = "dijkstra"
    ):
        """
        Initialize parallel route processor.
        
        Args:
            network: NetworkGraph object
            num_processes: Number of worker processes (None = CPU count)
            batch_size: Number of route requests per batch
            algorithm: Routing algorithm to use
        """
        self.network = network
        self.num_processes = num_processes or cpu_count()
        self.batch_size = batch_size
        self.algorithm = algorithm
        
        # Serialize network for workers
        self._network_data = self._serialize_network()
        
        # Performance metrics
        self.total_processed = 0
        self.total_time = 0.0
        self.successful_routes = 0
        
    def _serialize_network(self) -> dict:
        """Serialize network graph for multiprocessing"""
        edges_data = []
        for edge_id, edge in self.network.edges.items():
            edges_data.append({
                "edge_id": edge.edge_id,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "length": edge.length,
                "speed_limit": edge.speed_limit,
                "num_lanes": edge.num_lanes,
                "current_travel_time": edge.current_travel_time
            })
        
        return {
            "edges": edges_data,
            "node_positions": dict(getattr(self, 'node_positions', {}))
        }
    
    def update_network(self, travel_times: Dict[str, float]):
        """Update network travel times and re-serialize"""
        self.network.update_travel_times(travel_times)
        self._network_data = self._serialize_network()
    
    def _create_batches(self, requests: List[dict]) -> List[List[dict]]:
        """Split requests into batches for parallel processing"""
        batches = []
        for i in range(0, len(requests), self.batch_size):
            batches.append(requests[i:i + self.batch_size])
        return batches
    
    def process_routes(self, route_requests: List[dict]) -> List[dict]:
        """
        Process route requests in parallel.
        
        Args:
            route_requests: List of route request dictionaries with keys:
                - request_id, vehicle_id, origin_edge, destination_edge,
                  departure_time, criteria (optional)
                  
        Returns:
            List of route result dictionaries
        """
        if not route_requests:
            return []
        
        start_time = time.time()
        
        # Create batches
        batches = self._create_batches(route_requests)
        
        # Prepare arguments for workers
        worker_args = [
            (batch, self._network_data, self.algorithm) 
            for batch in batches
        ]
        
        all_results = []
        
        if self.num_processes == 1:
            # Sequential processing for baseline
            for args in worker_args:
                results = _route_calculation_worker(args)
                all_results.extend(results)
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                futures = [
                    executor.submit(_route_calculation_worker, args)
                    for args in worker_args
                ]
                
                for future in as_completed(futures):
                    results = future.result()
                    all_results.extend(results)
        
        elapsed = time.time() - start_time
        
        # Update metrics
        self.total_processed += len(route_requests)
        self.total_time += elapsed
        self.successful_routes += sum(1 for r in all_results if r["success"])
        
        return all_results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            "total_processed": self.total_processed,
            "successful_routes": self.successful_routes,
            "success_rate": self.successful_routes / self.total_processed if self.total_processed > 0 else 0,
            "total_time": self.total_time,
            "throughput": self.total_processed / self.total_time if self.total_time > 0 else 0,
            "num_processes": self.num_processes
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.total_processed = 0
        self.total_time = 0.0
        self.successful_routes = 0


class DynamicRerouter:
    """
    Handles dynamic rerouting based on changing traffic conditions.
    """
    
    def __init__(
        self,
        route_processor: ParallelRouteProcessor,
        rerouting_period: int = 60,
        rerouting_probability: float = 0.1
    ):
        """
        Initialize dynamic rerouter.
        
        Args:
            route_processor: ParallelRouteProcessor instance
            rerouting_period: Time between rerouting checks (seconds)
            rerouting_probability: Probability of rerouting a vehicle
        """
        self.route_processor = route_processor
        self.rerouting_period = rerouting_period
        self.rerouting_probability = rerouting_probability
        
        # Track current routes
        self.current_routes: Dict[str, List[str]] = {}
        self.last_reroute_time: Dict[str, float] = {}
        
    def update_vehicle_route(self, vehicle_id: str, route: List[str]):
        """Update stored route for a vehicle"""
        self.current_routes[vehicle_id] = route
        
    def check_rerouting(
        self, 
        current_time: float,
        vehicle_positions: Dict[str, str],  # vehicle_id -> current_edge
        vehicle_destinations: Dict[str, str],  # vehicle_id -> destination_edge
        congested_edges: Set[str]  # edges with significant congestion
    ) -> Dict[str, List[str]]:
        """
        Check which vehicles should be rerouted and compute new routes.
        
        Args:
            current_time: Current simulation time
            vehicle_positions: Current edge for each vehicle
            vehicle_destinations: Destination for each vehicle
            congested_edges: Set of congested edge IDs
            
        Returns:
            Dictionary of vehicle_id -> new route for vehicles that should reroute
        """
        reroute_requests = []
        
        for vehicle_id, current_edge in vehicle_positions.items():
            # Check if enough time has passed since last reroute
            last_time = self.last_reroute_time.get(vehicle_id, 0)
            if current_time - last_time < self.rerouting_period:
                continue
            
            # Check if current route passes through congested edges
            if vehicle_id in self.current_routes:
                current_route = self.current_routes[vehicle_id]
                route_congested = any(e in congested_edges for e in current_route)
                
                if route_congested and np.random.random() < self.rerouting_probability:
                    if vehicle_id in vehicle_destinations:
                        reroute_requests.append({
                            "request_id": f"reroute_{vehicle_id}_{current_time}",
                            "vehicle_id": vehicle_id,
                            "origin_edge": current_edge,
                            "destination_edge": vehicle_destinations[vehicle_id],
                            "departure_time": current_time,
                            "criteria": "time"
                        })
                        self.last_reroute_time[vehicle_id] = current_time
        
        if not reroute_requests:
            return {}
        
        # Calculate new routes in parallel
        results = self.route_processor.process_routes(reroute_requests)
        
        new_routes = {}
        for result in results:
            if result["success"]:
                new_routes[result["vehicle_id"]] = result["route"]
                self.current_routes[result["vehicle_id"]] = result["route"]
        
        return new_routes


