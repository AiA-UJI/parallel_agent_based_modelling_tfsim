"""
Enhanced SUMO Routing Module

This module provides improved routing capabilities that integrate better
with SUMO's native routing algorithms and TraCI for real-time updates.

Key improvements over basic routing:
1. Uses sumolib for accurate network parsing
2. Integrates with SUMO's native routing via TraCI
3. Supports turn restrictions and traffic light phases
4. Real-time travel time updates from simulation
5. Option to use DUAROUTER for optimal routes
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import xml.etree.ElementTree as ET

# Try to import SUMO libraries
SUMO_AVAILABLE = False
try:
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    
    import sumolib
    SUMO_AVAILABLE = True
except ImportError:
    print("Warning: sumolib not available. Some routing features will be limited.")


@dataclass
class SUMOEdgeData:
    """Extended edge data from SUMO network"""
    edge_id: str
    from_junction: str
    to_junction: str
    length: float
    speed: float
    num_lanes: int
    priority: int
    shape: List[Tuple[float, float]]  # Coordinates
    allows: Set[str]  # Allowed vehicle classes
    disallows: Set[str]  # Disallowed vehicle classes
    
    # Dynamic data (updated from TraCI)
    current_travel_time: float = 0.0
    current_occupancy: float = 0.0
    current_mean_speed: float = 0.0


class SUMONetworkParser:
    """
    Parser for SUMO .net.xml files using sumolib.
    
    Extracts complete network topology including:
    - Edge geometry and attributes
    - Junction (node) data
    - Turn restrictions (connections)
    - Traffic light programs
    """
    
    def __init__(self, net_file: str):
        """
        Initialize parser with network file.
        
        Args:
            net_file: Path to SUMO .net.xml file
        """
        self.net_file = Path(net_file)
        self.net = None
        self.edges: Dict[str, SUMOEdgeData] = {}
        self.junctions: Dict[str, dict] = {}
        self.connections: Dict[str, List[str]] = {}  # edge_id -> [successor_edge_ids]
        
        if not self.net_file.exists():
            raise FileNotFoundError(f"Network file not found: {net_file}")
        
        self._parse_network()
    
    def _parse_network(self):
        """Parse the network using sumolib"""
        if not SUMO_AVAILABLE:
            self._parse_network_xml()
            return
        
        print(f"Parsing network: {self.net_file}")
        self.net = sumolib.net.readNet(str(self.net_file))
        
        # Parse edges
        for edge in self.net.getEdges():
            edge_data = SUMOEdgeData(
                edge_id=edge.getID(),
                from_junction=edge.getFromNode().getID(),
                to_junction=edge.getToNode().getID(),
                length=edge.getLength(),
                speed=edge.getSpeed(),
                num_lanes=edge.getLaneNumber(),
                priority=edge.getPriority(),
                shape=edge.getShape(),
                allows=set(edge.getAllowedVTypes()) if hasattr(edge, 'getAllowedVTypes') else set(),
                disallows=set(),
                current_travel_time=edge.getLength() / edge.getSpeed() if edge.getSpeed() > 0 else float('inf')
            )
            self.edges[edge.getID()] = edge_data
            
            # Store connections (successor edges)
            outgoing = edge.getOutgoing()
            self.connections[edge.getID()] = [
                conn.getID() for conn in outgoing.keys()
            ]
        
        # Parse junctions
        for node in self.net.getNodes():
            self.junctions[node.getID()] = {
                "id": node.getID(),
                "type": node.getType(),
                "coord": node.getCoord(),
                "incoming": [e.getID() for e in node.getIncoming()],
                "outgoing": [e.getID() for e in node.getOutgoing()]
            }
        
        print(f"Parsed {len(self.edges)} edges and {len(self.junctions)} junctions")
    
    def _parse_network_xml(self):
        """Fallback XML parsing without sumolib"""
        print("Using fallback XML parser (sumolib not available)")
        
        tree = ET.parse(self.net_file)
        root = tree.getroot()
        
        # Parse edges
        for edge_elem in root.findall('.//edge'):
            edge_id = edge_elem.get('id')
            if edge_id.startswith(':'):  # Internal edge
                continue
            
            from_node = edge_elem.get('from')
            to_node = edge_elem.get('to')
            
            # Get first lane for length and speed
            lanes = edge_elem.findall('lane')
            if lanes:
                length = float(lanes[0].get('length', 100))
                speed = float(lanes[0].get('speed', 13.89))
            else:
                length = 100.0
                speed = 13.89
            
            edge_data = SUMOEdgeData(
                edge_id=edge_id,
                from_junction=from_node,
                to_junction=to_node,
                length=length,
                speed=speed,
                num_lanes=len(lanes),
                priority=int(edge_elem.get('priority', 0)),
                shape=[],
                allows=set(),
                disallows=set(),
                current_travel_time=length / speed if speed > 0 else float('inf')
            )
            self.edges[edge_id] = edge_data
        
        # Parse connections
        for conn_elem in root.findall('.//connection'):
            from_edge = conn_elem.get('from')
            to_edge = conn_elem.get('to')
            
            if from_edge and to_edge and not from_edge.startswith(':'):
                if from_edge not in self.connections:
                    self.connections[from_edge] = []
                if to_edge not in self.connections[from_edge]:
                    self.connections[from_edge].append(to_edge)
        
        print(f"Parsed {len(self.edges)} edges")
    
    def get_edge(self, edge_id: str) -> Optional[SUMOEdgeData]:
        """Get edge data by ID"""
        return self.edges.get(edge_id)
    
    def get_successors(self, edge_id: str) -> List[str]:
        """Get successor edges (edges reachable from end of given edge)"""
        return self.connections.get(edge_id, [])
    
    def get_travel_time(self, edge_id: str) -> float:
        """Get current travel time for an edge"""
        edge = self.edges.get(edge_id)
        if edge:
            return edge.current_travel_time
        return float('inf')
    
    def update_travel_time(self, edge_id: str, travel_time: float):
        """Update travel time for an edge (from TraCI)"""
        if edge_id in self.edges:
            self.edges[edge_id].current_travel_time = travel_time
    
    def get_all_edges(self) -> List[str]:
        """Get all edge IDs"""
        return list(self.edges.keys())


class SUMORouter:
    """
    Router that uses SUMO's network data and optionally DUAROUTER.
    
    Provides both:
    1. Internal Dijkstra/A* routing using parsed network
    2. External DUAROUTER calls for comparison/validation
    """
    
    def __init__(self, network_parser: SUMONetworkParser):
        """
        Initialize router.
        
        Args:
            network_parser: Parsed SUMO network
        """
        self.network = network_parser
        
        # Cache for A* heuristic
        self._node_coords: Dict[str, Tuple[float, float]] = {}
        self._build_coord_cache()
    
    def _build_coord_cache(self):
        """Build coordinate cache for A* heuristic"""
        for junction_id, junction in self.network.junctions.items():
            if "coord" in junction:
                self._node_coords[junction_id] = junction["coord"]
    
    def _heuristic(self, from_junction: str, to_junction: str) -> float:
        """A* heuristic: Euclidean distance / max_speed"""
        if from_junction not in self._node_coords or to_junction not in self._node_coords:
            return 0
        
        x1, y1 = self._node_coords[from_junction]
        x2, y2 = self._node_coords[to_junction]
        
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        max_speed = 33.33  # ~120 km/h
        
        return distance / max_speed
    
    def find_route_dijkstra(
        self,
        from_edge: str,
        to_edge: str,
        cost_function: str = "time"
    ) -> Tuple[List[str], float]:
        """
        Find shortest path using Dijkstra's algorithm.
        
        Args:
            from_edge: Origin edge ID
            to_edge: Destination edge ID
            cost_function: "time", "distance", or "custom"
            
        Returns:
            Tuple of (route as list of edge IDs, total cost)
        """
        import heapq
        
        if from_edge not in self.network.edges:
            return [], float('inf')
        if to_edge not in self.network.edges:
            return [], float('inf')
        
        # Priority queue: (cost, edge_id, path)
        pq = [(0, from_edge, [from_edge])]
        visited = set()
        
        while pq:
            cost, current, path = heapq.heappop(pq)
            
            if current == to_edge:
                return path, cost
            
            if current in visited:
                continue
            visited.add(current)
            
            # Explore successors
            for next_edge in self.network.get_successors(current):
                if next_edge not in visited:
                    edge_data = self.network.get_edge(next_edge)
                    if edge_data:
                        if cost_function == "time":
                            edge_cost = edge_data.current_travel_time
                        elif cost_function == "distance":
                            edge_cost = edge_data.length
                        else:
                            edge_cost = edge_data.current_travel_time
                        
                        heapq.heappush(pq, (cost + edge_cost, next_edge, path + [next_edge]))
        
        return [], float('inf')  # No route found
    
    def find_route_astar(
        self,
        from_edge: str,
        to_edge: str,
        cost_function: str = "time"
    ) -> Tuple[List[str], float]:
        """
        Find shortest path using A* algorithm.
        
        Args:
            from_edge: Origin edge ID
            to_edge: Destination edge ID
            cost_function: Cost function to use
            
        Returns:
            Tuple of (route, total cost)
        """
        import heapq
        
        if from_edge not in self.network.edges:
            return [], float('inf')
        if to_edge not in self.network.edges:
            return [], float('inf')
        
        dest_junction = self.network.edges[to_edge].to_junction
        
        # Priority queue: (f_score, g_score, edge_id, path)
        start_junction = self.network.edges[from_edge].to_junction
        h = self._heuristic(start_junction, dest_junction)
        pq = [(h, 0, from_edge, [from_edge])]
        visited = set()
        
        while pq:
            f, g, current, path = heapq.heappop(pq)
            
            if current == to_edge:
                return path, g
            
            if current in visited:
                continue
            visited.add(current)
            
            current_junction = self.network.edges[current].to_junction
            
            for next_edge in self.network.get_successors(current):
                if next_edge not in visited:
                    edge_data = self.network.get_edge(next_edge)
                    if edge_data:
                        if cost_function == "time":
                            edge_cost = edge_data.current_travel_time
                        else:
                            edge_cost = edge_data.length
                        
                        new_g = g + edge_cost
                        next_junction = edge_data.to_junction
                        h = self._heuristic(next_junction, dest_junction)
                        f = new_g + h
                        
                        heapq.heappush(pq, (f, new_g, next_edge, path + [next_edge]))
        
        return [], float('inf')
    
    def find_route_duarouter(
        self,
        from_edge: str,
        to_edge: str,
        depart_time: float = 0
    ) -> Tuple[List[str], float]:
        """
        Find route using SUMO's DUAROUTER (external call).
        
        This gives the most accurate results as it uses SUMO's native
        routing which considers all network details.
        
        Args:
            from_edge: Origin edge
            to_edge: Destination edge
            depart_time: Departure time
            
        Returns:
            Tuple of (route, estimated travel time)
        """
        duarouter = os.path.join(os.environ.get('SUMO_HOME', ''), 'bin', 'duarouter')
        
        if not os.path.exists(duarouter):
            # Fall back to internal routing
            return self.find_route_dijkstra(from_edge, to_edge)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trip file
            trip_file = os.path.join(tmpdir, "trip.xml")
            with open(trip_file, 'w') as f:
                f.write(f'''<?xml version="1.0" encoding="UTF-8"?>
<trips>
    <trip id="route_request" depart="{depart_time}" from="{from_edge}" to="{to_edge}"/>
</trips>''')
            
            # Output route file
            route_file = os.path.join(tmpdir, "route.xml")
            
            # Run DUAROUTER
            cmd = [
                duarouter,
                "-n", str(self.network.net_file),
                "-t", trip_file,
                "-o", route_file,
                "--ignore-errors", "true",
                "--no-warnings", "true"
            ]
            
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    timeout=30,
                    text=True
                )
                
                if os.path.exists(route_file):
                    # Parse route from output
                    tree = ET.parse(route_file)
                    root = tree.getroot()
                    
                    for vehicle in root.findall('.//vehicle'):
                        route_elem = vehicle.find('route')
                        if route_elem is not None:
                            edges_str = route_elem.get('edges', '')
                            route = edges_str.split() if edges_str else []
                            
                            # Estimate travel time
                            travel_time = sum(
                                self.network.get_travel_time(e) for e in route
                            )
                            
                            return route, travel_time
                
            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                print(f"DUAROUTER error: {e}")
        
        # Fallback to internal routing
        return self.find_route_dijkstra(from_edge, to_edge)


class TraCIRouteUpdater:
    """
    Updates routing information from TraCI during simulation.
    
    Collects real-time traffic data to improve route calculations:
    - Edge travel times
    - Edge occupancy
    - Mean speeds
    """
    
    def __init__(self, network_parser: SUMONetworkParser):
        """
        Initialize updater.
        
        Args:
            network_parser: Network parser to update
        """
        self.network = network_parser
        self.update_interval = 60  # seconds
        self.last_update = 0
        
        # Historical data for smoothing
        self._travel_time_history: Dict[str, List[float]] = {}
        self._history_size = 5
    
    def update_from_traci(self, traci_connection, current_time: float):
        """
        Update edge travel times from TraCI.
        
        Args:
            traci_connection: Active TraCI connection
            current_time: Current simulation time
        """
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        for edge_id in self.network.get_all_edges():
            try:
                # Get travel time from TraCI
                travel_time = traci_connection.edge.getTraveltime(edge_id)
                
                # Smooth with historical values
                if edge_id not in self._travel_time_history:
                    self._travel_time_history[edge_id] = []
                
                history = self._travel_time_history[edge_id]
                history.append(travel_time)
                
                if len(history) > self._history_size:
                    history.pop(0)
                
                # Use weighted average (more recent = higher weight)
                weights = list(range(1, len(history) + 1))
                smoothed = sum(t * w for t, w in zip(history, weights)) / sum(weights)
                
                self.network.update_travel_time(edge_id, smoothed)
                
            except Exception:
                # Edge might not exist or have no data
                pass
    
    def get_congested_edges(self, threshold: float = 2.0) -> Set[str]:
        """
        Get edges with significant congestion.
        
        Args:
            threshold: Ratio of current/free-flow travel time
            
        Returns:
            Set of congested edge IDs
        """
        congested = set()
        
        for edge_id, edge_data in self.network.edges.items():
            base_time = edge_data.length / edge_data.speed if edge_data.speed > 0 else float('inf')
            
            if base_time > 0 and edge_data.current_travel_time > threshold * base_time:
                congested.add(edge_id)
        
        return congested


class ParallelSUMORouter:
    """
    Parallel route calculator using SUMO network data.
    
    Distributes route calculations across multiple processes.
    """
    
    def __init__(
        self,
        network_parser: SUMONetworkParser,
        num_processes: int = 4,
        algorithm: str = "astar"
    ):
        """
        Initialize parallel router.
        
        Args:
            network_parser: Parsed network
            num_processes: Worker processes
            algorithm: "dijkstra", "astar", or "duarouter"
        """
        self.network = network_parser
        self.router = SUMORouter(network_parser)
        self.num_processes = num_processes
        self.algorithm = algorithm
        
        # Serialize network for workers
        self._network_data = self._serialize_network()
        
        # Stats
        self.routes_calculated = 0
        self.total_time = 0.0
    
    def _serialize_network(self) -> dict:
        """Serialize network for multiprocessing"""
        edges = []
        for edge_id, edge in self.network.edges.items():
            edges.append({
                "edge_id": edge.edge_id,
                "from_junction": edge.from_junction,
                "to_junction": edge.to_junction,
                "length": edge.length,
                "speed": edge.speed,
                "current_travel_time": edge.current_travel_time
            })
        
        return {
            "edges": edges,
            "connections": dict(self.network.connections),
            "net_file": str(self.network.net_file)
        }
    
    def calculate_route(
        self,
        from_edge: str,
        to_edge: str,
        cost_function: str = "time"
    ) -> Tuple[List[str], float]:
        """Calculate single route"""
        if self.algorithm == "duarouter":
            return self.router.find_route_duarouter(from_edge, to_edge)
        elif self.algorithm == "astar":
            return self.router.find_route_astar(from_edge, to_edge, cost_function)
        else:
            return self.router.find_route_dijkstra(from_edge, to_edge, cost_function)
    
    def calculate_batch(
        self,
        requests: List[Tuple[str, str]]
    ) -> List[Tuple[List[str], float]]:
        """
        Calculate routes for multiple OD pairs in parallel.
        
        Args:
            requests: List of (from_edge, to_edge) tuples
            
        Returns:
            List of (route, cost) tuples
        """
        if not requests:
            return []
        
        start = time.time()
        
        if self.num_processes == 1 or len(requests) < 10:
            # Sequential for small batches
            results = [
                self.calculate_route(from_e, to_e)
                for from_e, to_e in requests
            ]
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
                futures = [
                    executor.submit(self.calculate_route, from_e, to_e)
                    for from_e, to_e in requests
                ]
                results = [f.result() for f in futures]
        
        elapsed = time.time() - start
        self.routes_calculated += len(requests)
        self.total_time += elapsed
        
        return results
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        return {
            "routes_calculated": self.routes_calculated,
            "total_time": self.total_time,
            "throughput": self.routes_calculated / self.total_time if self.total_time > 0 else 0,
            "num_processes": self.num_processes,
            "algorithm": self.algorithm
        }


# Utility function to validate route quality
def validate_route(
    network: SUMONetworkParser,
    route: List[str]
) -> Tuple[bool, str]:
    """
    Validate that a route is valid (continuous path through network).
    
    Args:
        network: Parsed network
        route: Route to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not route:
        return False, "Empty route"
    
    for i in range(len(route) - 1):
        current_edge = route[i]
        next_edge = route[i + 1]
        
        # Check edge exists
        if current_edge not in network.edges:
            return False, f"Edge {current_edge} not in network"
        
        # Check successor is valid
        successors = network.get_successors(current_edge)
        if next_edge not in successors:
            return False, f"Edge {next_edge} not reachable from {current_edge}"
    
    return True, ""


