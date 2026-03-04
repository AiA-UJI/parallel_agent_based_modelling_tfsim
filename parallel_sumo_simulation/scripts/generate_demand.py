#!/usr/bin/env python3
"""
Traffic Demand Generation Script

Generates SUMO route files (.rou.xml) with different traffic levels
for benchmarking parallel simulation.
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import NETWORKS_DIR, TRAFFIC_CONFIGS


def generate_route_file(
    net_file: str,
    output_file: str,
    vehicles_per_hour: int,
    simulation_time: float = 3600,
    vehicle_types: List[str] = None,
    seed: int = 42
):
    """
    Generate a SUMO route file with random trips.
    
    Args:
        net_file: Path to SUMO network file
        output_file: Output route file path
        vehicles_per_hour: Vehicle departure rate
        simulation_time: Total simulation time in seconds
        vehicle_types: List of vehicle types to use
        seed: Random seed
    """
    try:
        import sumolib
    except ImportError:
        print("Error: sumolib not found. Please install SUMO and set SUMO_HOME.")
        return False
    
    random.seed(seed)
    
    # Load network
    net = sumolib.net.readNet(net_file)
    
    # Get all edges
    edges = [e for e in net.getEdges() if e.allows("passenger")]
    if not edges:
        edges = list(net.getEdges())
    
    edge_ids = [e.getID() for e in edges]
    
    if len(edge_ids) < 2:
        print("Error: Network needs at least 2 edges")
        return False
    
    # Calculate number of vehicles
    total_vehicles = int(vehicles_per_hour * simulation_time / 3600)
    
    # Default vehicle types
    if vehicle_types is None:
        vehicle_types = ["passenger"]
    
    # Generate XML content
    xml_content = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml_content.append('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                      'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
    
    # Vehicle types
    xml_content.append('    <vType id="passenger" accel="2.6" decel="4.5" '
                      'sigma="0.5" length="5" maxSpeed="50" guiShape="passenger"/>')
    
    # Generate vehicles with random routes
    for i in range(total_vehicles):
        # Random departure time (uniform distribution)
        depart_time = random.uniform(0, simulation_time)
        
        # Random origin and destination
        origin = random.choice(edge_ids)
        destination = random.choice([e for e in edge_ids if e != origin])
        
        veh_type = random.choice(vehicle_types)
        
        xml_content.append(
            f'    <trip id="veh_{i}" type="{veh_type}" depart="{depart_time:.2f}" '
            f'from="{origin}" to="{destination}"/>'
        )
    
    xml_content.append('</routes>')
    
    # Write file
    with open(output_file, 'w') as f:
        f.write('\n'.join(xml_content))
    
    print(f"Generated {total_vehicles} vehicles in {output_file}")
    return True


def generate_demand_for_network(
    network_name: str,
    simulation_time: float = 3600,
    seed: int = 42
):
    """
    Generate demand files for all traffic levels for a network.
    
    Args:
        network_name: Network directory name
        simulation_time: Simulation duration
        seed: Random seed
    """
    network_dir = NETWORKS_DIR / network_name.lower()
    net_file = network_dir / f"{network_name.lower()}.net.xml"
    
    if not net_file.exists():
        print(f"Error: Network file not found: {net_file}")
        return False
    
    for level, config in TRAFFIC_CONFIGS.items():
        output_file = network_dir / f"{network_name.lower()}_{level.lower()}.rou.xml"
        
        print(f"\nGenerating {level} traffic demand...")
        generate_route_file(
            str(net_file),
            str(output_file),
            config["vehicles_per_hour"],
            simulation_time,
            seed=seed
        )
    
    return True


def generate_sumocfg(
    network_name: str,
    traffic_level: str = "medium",
    simulation_time: float = 3600
):
    """
    Generate SUMO configuration file.
    
    Args:
        network_name: Network directory name
        traffic_level: Traffic level for route file
        simulation_time: Simulation duration
    """
    network_dir = NETWORKS_DIR / network_name.lower()
    
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{network_name.lower()}.net.xml"/>
        <route-files value="{network_name.lower()}_{traffic_level.lower()}.rou.xml"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="{int(simulation_time)}"/>
        <step-length value="1.0"/>
    </time>

    <processing>
        <time-to-teleport value="-1"/>
        <ignore-route-errors value="true"/>
    </processing>

    <routing>
        <routing-algorithm value="dijkstra"/>
    </routing>

    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>

</configuration>
"""
    
    config_file = network_dir / f"{network_name.lower()}.sumocfg"
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"Generated config: {config_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate traffic demand for SUMO simulation"
    )
    
    parser.add_argument(
        "--network", "-n",
        type=str,
        required=True,
        help="Network name (directory in networks/)"
    )
    
    parser.add_argument(
        "--time",
        type=float,
        default=3600,
        help="Simulation time in seconds (default: 3600)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--level",
        choices=["low", "medium", "high", "all"],
        default="all",
        help="Traffic level to generate (default: all)"
    )
    
    parser.add_argument(
        "--vehicles",
        type=int,
        default=None,
        help="Custom vehicles per hour (overrides level)"
    )
    
    parser.add_argument(
        "--config",
        action="store_true",
        help="Also generate SUMO config file"
    )
    
    args = parser.parse_args()
    
    network_dir = NETWORKS_DIR / args.network.lower()
    net_file = network_dir / f"{args.network.lower()}.net.xml"
    
    if not net_file.exists():
        print(f"Error: Network file not found: {net_file}")
        print(f"\nMake sure to create the network first using netconvert.")
        sys.exit(1)
    
    if args.vehicles:
        # Custom vehicle count
        output_file = network_dir / f"{args.network.lower()}_custom.rou.xml"
        generate_route_file(
            str(net_file),
            str(output_file),
            args.vehicles,
            args.time,
            seed=args.seed
        )
    elif args.level == "all":
        # Generate all traffic levels
        generate_demand_for_network(args.network, args.time, args.seed)
    else:
        # Single level
        config = TRAFFIC_CONFIGS[args.level.capitalize()]
        output_file = network_dir / f"{args.network.lower()}_{args.level}.rou.xml"
        generate_route_file(
            str(net_file),
            str(output_file),
            config["vehicles_per_hour"],
            args.time,
            seed=args.seed
        )
    
    if args.config:
        generate_sumocfg(args.network, args.level if args.level != "all" else "medium", args.time)


if __name__ == "__main__":
    main()


