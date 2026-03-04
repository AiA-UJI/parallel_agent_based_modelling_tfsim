#!/usr/bin/env python3
"""
Main Script to Run Parallel SUMO Simulation

This script provides a command-line interface to run SUMO simulations
with parallel emission and routing calculations.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    SimulationConfig, TRAFFIC_CONFIGS, SUMO_BINARY, 
    NETWORKS_DIR, RESULTS_DIR
)
from modules.simulation import ParallelSUMOSimulator, create_simple_network


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel SUMO traffic simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_simulation.py --network almenara
  
  # Run with 8 processes and high traffic
  python run_simulation.py --network rotterdam --processes 8 --traffic high
  
  # Run with accidents
  python run_simulation.py --network almenara --accidents 1 --time 3600
  
  # Create test network
  python run_simulation.py --create-network test_grid
        """
    )
    
    parser.add_argument(
        "--network", "-n",
        type=str,
        help="Network name (directory in networks/)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to SUMO config file (.sumocfg)"
    )
    
    parser.add_argument(
        "--processes", "-p",
        type=int,
        default=4,
        help="Number of parallel processes (default: 4)"
    )
    
    parser.add_argument(
        "--traffic", "-t",
        choices=["low", "medium", "high"],
        default="medium",
        help="Traffic level (default: medium)"
    )
    
    parser.add_argument(
        "--time",
        type=float,
        default=3600,
        help="Simulation time in seconds (default: 3600)"
    )
    
    parser.add_argument(
        "--accidents",
        type=int,
        default=0,
        help="Number of accidents to simulate (default: 0)"
    )
    
    parser.add_argument(
        "--no-rerouting",
        action="store_true",
        help="Disable dynamic rerouting"
    )
    
    parser.add_argument(
        "--no-emissions",
        action="store_true",
        help="Disable emission calculations"
    )
    
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Use SUMO-GUI instead of command line SUMO"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--create-network",
        type=str,
        metavar="NAME",
        help="Create a simple test network with given name"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for parallel emission processing"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Create test network if requested
    if args.create_network:
        output_dir = NETWORKS_DIR / args.create_network
        create_simple_network(str(output_dir), args.create_network)
        print(f"\nNetwork files created in: {output_dir}")
        print("\nTo generate the SUMO network, run:")
        print(f"  cd {output_dir}")
        print(f"  netconvert -n {args.create_network}.nod.xml -e {args.create_network}.edg.xml -o {args.create_network}.net.xml")
        return
    
    # Validate network/config
    if not args.network and not args.config:
        parser.error("Either --network or --config must be specified")
    
    # Build SUMO command
    if args.config:
        config_file = Path(args.config)
        if not config_file.exists():
            print(f"Error: Config file not found: {config_file}")
            sys.exit(1)
        
        sumo_binary = os.environ.get("SUMO_GUI", "sumo-gui") if args.gui else SUMO_BINARY
        sumo_cmd = [
            sumo_binary,
            "-c", str(config_file),
            "--step-length", "1.0",
            "--seed", str(args.seed),
            "--no-warnings", "true",
            "--no-step-log", "true"
        ]
        network_file = None
        
    else:
        network_dir = NETWORKS_DIR / args.network.lower()
        network_file = network_dir / f"{args.network.lower()}.net.xml"
        
        if not network_file.exists():
            print(f"Error: Network file not found: {network_file}")
            print(f"\nAvailable networks:")
            for d in NETWORKS_DIR.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
            sys.exit(1)
        
        # Look for route file
        route_file = network_dir / f"{args.network.lower()}_{args.traffic}.rou.xml"
        if not route_file.exists():
            route_file = network_dir / f"{args.network.lower()}.rou.xml"
        
        sumo_binary = os.environ.get("SUMO_GUI", "sumo-gui") if args.gui else SUMO_BINARY
        
        sumo_cmd = [
            sumo_binary,
            "-n", str(network_file),
            "--step-length", "1.0",
            "--seed", str(args.seed),
            "--no-warnings", "true",
            "--no-step-log", "true"
        ]
        
        if route_file.exists():
            sumo_cmd.extend(["-r", str(route_file)])
        else:
            print(f"Warning: No route file found at {route_file}")
            print("Simulation may not have any vehicles.")
    
    print("\n" + "="*60)
    print(" PARALLEL SUMO SIMULATION")
    print("="*60)
    print(f"Network: {args.network or args.config}")
    print(f"Processes: {args.processes}")
    print(f"Traffic: {args.traffic}")
    print(f"Time: {args.time}s")
    print(f"Accidents: {args.accidents}")
    print(f"Rerouting: {not args.no_rerouting}")
    print(f"Emissions: {not args.no_emissions}")
    print("="*60 + "\n")
    
    # Initialize simulator
    simulator = ParallelSUMOSimulator(
        num_processes=args.processes,
        emission_batch_size=args.batch_size,
        use_async_processing=(args.processes > 1)
    )
    
    # Load network
    if network_file and network_file.exists():
        print(f"Loading network: {network_file}")
        simulator.load_network(str(network_file))
    
    # Run simulation
    try:
        result = simulator.run_simulation(
            sumo_cmd=sumo_cmd,
            end_time=args.time,
            enable_rerouting=not args.no_rerouting,
            accident_edges=None,  # TODO: implement accident edge selection
            collect_emissions=not args.no_emissions
        )
        
        print("\n" + "="*60)
        print(" SIMULATION RESULTS")
        print("="*60)
        print(f"Total time: {result.total_time:.2f}s")
        print(f"  - Simulation: {result.simulation_time:.2f}s")
        print(f"  - Emissions:  {result.emission_time:.2f}s")
        print(f"  - Routing:    {result.routing_time:.2f}s")
        print(f"  - Overhead:   {result.overhead_time:.2f}s")
        print(f"\nTotal vehicles: {result.total_vehicles}")
        print(f"Completed trips: {result.completed_trips}")
        print(f"Total steps: {result.total_steps}")
        
        if result.total_emissions:
            print(f"\nTotal emissions:")
            for pollutant, value in result.total_emissions.items():
                if pollutant == "fuel":
                    print(f"  - {pollutant}: {value:.2f} L")
                else:
                    print(f"  - {pollutant}: {value/1000:.2f} kg")
        
        # Save results
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = RESULTS_DIR
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        from datetime import datetime
        
        result_file = output_dir / f"simulation_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        result_dict = {
            "config": {
                "network": args.network or args.config,
                "processes": args.processes,
                "traffic": args.traffic,
                "time": args.time,
                "accidents": args.accidents,
                "seed": args.seed
            },
            "timing": {
                "total": result.total_time,
                "simulation": result.simulation_time,
                "emissions": result.emission_time,
                "routing": result.routing_time,
                "overhead": result.overhead_time
            },
            "statistics": {
                "total_vehicles": result.total_vehicles,
                "completed_trips": result.completed_trips,
                "total_steps": result.total_steps
            },
            "emissions": result.total_emissions
        }
        
        with open(result_file, "w") as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"\nResults saved to: {result_file}")
        
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


