#!/usr/bin/env python3
"""
Analysis and Visualization Script for Benchmark Results

Generates speedup plots similar to the reference figures showing:
- Speedup vs. Process Count for different machines
- Comparison of traffic levels (Low, Medium, High)
- Comparison of accident scenarios (0 vs 1 accidents)
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Configuration
SCENARIOS = ["Almenara", "Rotterdam"]
MACHINES = ["Machine A", "Machine B", "Machine C", "HPC Node"]
TRAFFIC_LEVELS = ["Low", "Medium", "High"]

TRAFFIC_COLORS = {
    "Low": "#3498db",      # Blue
    "Medium": "#f39c12",   # Orange
    "High": "#2ecc71",     # Green
}

TRAFFIC_LABELS = {
    "Low": "Low Traffic",
    "Medium": "Medium Traffic",
    "High": "High Traffic",
}


def load_results(results_path: Path) -> pd.DataFrame:
    """Load benchmark results from CSV or Excel"""
    if results_path.suffix == ".xlsx":
        df = pd.read_excel(results_path)
    elif results_path.suffix == ".csv":
        df = pd.read_csv(results_path)
    elif results_path.suffix == ".json":
        df = pd.read_json(results_path)
    else:
        raise ValueError(f"Unsupported file format: {results_path.suffix}")
    
    return df


def plot_speedup_single_scenario(
    df: pd.DataFrame,
    scenario: str,
    output_dir: Path,
    accident_filter: int = None
):
    """
    Plot speedup for a single scenario with all machines.
    
    Args:
        df: DataFrame with benchmark results
        scenario: Scenario name (Almenara or Rotterdam)
        output_dir: Output directory for figures
        accident_filter: If specified, only plot this accident count
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter data
    sub = df[df["scenario"] == scenario]
    
    if accident_filter is not None:
        sub = sub[sub["accidents"] == accident_filter]
        title_suffix = f" - {accident_filter} accident{'s' if accident_filter != 1 else ''}"
        file_suffix = f"_{accident_filter}_accidents"
    else:
        title_suffix = ""
        file_suffix = ""
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=True)
    axes = axes.ravel()
    
    for idx, machine in enumerate(MACHINES):
        ax = axes[idx]
        mdf = sub[sub["machine"] == machine]
        
        if mdf.empty:
            ax.set_title(f"{machine} – No data")
            ax.set_axis_off()
            continue
        
        # Plot line for each traffic level
        for traffic in TRAFFIC_LEVELS:
            tdf = mdf[mdf["traffic"] == traffic].sort_values("processes")
            if tdf.empty:
                continue
            
            # Group by processes and average speedup
            grouped = tdf.groupby("processes")["speedup"].mean().reset_index()
            
            ax.plot(
                grouped["processes"],
                grouped["speedup"],
                marker="o",
                linestyle="-",
                color=TRAFFIC_COLORS.get(traffic, "black"),
                label=TRAFFIC_LABELS.get(traffic, traffic),
                linewidth=2,
                markersize=6
            )
        
        ax.set_title(f"{machine} – {scenario}")
        ax.set_xlabel("Processes")
        ax.set_ylabel("Speedup")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)
    
    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10)
    
    fig.suptitle(f"{scenario}{title_suffix}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save figure
    fig_path = output_dir / f"{scenario}{file_suffix}_speedup.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved: {fig_path}")


def plot_speedup_comparison(
    df: pd.DataFrame,
    scenario: str,
    output_dir: Path
):
    """
    Plot speedup comparison between 0 and 1 accident scenarios.
    
    Matches the style of Rotterdam_combined_0_1_accidents.png
    
    Args:
        df: DataFrame with benchmark results
        scenario: Scenario name
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter data for this scenario
    sub = df[(df["scenario"] == scenario) & (df["accidents"].isin([0, 1]))]
    
    if sub.empty:
        print(f"No data for scenario: {scenario}")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=True)
    axes = axes.ravel()
    
    for idx, machine in enumerate(MACHINES):
        ax = axes[idx]
        mdf = sub[sub["machine"] == machine]
        
        if mdf.empty:
            ax.set_title(f"{machine} – No data")
            ax.set_axis_off()
            continue
        
        # Plot each traffic level with solid (0 acc) and dashed (1 acc) lines
        for traffic in TRAFFIC_LEVELS:
            traffic_data = mdf[mdf["traffic"] == traffic]
            color = TRAFFIC_COLORS.get(traffic, "black")
            
            # 0 accidents - solid line
            data_0 = traffic_data[traffic_data["accidents"] == 0]
            if not data_0.empty:
                grouped = data_0.groupby("processes")["speedup"].mean().reset_index()
                grouped = grouped.sort_values("processes")
                
                ax.plot(
                    grouped["processes"],
                    grouped["speedup"],
                    marker="o",
                    linestyle="-",
                    color=color,
                    label=f"{TRAFFIC_LABELS.get(traffic, traffic)} - 0 accidents",
                    linewidth=2,
                    markersize=6
                )
            
            # 1 accident - dashed line
            data_1 = traffic_data[traffic_data["accidents"] == 1]
            if not data_1.empty:
                grouped = data_1.groupby("processes")["speedup"].mean().reset_index()
                grouped = grouped.sort_values("processes")
                
                ax.plot(
                    grouped["processes"],
                    grouped["speedup"],
                    marker="o",
                    linestyle="--",
                    color=color,
                    label=f"{TRAFFIC_LABELS.get(traffic, traffic)} - 1 accident",
                    linewidth=2,
                    markersize=6
                )
        
        ax.set_title(f"{machine} – {scenario}")
        ax.set_xlabel("Processes")
        ax.set_ylabel("Speedup")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0)
    
    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, 
        loc="upper center", 
        ncol=3, 
        fontsize=9,
        bbox_to_anchor=(0.5, 0.98)
    )
    
    fig.suptitle(f"{scenario} – Comparison 0 vs 1 accident", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    
    # Save figure
    fig_path = output_dir / f"{scenario}_combined_0_1_accidents.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    
    print(f"Figure saved: {fig_path}")


def plot_efficiency(
    df: pd.DataFrame,
    output_dir: Path
):
    """
    Plot parallel efficiency (speedup / num_processes) vs process count.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, scenario in enumerate(SCENARIOS):
        ax = axes[idx]
        sub = df[df["scenario"] == scenario]
        
        for machine in MACHINES:
            mdf = sub[sub["machine"] == machine]
            if mdf.empty:
                continue
            
            # Average across traffic levels and accidents
            grouped = mdf.groupby("processes").agg({
                "speedup": "mean"
            }).reset_index()
            
            grouped["efficiency"] = grouped["speedup"] / grouped["processes"]
            grouped = grouped.sort_values("processes")
            
            ax.plot(
                grouped["processes"],
                grouped["efficiency"],
                marker="o",
                linestyle="-",
                label=machine,
                linewidth=2
            )
        
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Ideal")
        ax.set_title(f"{scenario} - Parallel Efficiency")
        ax.set_xlabel("Processes")
        ax.set_ylabel("Efficiency (Speedup / Processes)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best")
        ax.set_ylim(0, 1.2)
    
    fig.tight_layout()
    
    fig_path = output_dir / "parallel_efficiency.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved: {fig_path}")


def generate_statistics_table(df: pd.DataFrame, output_dir: Path):
    """Generate summary statistics table"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary by scenario, machine, and process count
    summary = df.groupby(["scenario", "machine", "processes"]).agg({
        "speedup": ["mean", "std", "min", "max"],
        "total_time": "mean"
    }).round(3)
    
    # Save to CSV
    csv_path = output_dir / "speedup_summary.csv"
    summary.to_csv(csv_path)
    print(f"Summary saved: {csv_path}")
    
    # Create formatted table for each scenario
    for scenario in SCENARIOS:
        print(f"\n{'='*60}")
        print(f" {scenario} - Speedup Summary")
        print(f"{'='*60}")
        
        scenario_df = df[df["scenario"] == scenario]
        
        pivot = scenario_df.pivot_table(
            values="speedup",
            index=["machine", "traffic"],
            columns="processes",
            aggfunc="mean"
        ).round(2)
        
        print(pivot.to_string())
    
    return summary


def create_sample_data(output_dir: Path):
    """Create sample benchmark data for testing visualization"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = []
    
    # Based on the reference CSV structure
    scenarios = ["Almenara", "Rotterdam"]
    machines_config = {
        "Machine A": [1, 2, 4, 8],
        "Machine B": [1, 2, 4, 8, 16],
        "Machine C": [1, 2, 4, 6],
        "HPC Node": [1, 2, 4, 6, 8, 12, 16, 20, 24, 32]
    }
    traffic_levels = ["Low", "Medium", "High"]
    accidents = [0, 1]
    
    np.random.seed(42)
    
    for scenario in scenarios:
        for machine, process_counts in machines_config.items():
            for traffic in traffic_levels:
                for acc in accidents:
                    baseline = None
                    
                    for procs in process_counts:
                        if procs == 1:
                            speedup = 1.0
                        else:
                            # Realistic speedup model
                            # Amdahl's law with some parallel fraction
                            parallel_fraction = 0.7 + 0.1 * (traffic_levels.index(traffic) / 2)
                            ideal_speedup = 1 / ((1 - parallel_fraction) + parallel_fraction / procs)
                            
                            # Add some efficiency loss for high process counts
                            efficiency_loss = 1.0 - 0.02 * (procs - 1)
                            
                            # Accidents reduce parallelization benefit
                            accident_factor = 1.0 - 0.1 * acc
                            
                            speedup = ideal_speedup * efficiency_loss * accident_factor
                            
                            # Add some noise
                            speedup *= (1 + np.random.uniform(-0.05, 0.05))
                            
                            # Cap at realistic values
                            speedup = min(speedup, procs * 0.5)
                            speedup = max(speedup, 0.5)
                        
                        data.append({
                            "scenario": scenario,
                            "machine": machine,
                            "processes": procs,
                            "traffic": traffic,
                            "accidents": acc,
                            "speedup": round(speedup, 2),
                            "total_time": round(1000 / speedup, 2),
                            "repetition": 0
                        })
    
    df = pd.DataFrame(data)
    
    csv_path = output_dir / "sample_benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Sample data saved: {csv_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize parallel SUMO benchmark results"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input results file (CSV, Excel, or JSON)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for figures"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample benchmark data for testing"
    )
    
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["single", "combined", "efficiency", "all"],
        default=["all"],
        help="Which plots to generate"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "results" / "figures"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data if requested
    if args.create_sample:
        df = create_sample_data(output_dir.parent)
        print(f"\nSample data created. Use --input to analyze it.")
        args.input = str(output_dir.parent / "sample_benchmark_results.csv")
    
    # Load data
    if not args.input:
        # Try to find latest results
        results_dir = Path(__file__).parent.parent / "results"
        csv_files = list(results_dir.glob("*.csv"))
        
        if not csv_files:
            print("No results file found. Use --create-sample to generate test data.")
            sys.exit(1)
        
        # Use most recent
        args.input = str(max(csv_files, key=lambda p: p.stat().st_mtime))
        print(f"Using latest results: {args.input}")
    
    df = load_results(Path(args.input))
    print(f"Loaded {len(df)} benchmark results")
    
    # Determine which plots to generate
    plots = set(args.plots)
    if "all" in plots:
        plots = {"single", "combined", "efficiency"}
    
    # Generate plots
    if "single" in plots:
        for scenario in df["scenario"].unique():
            for acc in df["accidents"].unique():
                plot_speedup_single_scenario(df, scenario, output_dir, accident_filter=acc)
    
    if "combined" in plots:
        for scenario in df["scenario"].unique():
            plot_speedup_comparison(df, scenario, output_dir)
    
    if "efficiency" in plots:
        plot_efficiency(df, output_dir)
    
    # Generate statistics
    generate_statistics_table(df, output_dir)
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()


