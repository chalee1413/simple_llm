"""
Visualization script for vector search benchmark results.

DECISION RATIONALE:
- Generate static charts (PNG/PDF) using matplotlib for portfolio presentation
- Create comprehensive visualizations of benchmark results
- Support multiple chart types: line charts, bar charts, scatter plots
- Generate summary dashboard combining multiple metrics

References:
- Matplotlib: Static visualization library (2024)
- Seaborn: Statistical visualization (optional, for better styling)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import seaborn for better styling
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("Seaborn not available. Using matplotlib defaults.")


class BenchmarkVisualizer:
    """
    Visualize vector search benchmark results.
    
    Generates static charts (PNG/PDF) for portfolio presentation.
    """
    
    def __init__(self, output_dir: Path = Path("output/benchmark_charts")):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"BenchmarkVisualizer initialized. Charts will be saved to: {self.output_dir}")
    
    def load_benchmark_data(self, csv_file: Path) -> pd.DataFrame:
        """
        Load benchmark data from CSV file.
        
        Args:
            csv_file: Path to benchmark CSV file
        
        Returns:
            DataFrame with benchmark results
        """
        if not csv_file.exists():
            raise FileNotFoundError(f"Benchmark file not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded benchmark data: {len(df)} rows from {csv_file}")
        return df
    
    def plot_query_performance(self, df: pd.DataFrame, format: str = "png") -> Path:
        """
        Plot query performance comparison (latency vs dataset size).
        
        Args:
            df: Benchmark DataFrame
            format: Output format (png or pdf)
        
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by method and dataset size
        for method in df['method'].unique():
            method_df = df[df['method'] == method].sort_values('n_vectors')
            ax.plot(
                method_df['n_vectors'],
                method_df['query_time_mean'],
                marker='o',
                linewidth=2,
                markersize=8,
                label=method
            )
            # Add error bars
            ax.fill_between(
                method_df['n_vectors'],
                method_df['query_time_mean'] - method_df['query_time_std'],
                method_df['query_time_mean'] + method_df['query_time_std'],
                alpha=0.2
            )
        
        ax.set_xlabel('Dataset Size (vectors)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Query Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Query Performance: Latency vs Dataset Size', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"query_performance.{format if format != 'jpeg' else 'jpg'}"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format=format if format != 'jpeg' else 'jpg')
        plt.close()
        
        logger.info(f"Query performance chart saved: {output_file}")
        return output_file
    
    def plot_queries_per_second(self, df: pd.DataFrame, format: str = "png") -> Path:
        """
        Plot queries per second comparison (bar chart).
        
        Args:
            df: Benchmark DataFrame
            format: Output format (png or pdf)
        
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Average QPS across all dataset sizes
        qps_by_method = df.groupby('method')['queries_per_second'].mean().sort_values(ascending=False)
        
        bars = ax.bar(
            range(len(qps_by_method)),
            qps_by_method.values,
            color=plt.cm.viridis(np.linspace(0, 1, len(qps_by_method)))
        )
        
        # Add value labels on bars
        for i, (method, value) in enumerate(qps_by_method.items()):
            ax.text(i, value, f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xticks(range(len(qps_by_method)))
        ax.set_xticklabels(qps_by_method.index, rotation=45, ha='right')
        ax.set_ylabel('Queries Per Second (QPS)', fontsize=12, fontweight='bold')
        ax.set_title('Throughput Comparison: Queries Per Second', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"queries_per_second.{format if format != 'jpeg' else 'jpg'}"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format=format if format != 'jpeg' else 'jpg')
        plt.close()
        
        logger.info(f"Queries per second chart saved: {output_file}")
        return output_file
    
    def plot_ingestion_rate(self, df: pd.DataFrame, format: str = "png") -> Path:
        """
        Plot ingestion rate comparison (bar chart).
        
        Args:
            df: Benchmark DataFrame
            format: Output format (png or pdf)
        
        Returns:
            Path to saved chart
        """
        # Filter to only methods with ingestion_rate
        df_with_ingestion = df[df['ingestion_rate'].notna()].copy()
        
        if df_with_ingestion.empty:
            logger.warning("No ingestion rate data available. Skipping ingestion rate chart.")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Average ingestion rate across all dataset sizes
        ingestion_by_method = df_with_ingestion.groupby('method')['ingestion_rate'].mean().sort_values(ascending=False)
        
        bars = ax.bar(
            range(len(ingestion_by_method)),
            ingestion_by_method.values,
            color=plt.cm.plasma(np.linspace(0, 1, len(ingestion_by_method)))
        )
        
        # Add value labels on bars
        for i, (method, value) in enumerate(ingestion_by_method.items()):
            ax.text(i, value, f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xticks(range(len(ingestion_by_method)))
        ax.set_xticklabels(ingestion_by_method.index, rotation=45, ha='right')
        ax.set_ylabel('Ingestion Rate (vectors/sec)', fontsize=12, fontweight='bold')
        ax.set_title('Ingestion Performance Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"ingestion_rate.{format if format != 'jpeg' else 'jpg'}"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format=format if format != 'jpeg' else 'jpg')
        plt.close()
        
        logger.info(f"Ingestion rate chart saved: {output_file}")
        return output_file
    
    def plot_scalability_analysis(self, df: pd.DataFrame, format: str = "png") -> Path:
        """
        Plot scalability analysis (query time increase vs dataset size).
        
        Args:
            df: Benchmark DataFrame
            format: Output format (png or pdf)
        
        Returns:
            Path to saved chart
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate query time increase for each method
        for method in df['method'].unique():
            method_df = df[df['method'] == method].sort_values('n_vectors')
            if len(method_df) < 2:
                continue
            
            # Calculate time increase factor
            small_time = method_df.iloc[0]['query_time_mean']
            time_increase = method_df['query_time_mean'] / small_time
            
            ax.plot(
                method_df['n_vectors'],
                time_increase,
                marker='s',
                linewidth=2,
                markersize=8,
                label=method
            )
        
        ax.set_xlabel('Dataset Size (vectors)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Query Time Increase (x)', fontsize=12, fontweight='bold')
        ax.set_title('Scalability Analysis: Query Time Increase vs Dataset Size', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline (1x)')
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"scalability_analysis.{format if format != 'jpeg' else 'jpg'}"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format=format if format != 'jpeg' else 'jpg')
        plt.close()
        
        logger.info(f"Scalability analysis chart saved: {output_file}")
        return output_file
    
    def plot_feature_comparison(self, df: pd.DataFrame, format: str = "png") -> Path:
        """
        Plot feature comparison matrix.
        
        Args:
            df: Benchmark DataFrame
            format: Output format (png or pdf)
        
        Returns:
            Path to saved chart
        """
        # Get unique methods
        methods = df['method'].unique()
        
        # Feature matrix
        features = {
            'Metadata Filtering': 'filter_support',
            'Persistence': 'persistence',
            'Batch Ingestion': 'ingestion_rate'
        }
        
        # Create feature matrix
        feature_matrix = []
        for method in methods:
            method_df = df[df['method'] == method].iloc[0]
            row = []
            for feature_name, feature_col in features.items():
                if feature_col == 'ingestion_rate':
                    has_feature = pd.notna(method_df.get(feature_col, None))
                else:
                    feature_val = method_df.get(feature_col, False)
                    has_feature = (feature_val is True) or (hasattr(feature_val, '__bool__') and bool(feature_val))
                row.append(1 if has_feature else 0)
            feature_matrix.append(row)
        
        feature_matrix = np.array(feature_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(feature_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(features)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(list(features.keys()))
        ax.set_yticklabels(methods)
        
        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(features)):
                text = ax.text(j, i, 'Yes' if feature_matrix[i, j] == 1 else 'No',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Feature Comparison Matrix', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        output_file = self.output_dir / f"feature_comparison.{format if format != 'jpeg' else 'jpg'}"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format=format if format != 'jpeg' else 'jpg')
        plt.close()
        
        logger.info(f"Feature comparison chart saved: {output_file}")
        return output_file
    
    def plot_summary_dashboard(self, df: pd.DataFrame, format: str = "png") -> Path:
        """
        Generate summary dashboard with multiple subplots.
        
        Args:
            df: Benchmark DataFrame
            format: Output format (png or pdf)
        
        Returns:
            Path to saved chart
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Query Performance (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        for method in df['method'].unique():
            method_df = df[df['method'] == method].sort_values('n_vectors')
            ax1.plot(method_df['n_vectors'], method_df['query_time_mean'], marker='o', label=method)
        ax1.set_xlabel('Dataset Size')
        ax1.set_ylabel('Query Latency (ms)')
        ax1.set_title('Query Performance')
        ax1.legend(fontsize=8)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 2. Queries Per Second (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        qps_by_method = df.groupby('method')['queries_per_second'].mean().sort_values(ascending=False)
        ax2.bar(range(len(qps_by_method)), qps_by_method.values, color=plt.cm.viridis(np.linspace(0, 1, len(qps_by_method))))
        ax2.set_xticks(range(len(qps_by_method)))
        ax2.set_xticklabels(qps_by_method.index, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('QPS')
        ax2.set_title('Throughput (QPS)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Scalability (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        for method in df['method'].unique():
            method_df = df[df['method'] == method].sort_values('n_vectors')
            if len(method_df) < 2:
                continue
            small_time = method_df.iloc[0]['query_time_mean']
            time_increase = method_df['query_time_mean'] / small_time
            ax3.plot(method_df['n_vectors'], time_increase, marker='s', label=method)
        ax3.set_xlabel('Dataset Size')
        ax3.set_ylabel('Time Increase (x)')
        ax3.set_title('Scalability Analysis')
        ax3.legend(fontsize=8)
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. Index Size (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        size_by_method = df.groupby('method')['index_size_mb'].mean().sort_values(ascending=False)
        ax4.bar(range(len(size_by_method)), size_by_method.values, color=plt.cm.plasma(np.linspace(0, 1, len(size_by_method))))
        ax4.set_xticks(range(len(size_by_method)))
        ax4.set_xticklabels(size_by_method.index, rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Index Size (MB)')
        ax4.set_title('Memory Usage')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Ingestion Rate (bottom left, if available)
        ax5 = fig.add_subplot(gs[2, 0])
        df_with_ingestion = df[df['ingestion_rate'].notna()].copy()
        if not df_with_ingestion.empty:
            ingestion_by_method = df_with_ingestion.groupby('method')['ingestion_rate'].mean().sort_values(ascending=False)
            ax5.bar(range(len(ingestion_by_method)), ingestion_by_method.values, color=plt.cm.coolwarm(np.linspace(0, 1, len(ingestion_by_method))))
            ax5.set_xticks(range(len(ingestion_by_method)))
            ax5.set_xticklabels(ingestion_by_method.index, rotation=45, ha='right', fontsize=8)
            ax5.set_ylabel('Vectors/sec')
            ax5.set_title('Ingestion Rate')
            ax5.grid(True, alpha=0.3, axis='y')
        else:
            ax5.text(0.5, 0.5, 'No ingestion data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Ingestion Rate')
        
        # 6. Feature Comparison (bottom right)
        ax6 = fig.add_subplot(gs[2, 1])
        methods = df['method'].unique()
        features = ['Filter', 'Persistence', 'Ingestion']
        feature_matrix = []
        for method in methods:
            method_df = df[df['method'] == method].iloc[0]
            row = [
                1 if (method_df.get('filter_support', False) is True or (hasattr(method_df.get('filter_support', False), '__bool__') and bool(method_df.get('filter_support', False)))) else 0,
                1 if (method_df.get('persistence', False) is True or (hasattr(method_df.get('persistence', False), '__bool__') and bool(method_df.get('persistence', False)))) else 0,
                1 if pd.notna(method_df.get('ingestion_rate', None)) else 0
            ]
            feature_matrix.append(row)
        
        feature_matrix = np.array(feature_matrix)
        im = ax6.imshow(feature_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax6.set_xticks(np.arange(len(features)))
        ax6.set_yticks(np.arange(len(methods)))
        ax6.set_xticklabels(features, fontsize=8)
        ax6.set_yticklabels(methods, fontsize=8)
        for i in range(len(methods)):
            for j in range(len(features)):
                ax6.text(j, i, 'Yes' if feature_matrix[i, j] == 1 else 'No',
                        ha="center", va="center", color="black", fontweight='bold', fontsize=8)
        ax6.set_title('Features')
        
        fig.suptitle('Vector Search Benchmark Summary Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        output_file = self.output_dir / f"summary_dashboard.{format if format != 'jpeg' else 'jpg'}"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format=format if format != 'jpeg' else 'jpg')
        plt.close()
        
        logger.info(f"Summary dashboard saved: {output_file}")
        return output_file
    
    def plot_resource_usage(self, df: pd.DataFrame, format: str = "png") -> Optional[Path]:
        """
        Plot CPU and disk usage comparison.
        
        Args:
            df: Benchmark DataFrame
            format: Output format (png or pdf)
        
        Returns:
            Path to saved chart or None if no resource data available
        """
        # Check if resource metrics are available
        if 'cpu_usage_mean' not in df.columns or df['cpu_usage_mean'].isna().all():
            logger.warning("No CPU usage data available. Skipping resource usage chart.")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # CPU Usage
        ax1 = axes[0, 0]
        for method in df['method'].unique():
            method_df = df[df['method'] == method].sort_values('n_vectors')
            if 'cpu_usage_mean' in method_df.columns:
                ax1.plot(method_df['n_vectors'], method_df['cpu_usage_mean'], 
                        marker='o', linewidth=2, markersize=6, label=method)
        ax1.set_xlabel('Dataset Size (vectors)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('CPU Usage (%)', fontsize=11, fontweight='bold')
        ax1.set_title('CPU Usage vs Dataset Size', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Memory Usage
        ax2 = axes[0, 1]
        for method in df['method'].unique():
            method_df = df[df['method'] == method].sort_values('n_vectors')
            if 'memory_usage_mean_mb' in method_df.columns:
                ax2.plot(method_df['n_vectors'], method_df['memory_usage_mean_mb'], 
                        marker='s', linewidth=2, markersize=6, label=method)
        ax2.set_xlabel('Dataset Size (vectors)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Memory Usage (MB)', fontsize=11, fontweight='bold')
        ax2.set_title('Memory Usage vs Dataset Size', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # Disk I/O Read
        ax3 = axes[1, 0]
        for method in df['method'].unique():
            method_df = df[df['method'] == method].sort_values('n_vectors')
            if 'disk_read_mb' in method_df.columns:
                ax3.bar(range(len(method_df)), method_df['disk_read_mb'].fillna(0), 
                       label=method, alpha=0.7)
        ax3.set_xlabel('Dataset Size Index', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Disk Read (MB)', fontsize=11, fontweight='bold')
        ax3.set_title('Disk Read I/O', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Disk I/O Write
        ax4 = axes[1, 1]
        for method in df['method'].unique():
            method_df = df[df['method'] == method].sort_values('n_vectors')
            if 'disk_write_mb' in method_df.columns:
                ax4.bar(range(len(method_df)), method_df['disk_write_mb'].fillna(0), 
                       label=method, alpha=0.7)
        ax4.set_xlabel('Dataset Size Index', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Disk Write (MB)', fontsize=11, fontweight='bold')
        ax4.set_title('Disk Write I/O', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Resource Usage Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = self.output_dir / f"resource_usage.{format if format != 'jpeg' else 'jpg'}"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format=format if format != 'jpeg' else 'jpg')
        plt.close()
        
        logger.info(f"Resource usage chart saved: {output_file}")
        return output_file
    
    def generate_all_charts(self, csv_file: Path, formats: List[str] = ["png"]) -> Dict[str, Path]:
        """
        Generate all charts from benchmark CSV.
        
        Args:
            csv_file: Path to benchmark CSV file
            formats: List of output formats (png, pdf, jpeg, or "both" for png and pdf)
        
        Returns:
            Dictionary mapping chart names to file paths
        """
        df = self.load_benchmark_data(csv_file)
        
        charts = {}
        
        for format in formats:
            logger.info(f"Generating charts in {format} format...")
            
            charts[f'query_performance_{format}'] = self.plot_query_performance(df, format)
            charts[f'queries_per_second_{format}'] = self.plot_queries_per_second(df, format)
            
            ingestion_chart = self.plot_ingestion_rate(df, format)
            if ingestion_chart:
                charts[f'ingestion_rate_{format}'] = ingestion_chart
            
            charts[f'scalability_analysis_{format}'] = self.plot_scalability_analysis(df, format)
            charts[f'feature_comparison_{format}'] = self.plot_feature_comparison(df, format)
            charts[f'summary_dashboard_{format}'] = self.plot_summary_dashboard(df, format)
            
            resource_chart = self.plot_resource_usage(df, format)
            if resource_chart:
                charts[f'resource_usage_{format}'] = resource_chart
        
        logger.info(f"Generated {len(charts)} charts")
        return charts


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize Vector Search Benchmark Results")
    parser.add_argument("--input", type=str, help="Input benchmark CSV file")
    parser.add_argument("--output", type=str, default="output/benchmark_charts", help="Output directory for charts")
    parser.add_argument("--format", type=str, choices=["png", "pdf", "jpeg", "both"], default="png",
                       help="Output format (default: png). Use 'jpeg' for README figures.")
    
    args = parser.parse_args()
    
    # Find latest benchmark file if not specified
    if args.input:
        input_file = Path(args.input)
    else:
        # Find latest benchmark CSV
        output_dir = Path("output")
        csv_files = sorted(output_dir.glob("vector_search_benchmark_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not csv_files:
            logger.error("No benchmark CSV files found. Run vector_search_comparison.py first.")
            return
        input_file = csv_files[0]
        logger.info(f"Using latest benchmark file: {input_file}")
    
    # Determine formats
    formats = ["png", "pdf"] if args.format == "both" else [args.format]
    
    # Create visualizer
    visualizer = BenchmarkVisualizer(output_dir=Path(args.output))
    
    # Generate all charts
    charts = visualizer.generate_all_charts(input_file, formats=formats)
    
    print(f"\nGenerated {len(charts)} charts:")
    for chart_name, chart_path in charts.items():
        print(f"  - {chart_name}: {chart_path}")


if __name__ == "__main__":
    main()
