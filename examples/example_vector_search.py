"""
Example: Vector Search Comparison

This example demonstrates how to run vector search benchmarks and analyze results.

DECISION RATIONALE:
- Complete benchmark workflow
- Result analysis and visualization
- Decision-making guidance
- Real-world use case demonstration
"""

import sys
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_vector_search_benchmark():
    """
    Run vector search benchmark and analyze results.
    """
    print("=" * 80)
    print("Vector Search Comparison Example")
    print("=" * 80)
    
    # Run benchmark
    print("\nStep 1: Running benchmark...")
    print("This may take a few minutes...")
    
    cmd = [
        sys.executable,
        str(project_root / "vector_search_comparison.py"),
        "--sizes", "1000", "5000",
        "--queries", "20",
        "--k", "5",
        "--visualize"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        print(e.stderr)
        return
    
    # Find latest benchmark CSV
    output_dir = project_root / "output"
    csv_files = sorted(
        output_dir.glob("vector_search_benchmark_*.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not csv_files:
        print("No benchmark CSV files found")
        return
    
    latest_csv = csv_files[0]
    print(f"\nBenchmark results saved to: {latest_csv}")
    
    # Analyze results
    print("\n" + "=" * 80)
    print("RESULT ANALYSIS")
    print("=" * 80)
    
    import pandas as pd
    df = pd.read_csv(latest_csv)
    
    # Average performance by method
    print("\nAverage Performance by Method:")
    print("-" * 80)
    
    avg_performance = df.groupby('method').agg({
        'query_time_mean': 'mean',
        'queries_per_second': 'mean',
        'index_size_mb': 'mean'
    }).round(2)
    
    print(avg_performance)
    
    # Feature comparison
    print("\nFeature Comparison:")
    print("-" * 80)
    
    methods = df['method'].unique()
    for method in methods:
        method_df = df[df['method'] == method].iloc[0]
        features = []
        
        if method_df.get('filter_support', False) is True or (hasattr(method_df.get('filter_support', False), '__bool__') and bool(method_df.get('filter_support', False))):
            features.append("Metadata Filtering")
        if method_df.get('persistence', False) is True or (hasattr(method_df.get('persistence', False), '__bool__') and bool(method_df.get('persistence', False))):
            features.append("Persistence")
        if pd.notna(method_df.get('ingestion_rate', None)):
            features.append("Batch Ingestion")
        
        print(f"{method}: {', '.join(features) if features else 'Basic search only'}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    fastest_query = df.loc[df['query_time_mean'].idxmin()]
    highest_qps = df.loc[df['queries_per_second'].idxmax()]
    
    print(f"Fastest Query: {fastest_query['method']} ({fastest_query['query_time_mean']:.2f}ms)")
    print(f"Highest QPS: {highest_qps['method']} ({highest_qps['queries_per_second']:.1f} QPS)")
    
    # Check for databases with features
    df_with_features = df[df['filter_support'].notna() & (df['filter_support'] == True)]
    if not df_with_features.empty:
        db_methods = df_with_features['method'].unique()
        print(f"\nMethods with Metadata Filtering: {', '.join(db_methods)}")
    
    # Decision guidance
    print("\n" + "=" * 80)
    print("DECISION GUIDANCE")
    print("=" * 80)
    print("Choose FAISS if:")
    print("  - Maximum query performance is critical")
    print("  - Dataset fits in memory")
    print("  - No need for metadata filtering")
    print("\nChoose Qdrant/Chroma if:")
    print("  - Need metadata filtering and persistence")
    print("  - Production deployment with scaling")
    print("  - Can accept slower query performance for features")
    print("\nChoose PostgreSQL+pgvector if:")
    print("  - Already using PostgreSQL")
    print("  - Need hybrid queries (SQL + vector search)")
    print("  - ACID transactions required")
    
    # Check for visualizations
    charts_dir = project_root / "output" / "benchmark_charts"
    if charts_dir.exists():
        chart_files = list(charts_dir.glob("*.png"))
        if chart_files:
            print(f"\nVisualization charts generated: {len(chart_files)} charts")
            print(f"Charts location: {charts_dir}")
            print("\nCharts available:")
            for chart in chart_files:
                print(f"  - {chart.name}")


if __name__ == "__main__":
    run_vector_search_benchmark()
