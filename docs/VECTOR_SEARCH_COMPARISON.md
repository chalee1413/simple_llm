# Vector Search Comparison: FAISS vs Qdrant vs Chroma vs PostgreSQL+pgvector

## Overview

Comprehensive performance comparison between FAISS (vector search library), vector databases (Qdrant, Chroma), and RDBMS with vector extensions (PostgreSQL + pgvector) across multiple dimensions.

## Comparison Script

Run `vector_search_comparison.py` to generate benchmarks:

```bash
# Start PostgreSQL container (required for PostgreSQL+pgvector benchmark)
docker-compose up -d

# Run benchmark
python vector_search_comparison.py --sizes 1000 5000 10000 --queries 50 --k 10

# Stop PostgreSQL container (optional)
docker-compose down
```

### PostgreSQL Setup

PostgreSQL + pgvector benchmark requires Docker:

1. Start PostgreSQL container:
   ```bash
   docker-compose up -d
   ```

2. Wait for PostgreSQL to be ready (health check completes)

3. Run benchmark (PostgreSQL will be included automatically if available)

4. Stop container (optional):
   ```bash
   docker-compose down
   ```

## Benchmarked Dimensions

### Performance Metrics

1. **Query Speed**: Latency (ms) and throughput (QPS)
2. **Ingestion Performance**: Insertion rate (vectors/sec)
3. **Index Building Time**: Time to build index
4. **Memory Usage**: Index size (MB)
5. **Scalability**: Performance across dataset sizes (1K, 5K, 10K, 100K vectors)

### Feature Comparison

1. **Metadata Filtering**: Support for filtering by metadata
2. **Persistence**: Data persistence across sessions
3. **Batch Ingestion**: Batch insert capabilities
4. **Accuracy**: Recall for approximate methods

## Comparison Results

Benchmark results vary by dataset size and hardware. Run the benchmark script to see actual performance metrics for your environment:

```bash
python vector_search_comparison.py --sizes 1000 5000 10000 --queries 50 --k 10 --visualize
```

The benchmark measures:
- Query performance (latency, throughput)
- Ingestion rate (vectors per second)
- Scalability (performance vs dataset size)
- Feature support (metadata filtering, persistence, ACID transactions)

### Feature Support

| Feature | FAISS | NumPy | Scikit-learn | Qdrant | Chroma | PostgreSQL+pgvector |
|---------|-------|-------|--------------|--------|--------|-------------------|
| Query Speed | Fast | Fastest (small) | Slow | Medium | Medium | Medium-Slow |
| Metadata Filtering | No | No | No | Yes | Yes | Yes (SQL WHERE) |
| Persistence | No | No | No | Yes | Yes | Yes (ACID) |
| Batch Ingestion | No | No | No | Yes | Yes | Yes (SQL INSERT) |
| Hybrid Queries | No | No | No | No | No | Yes (SQL JOIN) |
| ACID Transactions | No | No | No | No | No | Yes |
| Scalability | Excellent | Poor | Poor | Good | Good | Good |

## Recommendations

### Use FAISS when:

- Need maximum query performance
- Dataset fits in memory
- No need for metadata filtering
- Simple persistence is sufficient (manual save/load)

### Use Qdrant when:

- Need metadata filtering
- Require production-ready persistence
- Need batch ingestion
- Can accept slower query performance for database features

### Use Chroma when:

- Need easy Python integration
- Require embedded database (no separate server)
- Need metadata filtering and persistence
- Prioritize ease of use over maximum performance

### Use PostgreSQL + pgvector when:

- Already using PostgreSQL
- Need hybrid queries (SQL JOIN + vector search)
- Require ACID transactions
- Need SQL-based metadata filtering (WHERE clauses)
- Want to leverage existing PostgreSQL infrastructure

### Use NumPy when:

- Very small datasets (<1K vectors)
- Simple prototyping
- No database features needed

## Benchmark Methodology

### Test Configuration

- **Dimension**: 384 (all-MiniLM-L6-v2 embeddings)
- **Dataset Sizes**: 1,000, 5,000, 10,000 vectors
- **Queries**: 20-50 query vectors
- **K**: 5-10 nearest neighbors
- **Distance Metric**: Cosine similarity

### Test Environment

- Python 3.11
- FAISS-CPU (no GPU)
- Qdrant (local persistent mode)
- Chroma (persistent client)
- All benchmarks run on same hardware

### Metrics Collected

- Query latency (mean, std, min, max)
- Queries per second (QPS)
- Index building time
- Ingestion time and rate
- Index size (memory/disk)
- Filter support and performance
- Persistence support

## Implementation Notes

### FAISS

- In-memory index only
- No built-in persistence (requires manual save/load)
- No metadata filtering
- Excellent query performance
- Scales well with dataset size

### Qdrant

- Persistent storage
- Metadata filtering support
- Batch ingestion
- Rust implementation (good performance)
- Requires separate database instance (or local file)

### Chroma

- Embedded database (no separate server)
- Python-native implementation
- Metadata filtering support
- Persistent storage
- Requires documents field (more overhead)

## References

- FAISS: Facebook AI Similarity Search (2024)
- Qdrant: Vector Database Documentation (2024-2025)
- Chroma: Vector Database Documentation (2024-2025)
- Benchmarks: See `output/vector_search_benchmark_*.csv` for detailed results

