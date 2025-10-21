# Proxiss: Fast Vector Similarity Search

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Proxiss** is a high-performance C++ library with Python bindings, designed for fast vector similarity search in high-dimensional data. It provides efficient nearest-neighbor search capabilities for applications like semantic search, recommendation systems, and machine learning, currently optimized for Linux environments.

## Key Features

*   **High Performance:** Optimized C++ implementation with OpenMP parallelization for fast k-NN searches
*   **Multiple Distance Metrics:** Supports common distance functions:
    *   Euclidean (L2)
    *   Manhattan (L1) 
    *   Cosine Similarity
*   **Three Search Modes:**
    *   **ProxiFlat:** Vector-only indexing for pure similarity search
    *   **ProxiKNN:** Classification-focused search with label storage
    *   **ProxiPCA:** Dimensionality reduction combined with similarity search
*   **Batched Operations:** Efficient batch processing for multiple queries
*   **Python Integration**


## Why Proxiss?

Vector similarity search is fundamental to many modern applications, but traditional methods can be slow and resource-intensive. Proxiss addresses this by:

*   Providing optimized C++ implementations with parallel processing
*   Offering clean, simple APIs that hide implementation complexity  
*   Focusing on core functionality without unnecessary overhead
*   Supporting pure vector search, classification, and dimensionality reduction use cases

## Installation

Proxiss builds from source with automatic dependency management. or from PyPI https://pypi.org/project/proxiss/

### Prerequisites

*   Linux environment (Ubuntu, Debian, CentOS, etc.)
*   Python 3.10 or higher
*   CMake 3.16 or higher
*   UV package manager

**Note:** The build system automatically installs clang++, OpenMP and pybind11 if not found.

### Building from Source

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BiradarSiddhant02/Proxiss.git
    cd Proxiss
    ```

2.  **Install UV (if not already installed):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Create virtual environment and install:**
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install . -v
    ```

## Quick Start

### ProxiFlat: Vector Similarity Search

```python
from proxiss import ProxiFlat
import numpy as np

# Sample data
embeddings = np.array([
    [0.0, 0.0],
    [1.0, 1.0], 
    [2.0, 2.0],
    [3.0, 3.0]
], dtype=np.float32)

# Initialize ProxiFlat
px = ProxiFlat(k=2, num_threads=2, objective_function="l2")

# Index your vectors
px.index_data(embeddings)

# Query for nearest neighbors
query = np.array([1.5, 1.5], dtype=np.float32)
indices = px.find_indices(query)
print(f"Nearest neighbor indices: {indices}")

# Batch queries
queries = np.array([[0.5, 0.5], [2.5, 2.5]], dtype=np.float32)
batch_indices = px.find_indices_batched(queries)
print(f"Batch results: {batch_indices}")

# Save and load index
px.save_state("index.bin")
px_loaded = ProxiFlat(k=2, num_threads=2, objective_function="l2")
px_loaded.load_state("index.bin")
```

### ProxiKNN: Classification Search

```python
from proxiss import ProxiKNN
import numpy as np

# Sample data with labels
features = np.array([
    [0.0, 0.0], [1.0, 1.0],
    [5.0, 5.0], [6.0, 6.0]
], dtype=np.float32)
labels = np.array([0, 0, 1, 1], dtype=np.float32)

# Initialize and train
knn = ProxiKNN(n_neighbours=2, n_jobs=2, distance_function="l2")
knn.fit(features, labels)

# Predict
query = np.array([0.5, 0.5], dtype=np.float32)
prediction = knn.predict([query])
print(f"Predicted class: {prediction}")

# Save and load model
knn.save_state("model_dir")
knn_loaded = ProxiKNN(n_neighbours=2, n_jobs=2, distance_function="l2")
knn_loaded.load_state("model_dir")
```

### ProxiPCA: Dimensionality Reduction + Search

```python
from proxiss import ProxiPCA
import numpy as np

# High-dimensional sample data (e.g., 768-dimensional embeddings)
embeddings = np.random.randn(1000, 768).astype(np.float32)

# Initialize ProxiPCA with dimensionality reduction
# n_components as percentage: 0.065 means reduce to 6.5% of original dimensions
# For 768D data: 768 * 0.065 ≈ 50 dimensions
pca = ProxiPCA(k=5, num_threads=4, objective_function="l2", n_components=0.065)

# Fit PCA, transform data, and index in one step
pca.fit_transform_index(embeddings)

print(f"Original dimensions: {embeddings.shape[1]}")
print(f"Reduced dimensions: {pca.get_n_components()}")

# Query for nearest neighbors (query is automatically transformed)
query = np.random.randn(768).astype(np.float32)
indices = pca.find_indices(query)
print(f"Nearest neighbor indices: {indices}")

# Batch queries
queries = np.random.randn(10, 768).astype(np.float32)
batch_indices = pca.find_indices_batched(queries)
print(f"Batch results shape: {batch_indices.shape}")

# Insert new data (automatically transformed)
new_data = np.random.randn(100, 768).astype(np.float32)
pca.insert_data(new_data)

# Save and load (saves both PCA transformation and index)
pca.save_state("pca_index.bin")
pca_loaded = ProxiPCA(k=5, num_threads=4, objective_function="l2", n_components=0.065)
pca_loaded.load_state("pca_index.bin")
```

## Benchmarking

Proxiss includes benchmarking scripts to evaluate performance.

### 1. Generate Test Data

Create synthetic datasets for benchmarking:

```bash
python scripts/make_data.py --N 10000 --D 128 --X_path scripts/X.npy
```

### 2. Benchmark ProxiFlat

Test vector similarity search performance:

```bash
python scripts/bench_proxiss_flat.py --X_path scripts/X.npy -k 5 --threads 4 --objective l2
```

### 3. Benchmark ProxiKNN

Test classification performance:

```bash  
python scripts/bench_proxiss_knn.py --X_path scripts/X.npy -k 5 --threads 4 --objective l2
```

### 4. Benchmark ProxiPCA

Test dimensionality reduction + similarity search performance:

```bash
# -c flag specifies n_components as percentage (0.0-1.0)
# Example: -c 0.065 means reduce to 6.5% of original dimensions
python scripts/bench_proxiss_pca.py --X_path scripts/X.npy -k 5 --threads 4 --objective l2 -c 0.065
```

### 5. Compare with FAISS

Install FAISS and compare performance:

```bash
uv pip install faiss-cpu
python scripts/bench_faiss.py --X_path scripts/X.npy -k 5 --threads 4 --objective l2
```

### 6. Compare with scikit-learn

Install scikit-learn and compare KNN classification performance:

```bash
uv pip install scikit-learn
python scripts/bench_sklearn_knn.py --X_path scripts/X.npy -k 5 --threads 4 --objective l2
```

## Example Usage

### Interactive Inference

The `examples/inference.py` script demonstrates similarity search on real embeddings:

```bash
python examples/inference.py --embeddings examples/embeddings.npy --words examples/words.npy -k 5
```

This script loads pre-computed embeddings and allows interactive similarity search.

## Development

### Project Structure

*   **Core C++ Implementation:**
    *   `src/proxi_flat.cc`, `include/proxi_flat.h` - Vector similarity search
    *   `src/proxi_knn.cc`, `include/proxi_knn.h` - KNN classification
    *   `src/pca.cc`, `include/pca.h` - PCA dimensionality reduction
    *   `src/proxi_pca.cc`, `include/proxi_pca.h` - PCA + similarity search wrapper
    *   `src/priority_queue.cc`, `include/priority_queue.h` - Custom priority queue
    *   `include/distance.hpp` - Distance function implementations

*   **Python Bindings:**
    *   `bindings/proxi_flat_binding.cc` - ProxiFlat Python interface  
    *   `bindings/proxi_knn_binding.cc` - ProxiKNN Python interface
    *   `bindings/proxi_pca_binding.cc` - ProxiPCA Python interface
    *   `proxiss/ProxiFlat.py` - Python wrapper for ProxiFlat
    *   `proxiss/ProxiKNN.py` - Python wrapper for ProxiKNN
    *   `proxiss/ProxiPCA.py` - Python wrapper for ProxiPCA

*   **Build System:**
    *   `CMakeLists.txt` - C++ build configuration with automatic dependencies
    *   `pyproject.toml` - Python package configuration

### Running Tests

```bash
# Install test dependencies
uv pip install pytest

# Run all tests
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/test_proxi_flat.py -v
python -m pytest tests/test_proxi_knn.py -v
python -m pytest tests/test_proxi_pca.py -v
```

### Building for Development

```bash
# Set up development environment
uv venv
source .venv/bin/activate

# Install development dependencies
uv pip install -r requirements.txt

# Reinstall after C++ changes
uv pip install -e . --force-reinstall --no-deps
```

## API Reference

### ProxiFlat Methods
*   `__init__(k, num_threads, objective_function)` - Initialize index
*   `index_data(embeddings)` - Index vector data
*   `find_indices(query)` - Find nearest neighbor indices
*   `find_indices_batched(queries)` - Batch query processing
*   `save_state(filepath)` - Save index to file
*   `load_state(filepath)` - Load index from file

### ProxiKNN Methods  
*   `__init__(n_neighbours, n_jobs, distance_function)` - Initialize classifier
*   `fit(features, labels)` - Train on labeled data
*   `predict(features)` - Predict class labels
*   `save_state(directory)` - Save model to directory
*   `load_state(directory)` - Load model from directory

### ProxiPCA Methods
*   `__init__(k, num_threads, objective_function, n_components)` - Initialize with PCA reduction
*   `fit_transform_index(embeddings)` - Fit PCA, transform data, and index
*   `find_indices(query)` - Find nearest neighbors (query auto-transformed)
*   `find_indices_batched(queries)` - Batch query processing
*   `insert_data(embeddings)` - Insert new data (auto-transformed)
*   `get_n_components()` - Get actual number of PCA components used
*   `get_components()` - Get PCA component vectors
*   `get_mean()` - Get PCA mean vector
*   `get_explained_variance()` - Get variance explained by each component
*   `save_state(filepath)` - Save PCA transformation and index
*   `load_state(filepath)` - Load PCA transformation and index

## License

Proxiss is licensed under the Apache License, Version 2.0. See [LICENSE.txt](LICENSE.txt) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

**Proxiss - Fast Vector Similarity Search**
