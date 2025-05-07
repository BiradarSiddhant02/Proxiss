# Proxi: Blazing-Fast Nearest Neighbor Search 🚀

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Proxi** is a high-performance C++ library with Python bindings, designed to accelerate nearest-neighbor search for high-dimensional data. Whether you're working on semantic search, recommendation systems, anomaly detection, or any application requiring fast similarity searches, Proxi offers an efficient and easy-to-use solution.

## 🌟 Key Features

*   **Blazing Fast:** Leverages C++ for core computations and OpenMP for parallel processing to deliver high-speed k-NN searches.
*   **Multiple Distance Metrics:** Supports common distance functions:
    *   Euclidean (L2)
    *   Manhattan (L1)
    *   Cosine Similarity (normalized dot product)
*   **Python-Friendly API:** Easy-to-use Python bindings powered by pybind11, making integration into your Python projects seamless.
*   **Batched Operations:** Efficiently process multiple queries at once with batched search methods.
*   **Simple Indexing:** Straightforward data indexing process.
*   **Lightweight:** Minimal dependencies, focused on delivering core k-NN functionality efficiently.

## 🤔 Why Proxi?

Searching for similar items in large, high-dimensional datasets is a common challenge. Traditional methods can be slow and computationally expensive. Proxi tackles this by:

*   Providing optimized C++ implementations of search algorithms.
*   Utilizing parallel processing to speed up computations on multi-core processors.
*   Offering a simple API that doesn't require deep expertise in low-level programming.

## 🛠️ Installation

You can build and install Proxi from source.

### Prerequisites

*   A C++ compiler supporting C++20 (e.g., GCC, Clang, MSVC)
*   CMake (version 3.15 or higher)
*   Python (version 3.8 or higher)
*   Setuptools and Pybind11 (will be handled by the build process if `scikit-build-core` is used)

### Building from Source

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/BiradarSiddhant02/Proxi.git
    cd Proxi
    ```

2.  **Install using pip:**
    This is the recommended way as it handles dependencies and the build process.
    ```bash
    pip install .
    ```
    Alternatively, for development, you can use:
    ```bash
    pip install -e .
    ```

3.  **Manual build (alternative):**
    ```bash
    python setup.py build
    python setup.py install
    ```

## 🚀 Quick Start

Here's a simple example of how to use Proxi in Python:

```python
import proxi
import numpy as np

# 1. Sample data
embeddings = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.0],
    [3.0, 3.0]
], dtype=np.float32)
doc_ids = ["doc_a", "doc_b", "doc_c", "doc_d"]

# 2. Initialize Proxi
# Parameters: k (number of neighbors), num_threads, objective_function ("l2", "l1", or "cos")
px = proxi.Proxi(k=2, num_threads=2, objective_function="l2")

# 3. Index your data
px.index_data(embeddings.tolist(), doc_ids) # Proxi expects lists of lists for embeddings

# 4. Prepare a query vector
query_vector = np.array([1.5, 1.5], dtype=np.float32)

# 5. Find nearest neighbor indices
indices = px.find_indices(query_vector.tolist())
print(f"Indices of nearest neighbors: {indices}")
# Example output: Indices of nearest neighbors: [1, 2] (or [2, 1] depending on exact distances)

# 6. Find nearest neighbor documents
docs = px.find_docs(query_vector.tolist())
print(f"Nearest documents: {docs}")
# Example output: Nearest documents: ['doc_b', 'doc_c']

# 7. Batched queries (for multiple queries at once)
query_batch = np.array([
    [0.5, 0.5],
    [2.5, 2.5]
], dtype=np.float32)

batch_indices = px.find_indices_batched(query_batch)
print(f"Batch indices: {batch_indices}")
# Example output: Batch indices: [[0, 1], [2, 3]]

batch_docs = px.find_docs_batched(query_batch)
print(f"Batch documents: {batch_docs}")
# Example output: Batch documents: [['doc_a', 'doc_b'], ['doc_c', 'doc_d']]
```

## ⏱️ Benchmarking

Proxi includes scripts to benchmark its performance against FAISS (a popular similarity search library) and to generate sample data.

### 1. Generate Mock Data

Use the `make_data.py` script to create synthetic datasets for benchmarking:

```bash
python scripts/make_data.py --N 10000 --D 128 --X_path X_data.npy --docs_path docs_data.npy
```

This will generate `X_data.npy` (feature vectors) and `docs_data.npy` (document identifiers).

### 2. Run Proxi Benchmark

Use `bench_proxi.py` to test Proxi's performance:

```bash
python scripts/bench_proxi.py --X_path X_data.npy --docs_path docs_data.npy -k 5 --threads 4 --objective l2
```
Adjust `-k` (number of neighbors), `--threads`, and `--objective` as needed.

### 3. Run FAISS Benchmark (for comparison)

If you have FAISS installed (`pip install faiss-cpu` or `faiss-gpu`), you can run a comparative benchmark:

```bash
python scripts/bench_faiss.py --X_path X_data.npy --docs_path docs_data.npy -k 5 --threads 4 --objective l2
```

## 💡 Interactive Inference Example

Proxi includes an interactive script `examples/inference.py` that allows you to perform similarity searches on your own data and compare results with FAISS.

### 1. Download Embeddings and Corresponding Text

To use the inference script, you first need a dataset of embeddings and the corresponding text/words they represent.

For a quick demonstration, you can download pre-computed embeddings and words:
*   **Embeddings and words:** [https://www.kaggle.com/datasets/siddhantbiradar/proxi-live-inference-dataset](https://www.kaggle.com/datasets/siddhantbiradar/proxi-live-inference-dataset)

Download the zip file and unzip it in a directory of your choice. The variety of words is relatively small but enough for a demo

### 2. Run the Inference Script

Navigate to the `Proxi` directory and run the script from your terminal:

```bash
python examples/inference.py --embeddings /path/to/your/embeddings.npy --words /path/to/your/words.npy -k 5
```

**Arguments:**

*   `--embeddings`: Path to the `.npy` file containing your numerical embeddings.
*   `--words`: Path to the `.npy` file containing the corresponding text entries (words, sentences, or document IDs).
*   `-k`: The number of nearest neighbors to retrieve for each query.

### 3. Interactive Search

Once the script loads the data and builds the Proxi and FAISS indexes, it will prompt you to enter a word or phrase:

```
Loading data...
Loaded 384000 embeddings with dimension 384

Building Proxi index...
Building FAISS index...
Loading sentence transformer model...

==================================================
Enter a word or phrase (or 'quit' to exit): your search query
```

Type your query and press Enter. The script will then display a table comparing the top-k results from Proxi and FAISS.

```
Similarity search results for: 'your search query'
+--------+-----------------+-----------------+
|   Rank | Proxi Results   | FAISS Results   |
+========+=================+=================+
|      1 | result_proxi_1  | result_faiss_1  |
|      2 | result_proxi_2  | result_faiss_2  |
|    ... | ...             | ...             |
+--------+-----------------+-----------------+
```

Enter 'quit' to exit the script.

This example provides a hands-on way to see Proxi in action and compare its output directly with another popular library.

## 🏗️ Building and Development

*   The core logic is in C++ (`src/proxi.cc`, `include/proxi.h`, `include/distance.hpp`).
*   Python bindings are defined in `bindings/proxi_binding.cc`.
*   `CMakeLists.txt` manages the C++ build process.
*   `setup.py` (using `scikit-build-core`) handles the Python package build and C++ compilation.
*   Tests are located in `tests/test_proxi.py`. Run them using `unittest`:
    ```bash
    python -m unittest tests/test_proxi.py
    ```

## 📜 License

Proxi is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE.txt) file for details (You might need to create this file if it doesn't exist and copy the Apache 2.0 text into it, or refer to the license text in `project.toml`).

## 🙌 Contributing

Contributions are welcome! If you have suggestions, bug reports, or want to contribute code, please feel free to open an issue or submit a pull request.

---

Happy Searching with Proxi!
