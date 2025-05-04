import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from time import perf_counter_ns

try:
    import faiss
except ImportError:
    raise SystemExit("Could not import `faiss`. Install it via `pip install faiss-cpu` or `conda install -c pytorch faiss-cpu`.")

def get_unit(ns: int):
    if ns >= 1e9:
        return ns / 1e9, "s"
    if ns >= 1e6:
        return ns / 1e6, "ms"
    if ns >= 1e3:
        return ns / 1e3, "us"
    return ns, "ns"

def main():
    parser = argparse.ArgumentParser(description="Benchmark FAISS on a dataset")
    parser.add_argument("--X_path", required=True, help="Path to features (.npy)")
    parser.add_argument("--docs_path", required=True, help="Path to documents (.npy of strings)")
    parser.add_argument("-k", type=int, default=5, help="Number of neighbors")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data to hold out for queries")
    parser.add_argument("--objective", choices=["l2", "l1", "cos"], default="l2", help="Distance function")
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.threads)

    # 1) Load data
    t0 = perf_counter_ns()
    X = np.load(args.X_path)
    docs = np.load(args.docs_path, allow_pickle=True)
    t1 = perf_counter_ns()
    load_val, load_unit = get_unit(t1 - t0)

    # 2) Train/test split
    t0 = perf_counter_ns()
    X_train, X_test, docs_train, docs_test = train_test_split(
        X, docs, test_size=args.test_size, random_state=42
    )
    t1 = perf_counter_ns()
    split_val, split_unit = get_unit(t1 - t0)

    # 3) Normalize if cosine similarity
    if args.objective == "cos":
        X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
        X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

    # 4) Build & index
    t0 = perf_counter_ns()
    d = X_train.shape[1]
    if args.objective == "l2":
        index = faiss.IndexFlatL2(d)
    elif args.objective == "l1":
        index = faiss.IndexFlat(d, faiss.METRIC_L1)
    elif args.objective == "cos":
        index = faiss.IndexFlatIP(d)

    index.add(X_train.astype(np.float32))
    t1 = perf_counter_ns()
    index_val, index_unit = get_unit(t1 - t0)
    index_ns_per_sample = (t1 - t0) // X_train.shape[0]

    # 5) find_indices_batched
    t0 = perf_counter_ns()
    D, I = index.search(X_test.astype(np.float32), args.k)
    t1 = perf_counter_ns()
    q_ns = t1 - t0
    q_val, q_unit = get_unit(q_ns)
    per_q_val, per_q_unit = get_unit(q_ns // len(X_test))

    # 6) find_docs_batched
    t0 = perf_counter_ns()
    docs_out = [[docs_train[i] for i in row] for row in I]
    t1 = perf_counter_ns()
    d_ns = t1 - t0
    d_val, d_unit = get_unit(d_ns)
    per_d_val, per_d_unit = get_unit(d_ns // len(X_test))

    # 7) Print summary
    print(f"\nLoaded   : X {X.shape}, docs {docs.shape} in {load_val:.3f}{load_unit}")
    print(f"Split    : Train {X_train.shape[0]}, Test {X_test.shape[0]} in {split_val:.3f}{split_unit}")
    print(f"Indexing : k={args.k}, threads={args.threads}, obj={args.objective} → {index_val:.3f}{index_unit} "
          f"({index_ns_per_sample} ns/sample)")
    print(f"Queries  : {len(X_test)} vectors")
    print(f" • find_indices_batched total: {q_val:.3f}{q_unit} → {per_q_val:.0f}{per_q_unit}/query")
    print(f" • find_docs_batched    total: {d_val:.3f}{d_unit} → {per_d_val:.0f}{per_d_unit}/query")

    print("\nSample outputs (first 3 test points):")
    for i in range(min(3, len(X_test))):
        print(f" • idxs: {I[i].tolist()} → docs: {docs_out[i]}")
    print("FAISS using", faiss.omp_get_max_threads(), "threads")

if __name__ == "__main__":
    main()
