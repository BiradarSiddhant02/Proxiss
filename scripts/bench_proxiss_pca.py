import argparse
import numpy as np

from sklearn.model_selection import train_test_split
from time import perf_counter_ns

try:
    import proxiss
except ImportError:
    raise SystemExit("Could not import `proxiss`. Build & install it first.")

print(proxiss.__file__)


def get_unit(ns: int):
    """Return (value, unit) scaled from nanoseconds to the largest convenient unit."""
    if ns >= 1e9:
        return ns / 1e9, "s"
    if ns >= 1e6:
        return ns / 1e6, "ms"
    if ns >= 1e3:
        return ns / 1e3, "us"
    return ns, "ns"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ProxiPCA (PCA + vector-similarity retriever) on a dataset"
    )
    parser.add_argument("--X_path", required=True, help="Path to features (.npy)")
    parser.add_argument("-k", type=int, default=5, help="Number of neighbors")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads")
    parser.add_argument(
        "-c", "--n_components", type=float, default=0.1, 
        help="Number of PCA components as percentage of original dimensions (0.0-1.0, e.g., 0.1 = 10%%)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data to hold out for queries",
    )
    parser.add_argument(
        "--objective",
        choices=["l2", "l1", "cos"],
        default="l2",
        help="Distance function",
    )
    args = parser.parse_args()

    # 1) Load data
    print("Loading data...")
    t0 = perf_counter_ns()
    X = np.load(args.X_path)  # shape (N, D)
    t1 = perf_counter_ns()
    load_ns = t1 - t0
    load_val, load_unit = get_unit(load_ns)

    # 2) Train/test split
    print("Splitting data...")
    t0 = perf_counter_ns()
    X_train, X_test = train_test_split(
        X, test_size=args.test_size, random_state=42
    )
    t1 = perf_counter_ns()
    split_ns = t1 - t0
    split_val, split_unit = get_unit(split_ns)

    # Calculate n_components from percentage
    if args.n_components <= 0 or args.n_components > 1:
        raise ValueError(f"n_components must be between 0.0 and 1.0, got {args.n_components}")
    n_components = max(1, int(X.shape[1] * args.n_components))
    print(f"Using {args.n_components*100:.1f}% of dimensions: {X.shape[1]} → {n_components}")

    # 3) Fit PCA, transform, and index
    print(f"Fitting PCA (reducing {X.shape[1]}D → {n_components}D), transforming, and indexing...")
    t0 = perf_counter_ns()
    pca = proxiss.ProxiPCA(
        n_components=n_components, 
        k=args.k, 
        num_threads=args.threads, 
        objective_function=args.objective
    )
    pca.fit_transform_index(X_train.astype(np.float32))
    t1 = perf_counter_ns()
    fit_transform_index_ns = t1 - t0
    fit_transform_index_val, fit_transform_index_unit = get_unit(fit_transform_index_ns)

    # 4) Batched query (`find_indices_batched`)
    # Note: queries are automatically reduced from D dimensions to n_components before search
    print("Batch querying...")
    t0 = perf_counter_ns()
    idxs = pca.find_indices_batched(X_test.astype(np.float32))
    t1 = perf_counter_ns()
    q_ns = t1 - t0
    q_val, q_unit = get_unit(q_ns)
    per_q_ns = q_ns // len(X_test)
    per_q_val, per_q_unit = get_unit(per_q_ns)

    # 5) Print summary
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS: ProxiPCA")
    print(f"{'='*70}")
    print(f"Loaded          : X {X.shape} in {load_val:.3f}{load_unit}")
    print(
        f"Split           : Train {X_train.shape[0]}, Test {X_test.shape[0]} in {split_val:.3f}{split_unit}"
    )
    print(
        f"PCA Reduction   : {X.shape[1]}D → {n_components}D ({args.n_components*100:.1f}%)"
    )
    print(
        f"Fit+Transform+  : k={args.k}, threads={args.threads}, obj={args.objective}"
    )
    print(
        f"Index           : {fit_transform_index_val:.3f}{fit_transform_index_unit} "
        f"({fit_transform_index_ns//X_train.shape[0]} ns/sample)"
    )
    print(f"Queries         : {len(X_test)} vectors (auto-reduced from {X.shape[1]}D → {n_components}D)")
    print(f" • find_indices_batched total: {q_val:.3f}{q_unit} → {per_q_val:.0f}{per_q_unit}/query")
    print(f"{'='*70}")

    # (Optional) show first few matches
    print("\nSample outputs (first 3 test points):")
    for i in range(min(3, len(X_test))):
        print(f" • Query {i}: nearest indices → {idxs[i]}")

    # 6) Additional info
    print(f"\nProxiPCA Info:")
    print(f" • PCA fitted: {pca.is_fitted()}")
    print(f" • n_components: {pca.get_n_components()}")
    print(f" • k neighbors: {pca.get_k()}")
    print(f" • threads: {pca.get_num_threads()}")


if __name__ == "__main__":
    main()
