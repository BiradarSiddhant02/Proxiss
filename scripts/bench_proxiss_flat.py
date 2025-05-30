import argparse
import numpy as np

# import sys
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
        description="Benchmark Proxi (vector-similarity retriever) on a dataset"
    )
    parser.add_argument("--X_path", required=True, help="Path to features (.npy)")
    parser.add_argument("--docs_path", required=True, help="Path to documents (.npy of strings)")
    parser.add_argument("-k", type=int, default=5, help="Number of neighbors")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads")
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
    t0 = perf_counter_ns()
    X = np.load(args.X_path)  # shape (N, D)
    docs = np.load(args.docs_path, allow_pickle=True)  # shape (N,)
    t1 = perf_counter_ns()
    load_ns = t1 - t0
    load_val, load_unit = get_unit(load_ns)

    # 2) Train/test split
    t0 = perf_counter_ns()
    X_train, X_test, docs_train, docs_test = train_test_split(
        X, docs, test_size=args.test_size, random_state=42
    )
    t1 = perf_counter_ns()
    split_ns = t1 - t0
    split_val, split_unit = get_unit(split_ns)

    # 3) Build & index
    t0 = perf_counter_ns()
    p = proxiss.ProxiFlat(k=args.k, num_threads=args.threads, objective_function=args.objective)
    p.index_data(X_train.astype(np.float32), docs_train)
    t1 = perf_counter_ns()
    index_ns = t1 - t0
    index_val, index_unit = get_unit(index_ns)

    # 4) Batched query (`find_indices_batched`)
    t0 = perf_counter_ns()
    idxs = p.find_indices_batched(X_test.astype(np.float32))
    t1 = perf_counter_ns()
    q_ns = t1 - t0
    q_val, q_unit = get_unit(q_ns)
    per_q_ns = q_ns // len(X_test)
    per_q_val, per_q_unit = get_unit(per_q_ns)

    # 5) Batched query (`find_docs_batched`)
    t0 = perf_counter_ns()
    docs_out = p.find_docs_batched(X_test.astype(np.float32))
    t1 = perf_counter_ns()
    d_ns = t1 - t0
    d_val, d_unit = get_unit(d_ns)
    per_d_ns = d_ns // len(X_test)
    per_d_val, per_d_unit = get_unit(per_d_ns)

    # 6) Print summary
    print(f"\nLoaded   : X {X.shape}, docs {docs.shape} in {load_val:.3f}{load_unit}")
    print(
        f"Split    : Train {X_train.shape[0]}, Test {X_test.shape[0]} in {split_val:.3f}{split_unit}"
    )
    print(
        f"Indexing : k={args.k}, threads={args.threads}, obj={args.objective} → {index_val:.3f}{index_unit} "
        f"({index_ns//X_train.shape[0]} ns/sample)"
    )
    print(f"Queries  : {len(X_test)} vectors")
    print(f" • find_indices_batched total: {q_val:.3f}{q_unit} → {per_q_val:.0f}{per_q_unit}/query")
    print(f" • find_docs_batched    total: {d_val:.3f}{d_unit} → {per_d_val:.0f}{per_d_unit}/query")

    # (Optional) show first few matches
    print("\nSample outputs (first 3 test points):")
    for i in range(min(3, len(X_test))):
        print(f" • idxs: {idxs[i]} → docs: {docs_out[i]}")


if __name__ == "__main__":
    main()
