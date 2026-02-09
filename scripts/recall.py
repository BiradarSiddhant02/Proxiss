import argparse
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
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


def recall_at_k(pred: np.ndarray, truth: np.ndarray) -> float:
    """Compute recall@k: fraction of ground truth neighbors found in predictions."""
    total = 0.0
    for i in range(len(pred)):
        total += len(set(pred[i]) & set(truth[i])) / len(truth[i])
    return total / len(pred)


def main():
    parser = argparse.ArgumentParser(
        description="Test recall of Proxiss modules against sklearn ground truth"
    )
    parser.add_argument("--X_path", required=True, help="Path to features (.npy)")
    parser.add_argument("--docs_path", help="Path to labels (.npy) for KNN classification accuracy")
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
    parser.add_argument(
        "--pca_ratio",
        type=float,
        default=0.25,
        help="PCA component ratio (0.0-1.0)",
    )
    args = parser.parse_args()

    sklearn_metric = {"l2": "euclidean", "l1": "manhattan", "cos": "cosine"}[args.objective]

    # 1) Load data
    t0 = perf_counter_ns()
    X = np.load(args.X_path)  # shape (N, D)
    if args.docs_path:
        docs = np.load(args.docs_path, allow_pickle=True)
    t1 = perf_counter_ns()
    load_val, load_unit = get_unit(t1 - t0)

    # 2) Train/test split
    t0 = perf_counter_ns()
    if args.docs_path:
        X_train, X_test, docs_train, docs_test = train_test_split(
            X, docs, test_size=args.test_size, random_state=42
        )
        le = LabelEncoder()
        labels_train = le.fit_transform(docs_train)
    else:
        X_train, X_test = train_test_split(X, test_size=args.test_size, random_state=42)
        labels_train = np.random.randint(0, 4, size=len(X_train))
    t1 = perf_counter_ns()
    split_val, split_unit = get_unit(t1 - t0)

    print(f"\nLoaded   : X {X.shape} in {load_val:.3f}{load_unit}")
    print(f"Split    : Train {X_train.shape[0]}, Test {X_test.shape[0]} in {split_val:.3f}{split_unit}")
    print(f"Config   : k={args.k}, threads={args.threads}, obj={args.objective}")

    # 3) Compute ground truth with sklearn
    print("\nComputing ground truth (sklearn brute force)...")
    t0 = perf_counter_ns()
    nn = NearestNeighbors(n_neighbors=args.k, metric=sklearn_metric, algorithm="brute")
    nn.fit(X_train)
    _, gt_indices = nn.kneighbors(X_test)
    t1 = perf_counter_ns()
    gt_val, gt_unit = get_unit(t1 - t0)
    print(f"Ground truth computed in {gt_val:.3f}{gt_unit}")

    # 4) Test ProxiFlat
    print("\n--- ProxiFlat ---")
    t0 = perf_counter_ns()
    flat = proxiss.ProxiFlat(k=args.k, num_threads=args.threads, objective_function=args.objective)
    flat.index_data(X_train.astype(np.float32))
    t1 = perf_counter_ns()
    idx_val, idx_unit = get_unit(t1 - t0)

    t0 = perf_counter_ns()
    flat_pred = np.array(flat.find_indices_batched(X_test.astype(np.float32)))
    t1 = perf_counter_ns()
    q_val, q_unit = get_unit(t1 - t0)

    flat_recall = recall_at_k(flat_pred, gt_indices)
    print(f"Index    : {idx_val:.3f}{idx_unit}")
    print(f"Query    : {q_val:.3f}{q_unit}")
    print(f"Recall@{args.k} : {flat_recall*100:.2f}%")

    # 5) Test ProxiKNN
    print("\n--- ProxiKNN ---")
    t0 = perf_counter_ns()
    knn = proxiss.ProxiKNN(n_neighbours=args.k, n_jobs=args.threads, distance_function=args.objective)
    knn.fit(X_train.astype(np.float32), labels_train.astype(np.float32))
    t1 = perf_counter_ns()
    idx_val, idx_unit = get_unit(t1 - t0)

    t0 = perf_counter_ns()
    knn_preds = np.array(knn.predict_batch(X_test.astype(np.float32))).astype(int)
    t1 = perf_counter_ns()
    q_val, q_unit = get_unit(t1 - t0)

    # Compute expected labels via majority vote of ground truth neighbors
    gt_labels = np.array([
        np.bincount(labels_train[gt_indices[i]]).argmax()
        for i in range(len(X_test))
    ])
    knn_acc = np.mean(knn_preds == gt_labels)
    print(f"Index    : {idx_val:.3f}{idx_unit}")
    print(f"Query    : {q_val:.3f}{q_unit}")
    print(f"Accuracy : {knn_acc*100:.2f}% (vs ground truth neighbor majority vote)")

    # 6) Test ProxiPCA
    n_components = max(1, int(X.shape[1] * args.pca_ratio))
    print(f"\n--- ProxiPCA ({args.pca_ratio*100:.0f}% → {n_components}D) ---")
    t0 = perf_counter_ns()
    pca = proxiss.ProxiPCA(
        n_components=n_components,
        k=args.k,
        num_threads=args.threads,
        objective_function=args.objective
    )
    pca.fit_transform_index(X_train.astype(np.float32))
    t1 = perf_counter_ns()
    idx_val, idx_unit = get_unit(t1 - t0)

    t0 = perf_counter_ns()
    pca_pred = np.array(pca.find_indices_batched(X_test.astype(np.float32)))
    t1 = perf_counter_ns()
    q_val, q_unit = get_unit(t1 - t0)

    pca_recall = recall_at_k(pca_pred, gt_indices)
    print(f"Index    : {idx_val:.3f}{idx_unit}")
    print(f"Query    : {q_val:.3f}{q_unit}")
    print(f"Recall@{args.k} : {pca_recall*100:.2f}%")

    # 7) Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"ProxiFlat  Recall@{args.k} : {flat_recall*100:>6.2f}%")
    print(f"ProxiKNN   Accuracy   : {knn_acc*100:>6.2f}%")
    print(f"ProxiPCA   Recall@{args.k} : {pca_recall*100:>6.2f}%")
    print(f"{'='*50}")

    if flat_recall == 1.0:
        print("\nProxiFlat: 100% recall (exact search, expected)")
    elif flat_recall < 0.99:
        print(f"\nWARNING: ProxiFlat recall < 99% — possible bug")


if __name__ == "__main__":
    main()
