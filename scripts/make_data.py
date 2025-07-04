import argparse
import numpy as np

from sklearn.datasets import make_blobs


def main():
    parser = argparse.ArgumentParser(description="Generate mock data for Proxi benchmark.")
    parser.add_argument(
        "--N", type=int, default=20000, help="Number of data points (default: 20000)"
    )
    parser.add_argument(
        "--D",
        type=int,
        default=768,
        help="Dimensionality of feature vectors (default: 768)",
    )
    parser.add_argument(
        "--X_path",
        default="X.npy",
        help="Output path for feature array (default: X.npy)",
    )
    parser.add_argument(
        "--docs_path",
        default="docs.npy",
        help="Output path for documents array (default: docs.npy)",
    )
    args = parser.parse_args()

    X, y = make_blobs(n_features=args.D, n_samples=args.N, cluster_std=1, centers=4)

    print(f"Generating random feature data of shape ({args.N}, {args.D})...")
    X = X.astype(np.float32)

    print(f"Generating {args.N} mock document strings...")
    docs = np.array([f"{y[i]}" for i in range(args.N)], dtype=object)

    print(f"Saving feature array to '{args.X_path}' and documents to '{args.docs_path}'...")
    np.save(args.X_path, X)
    np.save(args.docs_path, docs)

    print("Done.")


if __name__ == "__main__":
    main()
