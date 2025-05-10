import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from time import perf_counter_ns

try:
    import proxi
except ImportError:
    raise SystemExit("Could not import `proxi`. Build & install it first.")

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
    parser.add_argument("-k",    type=int,   default=5,    help="Number of neighbors")
    parser.add_argument("-t",    "--threads", type=int, default=1, help="Number of threads")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to hold out for queries")
    parser.add_argument("--objective", choices=["l2", "l1", "cos"],
                        default="l2", help="Distance function")
    args = parser.parse_args()
    
    X = np.load(args.X_path)
    docs = np.load(args.docs_path, allow_pickle=True)
    
    X_train, X_test, docs_train, docs_test = train_test_split(
        X, docs, test_size=args.test_size, random_state=42
    )
    
    ## Index data
    p = proxi.ProxiFlat(k=args.k, num_threads=args.threads, objective_function=args.objective)
    p.index_data(X_train.astype(np.float32), docs_train)

    ## Print neighbours
    random_query_index = np.random.randint(0, docs_test.shape[0])
    neighbours = p.find_docs(X_test[random_query_index])
    print(neighbours)
    
    ## Save index
    p.save(".")
    
    ## Load index
    p = proxi.ProxiFlat("data.bin")
    
    ## Print neighbours
    print(p.find_docs(X_test[random_query_index]))
    
main()