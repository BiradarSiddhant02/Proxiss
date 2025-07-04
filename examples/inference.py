import argparse
import numpy as np
import faiss
from proxiss import ProxiFlat
from tabulate import tabulate
from sentence_transformers import SentenceTransformer
from time import perf_counter_ns


def print_time(time_ns):
    if time_ns > 1e9:
        return f"{time_ns / 1e9:.3f} s"
    elif time_ns > 1e6:
        return f"{time_ns / 1e6:.3f} ms"
    elif time_ns > 1e3:
        return f"{time_ns / 1e3:.3f} us"
    else:
        return f"{time_ns} ns"


def main():
    parser = argparse.ArgumentParser(description="Compare Proxi and FAISS similarity search.")
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to the .npy file with embeddings.",
    )
    parser.add_argument(
        "--words",
        type=str,
        required=True,
        help="Path to the .npy file with text entries.",
    )
    parser.add_argument("-k", type=int, required=True, help="Number of neighbors to retrieve.")
    args = parser.parse_args()

    k = args.k
    embedding_file = args.embeddings
    text_file = args.words

    print("Loading data...")
    embeddings_np = np.load(embedding_file)
    text_np = np.load(text_file, allow_pickle=True)

    print(f"Loaded {len(embeddings_np)} embeddings with dimension {embeddings_np.shape[1]}")

    # Convert to float32, preferred for ML and FAISS
    if embeddings_np.dtype != np.float32:
        embeddings_f32 = embeddings_np.astype(np.float32)
        del embeddings_np
    else:
        embeddings_f32 = embeddings_np

    print("\nBuilding Proxi index...")
    proxi_index = ProxiFlat(k=k, num_threads=4, objective_function="l2")
    proxi_index.index_data(embeddings_f32, text_np)

    print("Building FAISS index...")
    dimension = embeddings_f32.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings_f32)

    # Free embeddings if no longer needed
    del embeddings_f32

    print("Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.to("cpu")

    while True:
        print("\n" + "=" * 50)
        user_input = input("Enter a word or phrase (or 'quit' to exit): ")

        if user_input.lower() == "quit":
            print("Exiting program. Goodbye!")
            break

        input_embedding = model.encode([user_input], return_numpy=True)
        input_embedding_reshaped = input_embedding.reshape(1, -1).astype(np.float32)

        proxi_start = perf_counter_ns()
        proxi_results = proxi_index.find_docs(input_embedding_reshaped[0].tolist())
        proxi_end = perf_counter_ns()

        faiss_start = perf_counter_ns()
        _, faiss_indices = faiss_index.search(input_embedding_reshaped, k)
        faiss_results = [text_np[idx] for idx in faiss_indices[0]]
        faiss_end = perf_counter_ns()

        print(f"\nSimilarity search results for: '{user_input}'")

        table_data = []
        for i in range(min(len(proxi_results), len(faiss_results))):
            table_data.append([i + 1, proxi_results[i], faiss_results[i]])

        headers = ["Rank", "Proxi Results", "FAISS Results"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print(f"ProxiFlat time: {print_time(proxi_end - proxi_start)}")
        print(f"FAISS time    : {print_time(faiss_end - faiss_start)}")


if __name__ == "__main__":
    main()
