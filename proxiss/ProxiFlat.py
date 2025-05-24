import proxi_cpp  # type: ignore
import numpy as np
from typing import List


class ProxiFlat:
    """
    Python wrapper for the C++ ProxiFlat class, providing high-performance nearest-neighbor search.

    Methods:
        index_data(embeddings, documents): Indexes embeddings and associated documents.
        find_indices(query): Finds indices of nearest neighbors for a single query.
        find_indices_batched(queries): Finds indices for a batch of queries.
        find_docs(query): Finds documents for a single query.
        find_docs_batched(queries): Finds documents for a batch of queries.
        insert_data(embedding, text): Inserts a single embedding and document.
        save_state(path): Saves the current index state to disk.
        load_state(path): Loads the index state from disk.
    """
    def __init__(self, k: int, num_threads: int, objective_function: str = "l2") -> None:
        """
        Initialize the ProxiFlat index.

        Args:
            k (int): Number of nearest neighbors to search for.
            num_threads (int): Number of threads to use for computation.
            objective_function (str, optional): Distance metric/objective function. Default is "l2".
        Raises:
            ValueError: If k or num_threads is not positive.
        """
        if k <= 0:
            raise ValueError("K cannot be 0 or negative number.")
        if num_threads <= 0:
            raise ValueError("num_threads cannot be 0 or negative.")

        self.module = proxi_cpp.ProxiFlat(k, num_threads, objective_function)

    def index_data(
        self,
        embeddings: np.ndarray,
        documents: np.ndarray,
    ) -> None:
        """
        Index the provided embeddings and associated documents.

        Args:
            embeddings (np.ndarray): 2D array of embedding vectors.
            documents (np.ndarray): 1D array of document strings.
        Raises:
            ValueError: If input shapes or dtypes are invalid.
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a NumPy array.")
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if not (embeddings.ndim == 2 or (embeddings.ndim == 1 and embeddings.shape[0] == 0)):
            raise ValueError("Embeddings NumPy array must be 2D (N, D) or an empty 1D array.")

        if not isinstance(documents, np.ndarray):
            raise TypeError("Documents must be a NumPy array.")
        if documents.dtype != object:
            documents = documents.astype(object)
        if not (documents.ndim == 1 or (documents.ndim == 1 and documents.shape[0] == 0)):
            raise ValueError("Documents NumPy array must be 1D.")

        self.module.index_data(embeddings, documents.tolist())

    def find_indices(self, query: np.ndarray) -> np.ndarray:
        """
        Find indices of the k nearest neighbors for a single query vector.

        Args:
            query (np.ndarray): 1D query vector.
        Returns:
            np.ndarray: Indices of nearest neighbors.
        Raises:
            ValueError: If input shape or dtype is invalid.
        """
        if not isinstance(query, np.ndarray):
            raise TypeError("Query must be a NumPy array.")
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim != 1:
            raise ValueError("Query must be a 1D array.")
        result_list = self.module.find_indices(query)
        return np.array(result_list, dtype=np.int32)

    def find_indices_batched(self, queries: np.ndarray) -> np.ndarray:
        """
        Find indices of the k nearest neighbors for a batch of queries.

        Args:
            queries (np.ndarray): 2D array of query vectors.
        Returns:
            np.ndarray: Indices of nearest neighbors for each query.
        Raises:
            ValueError: If input shape or dtype is invalid.
        """
        if not isinstance(queries, np.ndarray):
            raise TypeError("Batched queries must be a NumPy array.")
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        if not (queries.ndim == 2 or (queries.ndim == 1 and queries.shape[0] == 0)):
            raise ValueError("Batched queries NumPy array must be 2D (M, D) or an empty 1D array.")
        result_list_of_lists = self.module.find_indices_batched(queries)
        return np.array(result_list_of_lists, dtype=np.int32)

    def find_docs(self, query: np.ndarray) -> List[str]:
        """
        Find the documents corresponding to the k nearest neighbors for a single query vector.

        Args:
            query (np.ndarray): 1D query vector.
        Returns:
            List[str]: Documents of nearest neighbors.
        Raises:
            ValueError: If input shape or dtype is invalid.
        """
        if not isinstance(query, np.ndarray):
            raise TypeError("Query must be a NumPy array.")
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if query.ndim != 1:
            raise ValueError("Query must be a 1D array.")
        return self.module.find_docs(query)

    def find_docs_batched(self, queries: np.ndarray) -> List[List[str]]:
        """
        Find the documents corresponding to the k nearest neighbors for a batch of queries.

        Args:
            queries (np.ndarray): 2D array of query vectors.
        Returns:
            List[List[str]]: Documents of nearest neighbors for each query.
        Raises:
            ValueError: If input shape or dtype is invalid.
        """
        if not isinstance(queries, np.ndarray):
            raise TypeError("Batched queries must be a NumPy array.")
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        if not (queries.ndim == 2 or (queries.ndim == 1 and queries.shape[0] == 0)):
            raise ValueError("Batched queries NumPy array must be 2D (M, D) or an empty 1D array.")
        return self.module.find_docs_batched(queries)

    def insert_data(self, embedding: np.ndarray, text: str) -> None:
        """
        Insert a single embedding and its associated document into the index.

        Args:
            embedding (np.ndarray): 1D embedding vector.
            text (str): Document string.
        Raises:
            ValueError: If input shape or dtype is invalid.
        """
        if not isinstance(embedding, np.ndarray):
            raise TypeError("Embedding must be a NumPy array.")
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        if embedding.ndim != 1:
            raise ValueError("Embedding must be a 1D array.")
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")
        self.module.insert_data(embedding, text)

    def save_state(self, path: str) -> None:
        """
        Save the current index state to disk.

        Args:
            path (str): File path to save the state.
        """
        self.module.save_state(path)

    def load_state(self, path: str) -> None:
        """
        Load the index state from disk.

        Args:
            path (str): File path to load the state from.
        """
        self.module.load_state(path)
