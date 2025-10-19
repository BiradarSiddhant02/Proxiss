import numpy as np
from typing import List, Union
import proxi_pca_cpp


class ProxiPCA:
    """
    Python wrapper for the C++ ProxiPCA class, combining PCA dimensionality reduction 
    with fast nearest-neighbor search.

    ProxiPCA reduces the dimensionality of embeddings using PCA, then indexes the reduced
    embeddings for efficient nearest-neighbor search. For queries, it automatically reduces
    the query dimensions before searching.

    Methods:
        fit_transform_index(embeddings): Fits PCA, transforms embeddings, and indexes them.
        find_indices(query): Finds indices of nearest neighbors for a single query.
        find_indices_batched(queries): Finds indices for a batch of queries.
        insert_data(embedding): Inserts a single embedding (automatically reduced).
        save_state(path): Saves the current state to disk.
        load_state(path): Loads the state from disk.
    """

    def __init__(
        self, 
        n_components: int, 
        k: int, 
        num_threads: int, 
        objective_function: str = "l2"
    ) -> None:
        """
        Initialize the ProxiPCA index.

        Args:
            n_components (int): Number of PCA components (reduced dimensions).
            k (int): Number of nearest neighbors to search for.
            num_threads (int): Number of threads to use for computation.
            objective_function (str, optional): Distance metric/objective function. Default is "l2".
                Options: "l2" (Euclidean), "l1" (Manhattan), "cos" (Cosine)
        
        Raises:
            ValueError: If n_components, k, or num_threads is not positive.
        """
        if n_components <= 0:
            raise ValueError("n_components cannot be 0 or negative number.")
        if k <= 0:
            raise ValueError("k cannot be 0 or negative number.")
        if num_threads <= 0:
            raise ValueError("num_threads cannot be 0 or negative.")

        self.module = proxi_pca_cpp.ProxiPCA(
            n_components, k, num_threads, objective_function
        )
        self.n_components = n_components

    def fit_transform_index(
        self,
        embeddings: Union[List[List[float]], np.ndarray, None],
    ) -> None:
        """
        Fit PCA on embeddings, transform them to reduced dimensions, and index them.

        This method performs three operations in sequence:
        1. Fits PCA model on the input embeddings
        2. Transforms embeddings to n_components dimensions
        3. Indexes the reduced embeddings for fast nearest neighbor search

        Args:
            embeddings (List[List[float]] | np.ndarray | None): 2D array or list of embedding vectors.
        
        Raises:
            ValueError: If input shapes are invalid or conversion fails.
            TypeError: If input types are invalid.
        """
        embeddings_np: np.ndarray
        if embeddings is None:
            embeddings_np = np.array([], dtype=np.float32)
        elif isinstance(embeddings, list):
            try:
                embeddings_np = np.array(embeddings, dtype=np.float32)
                if (
                    embeddings_np.ndim == 1
                    and len(embeddings) > 0
                    and isinstance(embeddings[0], list)
                ):
                    pass  # Allow, C++ might handle or error on inconsistent dimensions
                elif embeddings_np.ndim == 0 and len(embeddings) == 0:
                    embeddings_np = np.array([], dtype=np.float32)
                elif embeddings_np.ndim != 2 and not (
                    embeddings_np.ndim == 1 and embeddings_np.shape[0] == 0
                ):
                    raise ValueError(
                        "Embeddings list must be convertible to a 2D NumPy array or an empty 1D array."
                    )
            except ValueError as e:
                raise ValueError(f"Error converting embeddings list to NumPy array: {e}")
        elif isinstance(embeddings, np.ndarray):
            if embeddings.dtype != np.float32:
                embeddings_np = embeddings.astype(np.float32)
            else:
                embeddings_np = embeddings
            if not (
                embeddings_np.ndim == 2 or (embeddings_np.ndim == 1 and embeddings_np.shape[0] == 0)
            ):
                raise ValueError(
                    "Embeddings NumPy array must be 2D (e.g., (N, D)) or an empty 1D array."
                )
        else:
            raise TypeError("Embeddings must be a list of lists, a NumPy array, or None.")

        self.module.fit_transform_index(embeddings_np)

    def find_indices(self, query: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Find indices of the k nearest neighbors for a single query vector.

        The query is automatically reduced to n_components dimensions using the fitted PCA
        model before searching.

        Args:
            query (List[float] | np.ndarray): 1D query vector with original dimensionality.
        
        Returns:
            np.ndarray: Indices of nearest neighbors.
        
        Raises:
            ValueError: If input shape is invalid or conversion fails.
            TypeError: If input type is invalid.
        """
        query_np: np.ndarray
        if isinstance(query, list):
            try:
                query_np = np.array(query, dtype=np.float32)
            except ValueError as e:
                raise ValueError(f"Error converting query list to NumPy array: {e}")
        elif isinstance(query, np.ndarray):
            if query.dtype != np.float32:
                query_np = query.astype(np.float32)
            else:
                query_np = query
        else:
            raise TypeError("Query must be a list of floats or a 1D NumPy array.")

        if query_np.ndim != 1:
            raise ValueError("Query must be a 1D array.")

        # C++ returns list[int], convert to NumPy array
        result_list = self.module.find_indices(query_np)
        return np.array(result_list, dtype=np.int32)

    def find_indices_batched(self, queries: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """
        Find indices of the k nearest neighbors for a batch of queries.

        All queries are automatically reduced to n_components dimensions using the fitted
        PCA model before searching.

        Args:
            queries (List[List[float]] | np.ndarray): 2D array or list of query vectors.
        
        Returns:
            np.ndarray: Indices of nearest neighbors for each query.
        
        Raises:
            ValueError: If input shape is invalid or conversion fails.
            TypeError: If input type is invalid.
        """
        queries_np: np.ndarray
        if isinstance(queries, list):
            try:
                queries_np = np.array(queries, dtype=np.float32)
                if queries_np.ndim == 1 and len(queries) > 0 and isinstance(queries[0], list):
                    pass
                elif queries_np.ndim == 0 and len(queries) == 0:
                    queries_np = np.array([], dtype=np.float32).reshape(0, 0)
                elif queries_np.ndim != 2 and not (
                    queries_np.ndim == 1 and queries_np.shape[0] == 0
                ):
                    raise ValueError(
                        "Batched queries list must be convertible to a 2D NumPy array."
                    )
            except ValueError as e:
                raise ValueError(f"Error converting batched queries list to NumPy array: {e}")
        elif isinstance(queries, np.ndarray):
            if queries.dtype != np.float32:
                queries_np = queries.astype(np.float32)
            else:
                queries_np = queries
        else:
            raise TypeError(
                "Batched queries must be a list of lists of floats or a 2D NumPy array."
            )

        if not (queries_np.ndim == 2 or (queries_np.ndim == 1 and queries_np.shape[0] == 0)):
            if queries_np.ndim == 1 and queries_np.shape[0] == 0:
                queries_np = queries_np.reshape(0, 0)
            else:
                raise ValueError(
                    "Batched queries NumPy array must be 2D (e.g., (M, D)) or an empty 1D array."
                )

        # C++ returns list[list[int]], convert to NumPy array
        result_list_of_lists = self.module.find_indices_batched(queries_np)
        return np.array(result_list_of_lists, dtype=np.int32)

    def insert_data(self, embedding: Union[List[float], np.ndarray]) -> None:
        """
        Insert a single embedding into the index.

        The embedding is automatically reduced to n_components dimensions using the fitted
        PCA model before insertion.

        Args:
            embedding (List[float] | np.ndarray): 1D embedding vector with original dimensionality.
        
        Raises:
            ValueError: If input shape is invalid or conversion fails.
            TypeError: If input type is invalid.
        """
        embedding_np: np.ndarray
        if isinstance(embedding, list):
            try:
                embedding_np = np.array(embedding, dtype=np.float32)
            except ValueError as e:
                raise ValueError(f"Error converting embedding list to NumPy array: {e}")
        elif isinstance(embedding, np.ndarray):
            if embedding.dtype != np.float32:
                embedding_np = embedding.astype(np.float32)
            else:
                embedding_np = embedding
        else:
            raise TypeError("Embedding must be a list of floats or a 1D NumPy array.")

        if embedding_np.ndim != 1:
            raise ValueError("Embedding must be a 1D array.")

        self.module.insert_data(embedding_np)

    def save_state(self, path: str) -> None:
        """
        Save the current ProxiPCA state (PCA model and index) to disk.

        Args:
            path (str): Directory path to save the state.
        """
        self.module.save_state(path)

    def load_state(self, path: str) -> None:
        """
        Load the ProxiPCA state (PCA model and index) from disk.

        Args:
            path (str): Directory path to load the state from.
        """
        self.module.load_state(path)

    def set_k(self, k: int) -> None:
        """
        Set the number of nearest neighbors to retrieve.

        Args:
            k (int): The new number of neighbors (must be > 0).
        
        Raises:
            ValueError: If k is not positive.
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.module.set_k(k)

    def set_num_threads(self, num_threads: int) -> None:
        """
        Set the number of threads for parallel operations.

        Args:
            num_threads (int): The new number of threads (must be > 0).
        
        Raises:
            ValueError: If num_threads is not positive.
        """
        if num_threads <= 0:
            raise ValueError("num_threads must be greater than 0")
        self.module.set_num_threads(num_threads)

    def get_k(self) -> int:
        """
        Get the current number of nearest neighbors.

        Returns:
            int: Current value of k.
        """
        return self.module.get_k()

    def get_num_threads(self) -> int:
        """
        Get the current number of threads.

        Returns:
            int: Current number of threads.
        """
        return self.module.get_num_threads()

    def get_n_components(self) -> int:
        """
        Get the number of PCA components (reduced dimensions).

        Returns:
            int: Number of PCA components.
        """
        return self.module.get_n_components()

    def is_fitted(self) -> bool:
        """
        Check if the PCA model has been fitted.

        Returns:
            bool: True if fitted, False otherwise.
        """
        return self.module.is_fitted()
