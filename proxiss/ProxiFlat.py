import proxi_cpp  # type: ignore
import numpy as np
from typing import List, Optional, Union


class ProxiFlat:
    def __init__(self, k: int, num_threads: int, objective_function: str = "l2") -> None:
        if k <= 0:
            raise ValueError("K cannot be 0 or negative number.")
        if num_threads <= 0:
            raise ValueError("num_threads cannot be 0 or negative.")

        self.module = proxi_cpp.ProxiFlat(k, num_threads, objective_function)

    def index_data(
        self,
        embeddings: Union[List[List[float]], np.ndarray, None],
        documents: Union[List[str], np.ndarray],
    ) -> None:
        embeddings_np: np.ndarray
        if embeddings is None:
            embeddings_np = np.array([], dtype=np.float32)  # Empty 1D array for C++
        elif isinstance(embeddings, list):
            try:
                embeddings_np = np.array(embeddings, dtype=np.float32)
                if (
                    embeddings_np.ndim == 1
                    and len(embeddings) > 0
                    and isinstance(embeddings[0], list)
                ):  # list of lists but became 1D (e.g. [[1,2], []])
                    pass  # Allow, C++ might handle or error on inconsistent dimensions
                elif embeddings_np.ndim == 0 and len(embeddings) == 0:  # Handles []
                    embeddings_np = np.array([], dtype=np.float32)  # Correctly make it 1D empty
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

        documents_np: np.ndarray
        if isinstance(documents, list):
            try:
                # Attempt to create a NumPy array of strings (objects for pybind11)
                documents_np = np.array(documents, dtype=object)
            except Exception as e:  # Catch generic exception if conversion fails
                raise ValueError(f"Error converting documents list to NumPy array: {e}")
        elif isinstance(documents, np.ndarray):
            documents_np = documents  # Assume it's already a 1D array of strings or compatible
        else:
            raise TypeError("Documents must be a list of strings or a 1D NumPy array.")

        # Ensure documents_np is 1D for C++ bindings, then convert to list for C++
        final_documents_list: list[str]
        if documents_np.ndim != 1:
            # Allow documents_np to be an empty array with shape (0,)
            # but if it has elements, it must be strictly 1D.
            if documents_np.size > 0:
                raise ValueError("Documents NumPy array must be 1D.")
            # Handle cases like np.array([[]]) which is (1,0) or np.empty((0,X))
            # These should become an empty list for the C++ binding.
            final_documents_list = []
        else:
            final_documents_list = documents_np.tolist()

        self.module.index_data(embeddings_np, final_documents_list)

    def find_indices(self, query: Union[List[float], np.ndarray]) -> np.ndarray:
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
        queries_np: np.ndarray
        if isinstance(queries, list):
            try:
                queries_np = np.array(queries, dtype=np.float32)
                if queries_np.ndim == 1 and len(queries) > 0 and isinstance(queries[0], list):
                    pass  # Allow, C++ might handle or error on inconsistent dimensions
                elif queries_np.ndim == 0 and len(queries) == 0:  # Handles []
                    queries_np = np.array([], dtype=np.float32).reshape(
                        0, 0
                    )  # C++ expects 2D for batched
                elif queries_np.ndim != 2 and not (
                    queries_np.ndim == 1 and queries_np.shape[0] == 0
                ):
                    raise ValueError(
                        "Batched queries list must be convertible to a 2D NumPy array or an empty 1D array for empty case."
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
            # If it's an empty 1D array, reshape to (0,0) or (0,D) if D is known, for C++
            if queries_np.ndim == 1 and queries_np.shape[0] == 0:
                # Assuming C++ can handle (0,0) or (0,D) for empty batched queries.
                # If a specific D is required for empty 2D array, this might need adjustment.
                queries_np = queries_np.reshape(0, 0)
            else:
                raise ValueError(
                    "Batched queries NumPy array must be 2D (e.g., (M, D)) or an empty 1D array."
                )

        # C++ returns list[list[int]], convert to NumPy array
        result_list_of_lists = self.module.find_indices_batched(queries_np)
        return np.array(result_list_of_lists, dtype=np.int32)

    def find_docs(self, query: Union[List[float], np.ndarray]) -> List[str]:
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

        # C++ returns list[str]
        return self.module.find_docs(query_np)

    def find_docs_batched(self, queries: Union[List[List[float]], np.ndarray]) -> List[List[str]]:
        queries_np: np.ndarray
        if isinstance(queries, list):
            try:
                queries_np = np.array(queries, dtype=np.float32)
                if queries_np.ndim == 1 and len(queries) > 0 and isinstance(queries[0], list):
                    pass  # Allow
                elif queries_np.ndim == 0 and len(queries) == 0:
                    queries_np = np.array([], dtype=np.float32).reshape(0, 0)
                elif queries_np.ndim != 2 and not (
                    queries_np.ndim == 1 and queries_np.shape[0] == 0
                ):
                    raise ValueError(
                        "Batched queries list must be convertible to a 2D NumPy array or an empty 1D array for empty case."
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

        # C++ returns list[list[str]]
        return self.module.find_docs_batched(queries_np)

    def insert_data(self, embedding: Union[List[float], np.ndarray], text: str) -> None:
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

        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        self.module.insert_data(embedding_np, text)

    def save_state(self, path: str) -> None:
        self.module.save_state(path)

    def load_state(self, path: str) -> None:
        self.module.load_state(path)
