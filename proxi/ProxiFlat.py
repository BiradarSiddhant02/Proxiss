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
        processed_embeddings: List[List[float]]
        if embeddings is None:
            processed_embeddings = []
        elif isinstance(embeddings, np.ndarray):
            if embeddings.ndim == 1 and embeddings.shape[0] == 0:  # Handles np.array([])
                processed_embeddings = []
            elif embeddings.ndim == 2:
                processed_embeddings = embeddings.astype(np.float32).tolist()
            else:
                raise ValueError(
                    "Embeddings NumPy array must be 2D (e.g., (N, D)) or an empty 1D array."
                )
        elif isinstance(embeddings, list):
            # TODO: Consider adding validation for list of lists of floats
            processed_embeddings = embeddings
        else:
            raise TypeError("Embeddings must be a list of lists of floats, a NumPy array, or None.")

        processed_documents: List[str]
        if isinstance(documents, np.ndarray):
            if documents.ndim == 1:
                processed_documents = documents.tolist()
            else:
                raise ValueError("Documents NumPy array must be 1D.")
        elif isinstance(documents, list):
            # TODO: Consider adding validation for list of strings
            processed_documents = documents
        else:
            raise TypeError("Documents must be a list of strings or a 1D NumPy array of strings.")

        self.module.index_data(processed_embeddings, processed_documents)

    def find_indices(self, query: Union[List[float], np.ndarray]) -> np.ndarray:
        processed_query: List[float]
        if isinstance(query, np.ndarray):
            if query.ndim == 1:
                processed_query = query.astype(np.float32).tolist()
            else:
                raise ValueError("Query NumPy array must be 1D.")
        elif isinstance(query, list):
            # TODO: Consider adding validation for list of floats
            processed_query = query
        else:
            raise TypeError("Query must be a list of floats or a 1D NumPy array.")

        result_list = self.module.find_indices(processed_query)
        return np.array(result_list, dtype=np.int32)

    def find_indices_batched(self, queries: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        processed_queries: List[List[float]]
        if isinstance(queries, np.ndarray):
            if queries.ndim == 1 and queries.shape[0] == 0:  # Handles np.array([])
                processed_queries = []
            elif queries.ndim == 2:
                processed_queries = queries.astype(np.float32).tolist()
            else:
                raise ValueError(
                    "Batched queries NumPy array must be 2D (e.g., (M, D)) or an empty 1D array."
                )
        elif isinstance(queries, list):
            # TODO: Consider adding validation for list of lists of floats
            processed_queries = queries
        else:
            raise TypeError(
                "Batched queries must be a list of lists of floats or a 2D NumPy array."
            )

        result_list_of_lists = self.module.find_indices_batched(processed_queries)
        return np.array(result_list_of_lists, dtype=np.int32)

    def find_docs(self, query: Union[List[float], np.ndarray]) -> List[str]:
        processed_query: List[float]
        if isinstance(query, np.ndarray):
            if query.ndim == 1:
                processed_query = query.astype(np.float32).tolist()
            else:
                raise ValueError("Query NumPy array must be 1D.")
        elif isinstance(query, list):
            # TODO: Consider adding validation for list of floats
            processed_query = query
        else:
            raise TypeError("Query must be a list of floats or a 1D NumPy array.")

        return self.module.find_docs(processed_query)

    def find_docs_batched(self, queries: Union[List[List[float]], np.ndarray]) -> List[List[str]]:
        processed_queries: List[List[float]]
        if isinstance(queries, np.ndarray):
            if queries.ndim == 1 and queries.shape[0] == 0:  # Handles np.array([])
                processed_queries = []
            elif queries.ndim == 2:
                processed_queries = queries.astype(np.float32).tolist()
            else:
                raise ValueError(
                    "Batched queries NumPy array must be 2D (e.g., (M, D)) or an empty 1D array."
                )
        elif isinstance(queries, list):
            # TODO: Consider adding validation for list of lists of floats
            processed_queries = queries
        else:
            raise TypeError(
                "Batched queries must be a list of lists of floats or a 2D NumPy array."
            )

        return self.module.find_docs_batched(processed_queries)

    def insert_data(self, embedding: Union[List[float], np.ndarray], text: str) -> None:
        processed_embedding: List[float]
        if isinstance(embedding, np.ndarray):
            if embedding.ndim == 1:
                processed_embedding = embedding.astype(np.float32).tolist()
            else:
                raise ValueError("Embedding NumPy array must be 1D.")
        elif isinstance(embedding, list):
            # TODO: Consider adding validation for list of floats
            processed_embedding = embedding
        else:
            raise TypeError("Embedding must be a list of floats or a 1D NumPy array.")

        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        self.module.insert_data(processed_embedding, text)

    def save_state(self, path: str) -> None:
        self.module.save_state(path)

    def load_state(self, path: str) -> None:
        self.module.load_state(path)
