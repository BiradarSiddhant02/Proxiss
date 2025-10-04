import os
import numpy as np
from typing import List, Union
import proxi_knn_cpp


class ProxiKNN:
    """
    Python wrapper for the C++ ProxiKNN class, providing K-Nearest Neighbors classification.

    Methods:
        fit(features, labels): Trains the model on feature vectors and labels.
        predict(feature): Predicts the class label for a single feature vector.
        predict_batch(features): Predicts class labels for a batch of feature vectors.
    """

    def __init__(self, n_neighbours: int, n_jobs: int, distance_function: str = "l2") -> None:
        """
        Initialize the ProxiKNN classifier.

        Args:
            n_neighbours (int): Number of nearest neighbors to consider for classification.
            n_jobs (int): Number of threads to use for computation.
            distance_function (str, optional): Distance metric/objective function. Default is "l2".
        Raises:
            ValueError: If n_neighbours or n_jobs is not positive.
        """
        if n_neighbours <= 0:
            raise ValueError("n_neighbours cannot be 0 or negative number.")
        if n_jobs <= 0:
            raise ValueError("n_jobs cannot be 0 or negative.")

        self.module = proxi_knn_cpp.ProxiKNN(n_neighbours, n_jobs, distance_function)

    def fit(
        self,
        features: Union[List[List[float]], np.ndarray, None],
        labels: Union[List[float], np.ndarray],
    ) -> None:
        """
        Train the KNN model on the provided features and labels.

        Args:
            features (List[List[float]] | np.ndarray | None): 2D array or list of feature vectors, or None for empty.
            labels (List[float] | np.ndarray): 1D array or list of class labels.
        Raises:
            ValueError: If input shapes are invalid or conversion fails.
            TypeError: If input types are invalid.
        """
        features_np: np.ndarray
        if features is None:
            features_np = np.array([], dtype=np.float32).reshape(0, 0)  # Empty 2D array for C++
        elif isinstance(features, list):
            try:
                features_np = np.array(features, dtype=np.float32)
                if (
                    features_np.ndim == 1 and len(features) > 0 and isinstance(features[0], list)
                ):  # list of lists but became 1D (e.g. [[1,2], []])
                    pass  # Allow, C++ might handle or error on inconsistent dimensions
                elif features_np.ndim == 0 and len(features) == 0:  # Handles []
                    features_np = np.array([], dtype=np.float32).reshape(
                        0, 0
                    )  # Correctly make it 2D empty
                elif features_np.ndim != 2 and not (
                    features_np.ndim == 1 and features_np.shape[0] == 0
                ):
                    raise ValueError(
                        "Features list must be convertible to a 2D NumPy array or an empty array."
                    )
            except ValueError as e:
                raise ValueError(f"Error converting features list to NumPy array: {e}")
        elif isinstance(features, np.ndarray):
            if features.dtype != np.float32:
                features_np = features.astype(np.float32)
            else:
                features_np = features
            if not (features_np.ndim == 2 or (features_np.ndim == 1 and features_np.shape[0] == 0)):
                # If it's an empty 1D array, reshape to (0,0) for C++
                if features_np.ndim == 1 and features_np.shape[0] == 0:
                    features_np = features_np.reshape(0, 0)
                else:
                    raise ValueError(
                        "Features NumPy array must be 2D (e.g., (N, D)) or an empty array."
                    )
        else:
            raise TypeError("Features must be a list of lists, a NumPy array, or None.")

        labels_np: np.ndarray
        if isinstance(labels, list):
            try:
                labels_np = np.array(labels, dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Error converting labels list to NumPy array: {e}")
        elif isinstance(labels, np.ndarray):
            if labels.dtype != np.float32:
                labels_np = labels.astype(np.float32)
            else:
                labels_np = labels
        else:
            raise TypeError("Labels must be a list of floats or a 1D NumPy array.")

        # Ensure labels_np is 1D for C++ bindings
        if labels_np.ndim != 1:
            if labels_np.size > 0:
                raise ValueError("Labels NumPy array must be 1D.")
            # Handle empty case
            labels_np = np.array([], dtype=np.float32)

        self.module.fit(features_np, labels_np)

    def predict(self, feature: Union[List[float], np.ndarray]) -> int:
        """
        Predict the class label for a single feature vector.

        Args:
            feature (List[float] | np.ndarray): 1D feature vector.
        Returns:
            float: Predicted class label.
        Raises:
            ValueError: If input shape is invalid or conversion fails.
            TypeError: If input type is invalid.
        """
        feature_np: np.ndarray
        if isinstance(feature, list):
            try:
                feature_np = np.array(feature, dtype=np.float32)
            except ValueError as e:
                raise ValueError(f"Error converting feature list to NumPy array: {e}")
        elif isinstance(feature, np.ndarray):
            if feature.dtype != np.float32:
                feature_np = feature.astype(np.float32)
            else:
                feature_np = feature
        else:
            raise TypeError("Feature must be a list of floats or a 1D NumPy array.")

        if feature_np.ndim != 1:
            raise ValueError("Feature must be a 1D array.")

        return int(self.module.predict(feature_np))

    def predict_batch(self, features: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """
        Predict class labels for a batch of feature vectors.

        Args:
            features (List[List[float]] | np.ndarray): 2D array or list of feature vectors.
        Returns:
            np.ndarray: Predicted class labels for each feature vector.
        Raises:
            ValueError: If input shape is invalid or conversion fails.
            TypeError: If input type is invalid.
        """
        features_np: np.ndarray
        if isinstance(features, list):
            try:
                features_np = np.array(features, dtype=np.float32)
                if features_np.ndim == 1 and len(features) > 0 and isinstance(features[0], list):
                    pass  # Allow, C++ might handle or error on inconsistent dimensions
                elif features_np.ndim == 0 and len(features) == 0:  # Handles []
                    features_np = np.array([], dtype=np.float32).reshape(
                        0, 0
                    )  # C++ expects 2D for batched
                elif features_np.ndim != 2 and not (
                    features_np.ndim == 1 and features_np.shape[0] == 0
                ):
                    raise ValueError(
                        "Batched features list must be convertible to a 2D NumPy array or an empty 1D array for empty case."
                    )
            except ValueError as e:
                raise ValueError(f"Error converting batched features list to NumPy array: {e}")
        elif isinstance(features, np.ndarray):
            if features.dtype != np.float32:
                features_np = features.astype(np.float32)
            else:
                features_np = features
        else:
            raise TypeError(
                "Batched features must be a list of lists of floats or a 2D NumPy array."
            )

        if not (features_np.ndim == 2 or (features_np.ndim == 1 and features_np.shape[0] == 0)):
            # If it's an empty 1D array, reshape to (0,0) for C++
            if features_np.ndim == 1 and features_np.shape[0] == 0:
                features_np = features_np.reshape(0, 0)
            else:
                raise ValueError(
                    "Batched features NumPy array must be 2D (e.g., (M, D)) or an empty 1D array."
                )

        # C++ returns list[float], convert to NumPy array
        result_list = self.module.predict_batch(features_np)
        return np.array(result_list, dtype=np.float32)

    def save_state(self, path: Union[str, os.PathLike]) -> None:
        """
        Persist the trained classifier to disk using the underlying C++ serializer.

        Args:
            path (str | os.PathLike): Directory where the state should be saved. A file named
                ``data.bin`` will be created inside this directory.
        Raises:
            TypeError: If ``path`` is not a string or path-like object.
            RuntimeError: Propagated from the C++ layer if the model is not fitted or saving
                fails.
        """

        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("path must be a string or an os.PathLike object.")

        self.module.save_state(os.fspath(path))

    def load_state(self, path: Union[str, os.PathLike]) -> None:
        """
        Restore a previously saved classifier.

        Args:
            path (str | os.PathLike): Path to the ``data.bin`` file produced by ``save_state``.
        Raises:
            TypeError: If ``path`` is not a string or path-like object.
            RuntimeError: Propagated from the C++ layer if the file cannot be read or is invalid.
        """

        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("path must be a string or an os.PathLike object.")

        self.module.load_state(os.fspath(path))

    def set_n_neighbours(self, n_neighbours: int) -> None:
        """
        Set the number of nearest neighbors to use for prediction.

        Args:
            n_neighbours (int): The new number of neighbors (must be > 0).
        Raises:
            ValueError: If n_neighbours is not positive.
        """
        if n_neighbours <= 0:
            raise ValueError("n_neighbours must be greater than 0")
        self.module.set_n_neighbours(n_neighbours)

    def set_n_jobs(self, n_jobs: int) -> None:
        """
        Set the number of parallel jobs for prediction.

        Args:
            n_jobs (int): The new number of parallel jobs (must be > 0).
        Raises:
            ValueError: If n_jobs is not positive.
        """
        if n_jobs <= 0:
            raise ValueError("n_jobs must be greater than 0")
        self.module.set_n_jobs(n_jobs)

    def get_n_neighbours(self) -> int:
        """
        Get the current number of nearest neighbors.

        Returns:
            int: Current number of neighbors.
        """
        return self.module.get_n_neighbours()

    def get_n_jobs(self) -> int:
        """
        Get the current number of parallel jobs.

        Returns:
            int: Current number of parallel jobs.
        """
        return self.module.get_n_jobs()
