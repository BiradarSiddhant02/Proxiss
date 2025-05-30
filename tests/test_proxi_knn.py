import numpy as np
import pytest
from proxiss import ProxiKNN

class TestProxiKNN:
    def test_initialization(self):
        """Test ProxiKNN initialization with valid parameters."""
        knn = ProxiKNN(n_neighbours=3, n_jobs=1, distance_function="l2")
        assert knn is not None
        
    def test_initialization_invalid_params(self):
        """Test ProxiKNN initialization with invalid parameters."""
        with pytest.raises(ValueError):
            ProxiKNN(n_neighbours=0, n_jobs=1)
        
        with pytest.raises(ValueError):
            ProxiKNN(n_neighbours=3, n_jobs=0)
            
        with pytest.raises(ValueError):
            ProxiKNN(n_neighbours=-1, n_jobs=1)

    def test_simple_classification(self):
        """Test basic classification functionality."""
        # Create simple 2D dataset
        features = np.array([
            [1.0, 1.0],  # class 0
            [1.1, 1.1],  # class 0
            [0.9, 0.9],  # class 0
            [5.0, 5.0],  # class 1
            [5.1, 5.1],  # class 1
            [4.9, 4.9],  # class 1
        ], dtype=np.float32)
        
        labels = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        knn = ProxiKNN(n_neighbours=3, n_jobs=1, distance_function="l2")
        knn.fit(features, labels)
        
        # Test prediction for a point close to class 0
        test_point_0 = np.array([1.0, 1.0], dtype=np.float32)
        prediction_0 = knn.predict(test_point_0)
        assert prediction_0 == 0.0
        
        # Test prediction for a point close to class 1
        test_point_1 = np.array([5.0, 5.0], dtype=np.float32)
        prediction_1 = knn.predict(test_point_1)
        assert prediction_1 == 1.0

    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        # Create simple 2D dataset
        features = np.array([
            [1.0, 1.0],  # class 0
            [1.1, 1.1],  # class 0
            [0.9, 0.9],  # class 0
            [5.0, 5.0],  # class 1
            [5.1, 5.1],  # class 1
            [4.9, 4.9],  # class 1
        ], dtype=np.float32)
        
        labels = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        knn = ProxiKNN(n_neighbours=3, n_jobs=1, distance_function="l2")
        knn.fit(features, labels)
        
        # Test batch prediction
        test_points = np.array([
            [1.0, 1.0],  # should predict class 0
            [5.0, 5.0],  # should predict class 1
        ], dtype=np.float32)
        
        predictions = knn.predict_batch(test_points)
        assert len(predictions) == 2
        assert predictions[0] == 0.0
        assert predictions[1] == 1.0

    def test_different_distance_functions(self):
        """Test different distance functions."""
        features = np.array([
            [1.0, 1.0],
            [1.1, 1.1],
            [5.0, 5.0],
            [5.1, 5.1],
        ], dtype=np.float32)
        
        labels = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        
        for distance_func in ["l1", "l2", "cos"]:
            knn = ProxiKNN(n_neighbours=2, n_jobs=1, distance_function=distance_func)
            knn.fit(features, labels)
            
            test_point = np.array([1.0, 1.0], dtype=np.float32)
        prediction = knn.predict(test_point)
        assert prediction == 0.0

    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        knn = ProxiKNN(n_neighbours=3, n_jobs=1, distance_function="l2")

        # Test with empty features and labels - should raise an error
        empty_features = np.array([], dtype=np.float32).reshape(0, 2)
        empty_labels = np.array([], dtype=np.float32)

        with pytest.raises(RuntimeError, match="Embeddings or Documents cannot be empty"):
            knn.fit(empty_features, empty_labels)

    def test_input_type_validation(self):
        """Test input type validation."""
        knn = ProxiKNN(n_neighbours=3, n_jobs=1, distance_function="l2")
        
        # Invalid features type
        with pytest.raises(TypeError):
            knn.fit("invalid", [0.0, 1.0])
        
        # Invalid labels type
        with pytest.raises(TypeError):
            knn.fit([[1.0, 2.0]], "invalid")

    def test_list_input(self):
        """Test that list inputs work correctly."""
        features_list = [[1.0, 1.0], [1.1, 1.1], [5.0, 5.0], [5.1, 5.1]]
        labels_list = [0.0, 0.0, 1.0, 1.0]
        
        knn = ProxiKNN(n_neighbours=2, n_jobs=1, distance_function="l2")
        knn.fit(features_list, labels_list)
        
        # Test with list input for prediction
        test_point_list = [1.0, 1.0]
        prediction = knn.predict(test_point_list)
        assert prediction == 0.0
        
        # Test batch prediction with list input
        test_points_list = [[1.0, 1.0], [5.0, 5.0]]
        predictions = knn.predict_batch(test_points_list)
        assert len(predictions) == 2

    def test_multithreading(self):
        """Test multithreading functionality."""
        features = np.random.rand(100, 10).astype(np.float32)
        labels = np.random.choice([0.0, 1.0, 2.0], size=100).astype(np.float32)
        
        # Single thread
        knn_single = ProxiKNN(n_neighbours=5, n_jobs=1, distance_function="l2")
        knn_single.fit(features, labels)
        
        # Multiple threads
        knn_multi = ProxiKNN(n_neighbours=5, n_jobs=2, distance_function="l2")
        knn_multi.fit(features, labels)
        
        test_point = np.random.rand(10).astype(np.float32)
        
        # Results should be the same regardless of thread count
        pred_single = knn_single.predict(test_point)
        pred_multi = knn_multi.predict(test_point)
        
        assert pred_single == pred_multi

if __name__ == "__main__":
    pytest.main([__file__])
