import unittest
import os
import shutil
import numpy as np
import proxiss


class TestProxiPCA(unittest.TestCase):
    """Test suite for the ProxiPCA class and its Python bindings."""

    def setUp(self):
        """Set up test fixtures before each test method.

        Initializes common parameters (n_components, k, num_threads, dimensions) and sample data
        (embeddings, query vectors). Also creates a temporary directory for save/load tests.
        """
        self.n_components = 2  # Reduce to 2 dimensions for PCA
        self.k = 2
        self.num_threads = 1
        self.num_samples = 10
        self.num_features = 5  # Original dimensions

        # Sample data - enough samples for meaningful PCA
        np.random.seed(42)
        self.embeddings = np.random.randn(self.num_samples, self.num_features).astype(np.float32)
        
        # Make the data have some structure for PCA
        self.embeddings[:5] += [1.0, 2.0, 3.0, 4.0, 5.0]
        self.embeddings[5:] -= [1.0, 2.0, 3.0, 4.0, 5.0]

        self.query_vector = self.embeddings[0].copy()  # Use first embedding as query
        self.query_vector_new = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        self.batched_queries = np.array([
            self.embeddings[0],
            self.embeddings[5],
            self.query_vector_new,
        ], dtype=np.float32)

        # Temporary directory for save/load tests
        self.temp_dir = "temp_test_proxi_pca_data"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.save_directory = self.temp_dir
        self.save_file_path = os.path.join(self.temp_dir, "data.bin")

        # Convert numpy arrays to lists for testing
        self.embeddings_list = self.embeddings.tolist()
        self.batched_queries_list = self.batched_queries.tolist()

    def tearDown(self):
        """Tear down test fixtures after each test method.

        Removes the temporary directory created for save/load tests.
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_01_constructor_basic(self):
        """Test basic ProxiPCA constructor with default L2 distance.

        Ensures that a ProxiPCA object can be created with n_components, k, and num_threads.
        """
        pca = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        self.assertIsNotNone(pca)
        self.assertEqual(pca.get_n_components(), self.n_components)
        self.assertEqual(pca.get_k(), self.k)
        self.assertEqual(pca.get_num_threads(), self.num_threads)
        self.assertFalse(pca.is_fitted())

    def test_02_constructor_with_objective(self):
        """Test ProxiPCA constructor with different objective functions.

        Verifies that ProxiPCA can be created with l2, l1, and cos objectives.
        """
        for obj in ["l2", "l1", "cos"]:
            pca = proxiss.ProxiPCA(
                n_components=self.n_components,
                k=self.k,
                num_threads=self.num_threads,
                objective_function=obj
            )
            self.assertIsNotNone(pca)

    def test_03_fit_transform_index_basic(self):
        """Test fit_transform_index with valid data.

        Ensures that PCA can fit, transform, and index data successfully.
        """
        pca = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        pca.fit_transform_index(self.embeddings)
        self.assertTrue(pca.is_fitted())

    def test_04_fit_transform_index_with_lists(self):
        """Test fit_transform_index with list input instead of numpy array."""
        pca = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        pca.fit_transform_index(self.embeddings_list)
        self.assertTrue(pca.is_fitted())

    def test_05_find_indices_single_query(self):
        """Test find_indices with a single query after fitting.

        Verifies that querying returns the correct number of neighbors.
        """
        pca = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        pca.fit_transform_index(self.embeddings)
        
        indices = pca.find_indices(self.query_vector)
        self.assertEqual(len(indices), self.k)
        self.assertIn(0, indices)  # Query is the first embedding, so 0 should be in results

    def test_06_find_indices_batched(self):
        """Test find_indices_batched with multiple queries.

        Ensures that batch querying returns correct results for multiple queries.
        """
        pca = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        pca.fit_transform_index(self.embeddings)
        
        results = pca.find_indices_batched(self.batched_queries)
        self.assertEqual(len(results), len(self.batched_queries))
        self.assertEqual(len(results[0]), self.k)
        
        # First query is self.embeddings[0], so index 0 should be in results
        self.assertIn(0, results[0])

    def test_07_insert_data_after_fit(self):
        """Test insert_data to add new embeddings after initial indexing.

        Verifies that new data can be inserted and then retrieved.
        """
        pca = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        pca.fit_transform_index(self.embeddings)
        
        new_embedding = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float32)
        pca.insert_data(new_embedding)
        
        # Query for the newly inserted item
        indices = pca.find_indices(new_embedding)
        self.assertEqual(len(indices), self.k)
        # The new item should be one of its own closest neighbors
        self.assertIn(self.num_samples, indices)

    def test_08_save_and_load(self):
        """Test save_state and load_state functionality.

        Verifies that a fitted ProxiPCA can be saved and loaded, maintaining state.
        
        NOTE: PCA state (components, mean, variance) is now fully serialized.
        """
        pca_orig = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        pca_orig.fit_transform_index(self.embeddings)
        pca_orig.save_state(self.save_directory)
        
        # Load into a new instance (using directory path)
        pca_loaded = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        pca_loaded.load_state(self.save_directory)
        
        # Verify loaded instance works correctly
        self.assertTrue(pca_loaded.is_fitted())
        indices = pca_loaded.find_indices(self.query_vector)
        self.assertEqual(len(indices), self.k)
        self.assertIn(0, indices)

    def test_09_query_before_fit(self):
        """Test that querying before fit_transform_index raises an error.

        Verifies that attempting to query without fitting raises RuntimeError.
        """
        pca = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        with self.assertRaises(RuntimeError):
            pca.find_indices(self.query_vector)

    def test_10_save_before_fit(self):
        """Test that saving before fit_transform_index raises an error.

        Verifies that attempting to save without fitting raises RuntimeError.
        """
        pca = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        with self.assertRaises(RuntimeError):
            pca.save_state(self.save_directory)

    def test_11_load_invalid_path(self):
        """Test loading from a non-existent path raises an error."""
        pca = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        non_existent_path = os.path.join(self.temp_dir, "non_existent")
        with self.assertRaises(RuntimeError):
            pca.load_state(non_existent_path)

    def test_12_invalid_n_components(self):
        """Test constructor with invalid n_components values.

        Verifies that zero or negative n_components raise ValueError.
        """
        with self.assertRaises(ValueError):
            proxiss.ProxiPCA(n_components=0, k=self.k, num_threads=self.num_threads)
        
        with self.assertRaises(ValueError):
            proxiss.ProxiPCA(n_components=-1, k=self.k, num_threads=self.num_threads)

    def test_13_invalid_k(self):
        """Test constructor with invalid k values.

        Verifies that zero or negative k raise ValueError.
        """
        with self.assertRaises(ValueError):
            proxiss.ProxiPCA(n_components=self.n_components, k=0, num_threads=self.num_threads)
        
        with self.assertRaises(ValueError):
            proxiss.ProxiPCA(n_components=self.n_components, k=-1, num_threads=self.num_threads)

    def test_14_invalid_num_threads(self):
        """Test constructor with invalid num_threads values.

        Verifies that zero or negative num_threads raise ValueError.
        """
        with self.assertRaises(ValueError):
            proxiss.ProxiPCA(n_components=self.n_components, k=self.k, num_threads=0)
        
        with self.assertRaises(ValueError):
            proxiss.ProxiPCA(n_components=self.n_components, k=self.k, num_threads=-1)

    def test_15_invalid_objective_function(self):
        """Test constructor with invalid objective function.

        Verifies that an unsupported objective function raises RuntimeError.
        """
        with self.assertRaises(RuntimeError):
            proxiss.ProxiPCA(
                n_components=self.n_components,
                k=self.k,
                num_threads=self.num_threads,
                objective_function="invalid"
            )

    def test_16_setter_getter_k(self):
        """Test set_k and get_k methods."""
        pca = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        
        # Test initial value
        self.assertEqual(pca.get_k(), self.k)
        
        # Test setting new values
        pca.set_k(5)
        self.assertEqual(pca.get_k(), 5)
        
        pca.set_k(10)
        self.assertEqual(pca.get_k(), 10)
        
        # Test invalid values
        with self.assertRaises(ValueError):
            pca.set_k(0)
        
        with self.assertRaises(ValueError):
            pca.set_k(-1)

    def test_17_setter_getter_num_threads(self):
        """Test set_num_threads and get_num_threads methods."""
        pca = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        
        # Test initial value
        self.assertEqual(pca.get_num_threads(), self.num_threads)
        
        # Test setting new values
        pca.set_num_threads(4)
        self.assertEqual(pca.get_num_threads(), 4)
        
        pca.set_num_threads(8)
        self.assertEqual(pca.get_num_threads(), 8)
        
        # Test invalid values
        with self.assertRaises(ValueError):
            pca.set_num_threads(0)
        
        with self.assertRaises(ValueError):
            pca.set_num_threads(-1)

    def test_18_setters_affect_search_behavior(self):
        """Test that changing k affects search results."""
        pca = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=2,
            num_threads=self.num_threads
        )
        pca.fit_transform_index(self.embeddings)
        
        # Test with k=2
        result_k2 = pca.find_indices(self.query_vector)
        self.assertEqual(len(result_k2), 2)
        
        # Change k to 3 and test
        pca.set_k(3)
        result_k3 = pca.find_indices(self.query_vector)
        self.assertEqual(len(result_k3), 3)
        
        # Change k to 1 and test
        pca.set_k(1)
        result_k1 = pca.find_indices(self.query_vector)
        self.assertEqual(len(result_k1), 1)

    def test_19_fit_with_empty_data(self):
        """Test fit_transform_index with empty data raises an error."""
        pca = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        with self.assertRaises(RuntimeError):
            pca.fit_transform_index([])

    def test_20_n_components_larger_than_features(self):
        """Test that n_components larger than feature dimensions still works.

        PCA should handle this gracefully by using min(n_components, n_features).
        """
        large_n_components = self.num_features + 10
        pca = proxiss.ProxiPCA(
            n_components=large_n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        # This should work - PCA will use actual number of features
        pca.fit_transform_index(self.embeddings)
        self.assertTrue(pca.is_fitted())

    def test_21_dimensionality_reduction_effect(self):
        """Test that PCA actually reduces dimensionality and queries work correctly.

        Verifies that high-dimensional data can be reduced and queried.
        """
        # Create higher dimensional data
        high_dim_embeddings = np.random.randn(50, 100).astype(np.float32)
        n_components_small = 10
        
        pca = proxiss.ProxiPCA(
            n_components=n_components_small,
            k=5,
            num_threads=2
        )
        pca.fit_transform_index(high_dim_embeddings)
        
        # Query with original high-dimensional vector
        query = high_dim_embeddings[0]
        indices = pca.find_indices(query)
        
        self.assertEqual(len(indices), 5)
        self.assertIn(0, indices)  # Query vector should find itself

    def test_22_different_distance_metrics(self):
        """Test that all distance metrics (l2, l1, cos) work correctly."""
        for obj in ["l2", "l1", "cos"]:
            pca = proxiss.ProxiPCA(
                n_components=self.n_components,
                k=self.k,
                num_threads=self.num_threads,
                objective_function=obj
            )
            pca.fit_transform_index(self.embeddings)
            indices = pca.find_indices(self.query_vector)
            self.assertEqual(len(indices), self.k)
            self.assertIn(0, indices)

    def test_23_batch_query_consistency(self):
        """Test that batch queries produce consistent results with single queries."""
        pca = proxiss.ProxiPCA(
            n_components=self.n_components,
            k=self.k,
            num_threads=self.num_threads
        )
        pca.fit_transform_index(self.embeddings)
        
        # Single query
        single_result = pca.find_indices(self.query_vector)
        
        # Batch query with same vector
        batch_result = pca.find_indices_batched(np.array([self.query_vector]))
        
        # Results should match
        np.testing.assert_array_equal(single_result, batch_result[0])


if __name__ == "__main__":
    unittest.main()
