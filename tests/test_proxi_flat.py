import unittest
import os
import shutil
import numpy as np
import proxiss


class TestProxiFlat(unittest.TestCase):
    """Test suite for the ProxiFlat class and its Python bindings."""

    def setUp(self):
        """Set up test fixtures before each test method.

        Initializes common parameters (k, num_threads, dimensions) and sample data
        (embeddings, documents, query vectors). Also creates a temporary directory
        for save/load tests.
        """
        self.k = 2
        self.num_threads = 1
        self.num_samples = 5
        self.num_features = 3

        # Sample data
        self.embeddings = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 3.1],
                [5.0, 6.0, 7.0],
                [5.1, 6.1, 7.1],
                [0.1, 0.2, 0.3],
            ],
            dtype=np.float32,
        )

        self.query_vector_exact = [1.0, 2.0, 3.0]  # Exact match for the first embedding
        self.query_vector_near = [1.05, 2.05, 3.05]  # Near the first two embeddings

        self.batched_queries = np.array(
            [
                self.query_vector_exact,
                [5.0, 6.0, 7.0],  # Exact match for the third embedding
                self.query_vector_near,
            ],
            dtype=np.float32,
        )

        # Temporary directory for save/load tests
        self.temp_dir = "temp_test_proxi_data"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.save_directory = self.temp_dir
        self.save_file_path = os.path.join(self.temp_dir, "data.bin")

        # Convert numpy arrays to lists for the new implementation
        self.embeddings_list = self.embeddings.tolist()
        self.batched_queries_list = self.batched_queries.tolist()

    def tearDown(self):
        """Tear down test fixtures after each test method.

        Removes the temporary directory created for save/load tests.
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _common_index_and_validate(self, index_instance, embeddings_to_index):
        """Helper method to index data and perform a basic validation.

        Args:
            index_instance: An instance of proxi.ProxiFlat.
            embeddings_to_index: List of lists of embeddings to index.
        """
        index_instance.index_data(embeddings_to_index)
        # Basic check after indexing (more detailed checks in specific query tests)
        self.assertIsNotNone(index_instance)  # Placeholder, real checks in query methods

    def test_01_constructor_and_index_l2(self):
        """Test parameterized constructor and indexing with L2 distance (Euclidean).

        Ensures that a ProxiFlat object can be created with k, num_threads, and 'l2'
        objective, and that data can be indexed successfully.
        """
        idx = proxiss.ProxiFlat(self.k, self.num_threads, "l2")
        self._common_index_and_validate(idx, self.embeddings_list)
        self.assertIsNotNone(idx)

    def test_02_find_indices_single_l2(self):
        """Test find_indices (single query) with L2 distance.

        Verifies that querying with an exact match and a near match returns the
        correct indices and number of neighbours (k).
        """
        idx = proxiss.ProxiFlat(self.k, self.num_threads, "l2")
        idx.index_data(self.embeddings_list)

        indices = idx.find_indices(self.query_vector_exact)
        self.assertEqual(len(indices), self.k)
        self.assertIn(0, indices)  # Expecting index 0 (exact match)

        indices_near = idx.find_indices(self.query_vector_near)
        self.assertEqual(len(indices_near), self.k)
        self.assertTrue(0 in indices_near and 1 in indices_near)  # Expecting 0 and 1



    def test_04_find_indices_batched_l2(self):
        """Test find_indices_batched with L2 distance.

        Ensures that batched queries for indices return the correct results for multiple
        input queries, including exact and near matches.
        """
        idx = proxiss.ProxiFlat(self.k, self.num_threads, "l2")
        idx.index_data(self.embeddings_list)

        results = idx.find_indices_batched(self.batched_queries_list)
        self.assertEqual(len(results), len(self.batched_queries_list))
        self.assertEqual(len(results[0]), self.k)

        # Query 1 (exact match for self.embeddings[0])
        self.assertIn(0, results[0])
        # Query 2 (exact match for self.embeddings[2])
        self.assertIn(2, results[1])
        # Query 3 (near self.embeddings[0] and self.embeddings[1])
        self.assertTrue(0 in results[2] and 1 in results[2])



    def test_05_insert_data_l2(self):
        """Test insert_data after initial indexing with L2 distance.

        Verifies that a new embedding can be inserted into an
        existing index and then successfully retrieved.
        """
        idx = proxiss.ProxiFlat(self.k, self.num_threads, "l2")
        idx.index_data(self.embeddings_list)

        new_embedding = [10.0, 11.0, 12.0]
        idx.insert_data(new_embedding)

        # Query for the newly inserted item
        indices = idx.find_indices(new_embedding)
        self.assertEqual(len(indices), self.k)
        # The new item should be its own closest neighbor
        self.assertIn(self.num_samples, indices)  # New item is at index num_samples

    def test_06_save_and_load_l2(self):
        """Test save and load functionality with L2 distance.

        Covers saving an indexed ProxiFlat object to a file and then loading it back
        using the instance `load_state` method. Verifies that the
        loaded index returns correct query results.
        """
        idx_orig = proxiss.ProxiFlat(self.k, self.num_threads, "l2")
        idx_orig.index_data(self.embeddings_list)
        idx_orig.save_state(self.save_directory)  # Pass the directory to save

        # Test loading via instance method - needs full file path
        idx_loaded_method = proxiss.ProxiFlat(self.k, self.num_threads, "l2")  # Create empty
        idx_loaded_method.load_state(self.save_file_path)
        indices_method = idx_loaded_method.find_indices(self.query_vector_exact)
        self.assertEqual(len(indices_method), self.k)
        self.assertIn(0, indices_method)

        # Check if K, num_features, num_samples are loaded correctly
        # Note: ProxiFlat class would need getters for these or we infer from behavior
        self.assertEqual(len(idx_loaded_method.find_indices(self.query_vector_exact)), self.k)

    def test_07_constructor_and_index_l1(self):
        """Test parameterized constructor and indexing with L1 distance (Manhattan).

        Ensures that a ProxiFlat object can be created and data indexed using the
        'l1' distance metric.
        """
        idx = proxiss.ProxiFlat(self.k, self.num_threads, "l1")
        self._common_index_and_validate(idx, self.embeddings_list)

        indices = idx.find_indices(self.query_vector_exact)
        self.assertEqual(len(indices), self.k)
        self.assertIn(0, indices)

    def test_08_edge_case_empty_index_data(self):
        """Test indexing with empty embeddings list.

        Verifies that attempting to index with empty data raises a RuntimeError,
        as per current C++ implementation requiring non-empty data for index_data.
        """
        idx = proxiss.ProxiFlat(self.k, self.num_threads, "l2")
        with self.assertRaises(RuntimeError):  # Or specific pybind11 exception if mapped
            idx.index_data([])



    def test_09_edge_case_query_before_index(self):
        """Test querying (find_indices) before any data has been indexed.

        Verifies that a RuntimeError is raised if a query is attempted on an empty,
        un-indexed ProxiFlat instance.
        """
        idx = proxiss.ProxiFlat(self.k, self.num_threads, "l2")
        with self.assertRaises(RuntimeError):  # Adjust if a different error is expected
            idx.find_indices(self.query_vector_exact)

    def test_10_save_before_index(self):
        """Test saving the model before any data is indexed.

        Verifies that attempting to save an un-indexed ProxiFlat instance raises a
        RuntimeError, as the C++ `save` method checks `m_is_indexed`.
        """
        idx = proxiss.ProxiFlat(self.k, self.num_threads, "l2")
        with self.assertRaises(RuntimeError):  # Expecting an error as per C++ checks
            idx.save_state(self.save_directory)  # Pass the directory to save

    def test_11_load_invalid_path(self):
        """Test loading from a non-existent or invalid file path.

        Ensures that attempting to load data from an invalid path raises a RuntimeError,
        both via the constructor and the instance `load_state` method.
        """
        non_existent_file = os.path.join(self.temp_dir, "non_existent_file.bin")
        idx = proxiss.ProxiFlat(self.k, self.num_threads, "l2")
        with self.assertRaises(RuntimeError):
            idx.load_state(non_existent_file)

    def test_12_constructor_invalid_objective(self):
        """Test constructor with an invalid objective function string.

        Verifies that providing an unsupported string for the objective function
        (e.g., "invalid_function") to the ProxiFlat constructor raises a RuntimeError
        (mapped from std::invalid_argument in C++).
        """
        with self.assertRaises(RuntimeError):  # Or std::invalid_argument if mapped
            proxiss.ProxiFlat(self.k, self.num_threads, "invalid_function")


if __name__ == "__main__":
    unittest.main()
