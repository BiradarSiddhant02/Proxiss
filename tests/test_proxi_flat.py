import unittest
import os
import shutil
import numpy as np
import proxi  # Assuming 'proxi' is the name of your compiled module


class TestProxiFlat(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures, if any."""
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
        self.documents = [f"doc_{i}" for i in range(self.num_samples)]

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

    def tearDown(self):
        """Tear down test fixtures, if any."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _common_index_and_validate(
        self, index_instance, embeddings_to_index, docs_to_index
    ):
        index_instance.index_data(embeddings_to_index.tolist(), docs_to_index)
        # Basic check after indexing (more detailed checks in specific query tests)
        self.assertIsNotNone(
            index_instance
        )  # Placeholder, real checks in query methods

    def test_01_constructor_and_index_l2(self):
        """Test parameterized constructor and indexing with L2 distance."""
        idx = proxi.ProxiFlat(self.k, self.num_threads, "l2")
        self._common_index_and_validate(idx, self.embeddings, self.documents)
        self.assertIsNotNone(idx)

    def test_02_find_indices_single_l2(self):
        """Test find_indices (single query) with L2 distance."""
        idx = proxi.ProxiFlat(self.k, self.num_threads, "l2")
        idx.index_data(self.embeddings.tolist(), self.documents)

        indices = idx.find_indices(self.query_vector_exact)
        self.assertEqual(len(indices), self.k)
        self.assertIn(0, indices)  # Expecting index 0 (exact match)

        indices_near = idx.find_indices(self.query_vector_near)
        self.assertEqual(len(indices_near), self.k)
        self.assertTrue(0 in indices_near and 1 in indices_near)  # Expecting 0 and 1

    def test_03_find_docs_single_l2(self):
        """Test find_docs (single query) with L2 distance."""
        idx = proxi.ProxiFlat(self.k, self.num_threads, "l2")
        idx.index_data(self.embeddings.tolist(), self.documents)

        docs = idx.find_docs(self.query_vector_exact)
        self.assertEqual(len(docs), self.k)
        self.assertIn(self.documents[0], docs)

        docs_near = idx.find_docs(self.query_vector_near)
        self.assertEqual(len(docs_near), self.k)
        self.assertIn(self.documents[0], docs_near)
        self.assertIn(self.documents[1], docs_near)

    def test_04_find_indices_batched_l2(self):
        """Test find_indices_batched with L2 distance."""
        idx = proxi.ProxiFlat(self.k, self.num_threads, "l2")
        idx.index_data(self.embeddings.tolist(), self.documents)

        results = idx.find_indices_batched(self.batched_queries)
        self.assertEqual(len(results), self.batched_queries.shape[0])
        self.assertEqual(len(results[0]), self.k)

        # Query 1 (exact match for self.embeddings[0])
        self.assertIn(0, results[0])
        # Query 2 (exact match for self.embeddings[2])
        self.assertIn(2, results[1])
        # Query 3 (near self.embeddings[0] and self.embeddings[1])
        self.assertTrue(0 in results[2] and 1 in results[2])

    def test_05_find_docs_batched_l2(self):
        """Test find_docs_batched with L2 distance."""
        idx = proxi.ProxiFlat(self.k, self.num_threads, "l2")
        idx.index_data(self.embeddings.tolist(), self.documents)

        results = idx.find_docs_batched(self.batched_queries)
        self.assertEqual(len(results), self.batched_queries.shape[0])
        self.assertEqual(len(results[0]), self.k)

        # Query 1
        self.assertIn(self.documents[0], results[0])
        # Query 2
        self.assertIn(self.documents[2], results[1])
        # Query 3
        self.assertIn(self.documents[0], results[2])
        self.assertIn(self.documents[1], results[2])

    def test_06_insert_data_l2(self):
        """Test insert_data after initial indexing with L2 distance."""
        idx = proxi.ProxiFlat(self.k, self.num_threads, "l2")
        idx.index_data(self.embeddings.tolist(), self.documents)

        new_embedding = [10.0, 11.0, 12.0]
        new_doc = "new_doc_inserted"
        idx.insert_data(new_embedding, new_doc)

        # Query for the newly inserted item
        indices = idx.find_indices(new_embedding)
        self.assertEqual(len(indices), self.k)
        # The new item should be its own closest neighbor
        self.assertIn(self.num_samples, indices)  # New item is at index num_samples

        docs = idx.find_docs(new_embedding)
        self.assertIn(new_doc, docs)

    def test_07_save_and_load_l2(self):
        """Test save and load functionality (via constructor and method) with L2."""
        idx_orig = proxi.ProxiFlat(self.k, self.num_threads, "l2")
        idx_orig.index_data(self.embeddings.tolist(), self.documents)
        idx_orig.save(self.save_directory)  # Pass the directory to save

        # Test loading via constructor - needs full file path
        idx_loaded_constructor = proxi.ProxiFlat(self.save_file_path)
        docs_constructor = idx_loaded_constructor.find_docs(self.query_vector_exact)
        self.assertEqual(len(docs_constructor), self.k)
        self.assertIn(self.documents[0], docs_constructor)

        # Test loading via instance method - needs full file path
        idx_loaded_method = proxi.ProxiFlat(
            self.k, self.num_threads, "l2"
        )  # Create empty
        idx_loaded_method.load(self.save_file_path)
        docs_method = idx_loaded_method.find_docs(self.query_vector_exact)
        self.assertEqual(len(docs_method), self.k)
        self.assertIn(self.documents[0], docs_method)

        # Check if K, num_features, num_samples are loaded correctly
        # Note: ProxiFlat class would need getters for these or we infer from behavior
        self.assertEqual(
            len(idx_loaded_constructor.find_indices(self.query_vector_exact)), self.k
        )

    def test_08_constructor_and_index_l1(self):
        """Test with L1 distance."""
        idx = proxi.ProxiFlat(self.k, self.num_threads, "l1")
        self._common_index_and_validate(idx, self.embeddings, self.documents)

        indices = idx.find_indices(self.query_vector_exact)
        self.assertEqual(len(indices), self.k)
        self.assertIn(0, indices)

    def test_09_constructor_and_index_cos(self):
        """Test with Cosine distance."""
        # Cosine distance needs normalized vectors for intuitive nearest neighbor with non-normalized inputs
        # For simplicity, we'll use the same data but acknowledge results might differ from L2/L1
        # A more robust test would use normalized data or data where cosine similarity is clear.
        idx = proxi.ProxiFlat(self.k, self.num_threads, "cos")
        self._common_index_and_validate(idx, self.embeddings, self.documents)

        indices = idx.find_indices(self.query_vector_exact)
        self.assertEqual(len(indices), self.k)
        # With cosine, the exact vector [1,2,3] should be closest to itself.
        self.assertIn(0, indices)

    def test_10_edge_case_empty_index_data(self):
        """Test indexing with empty data."""
        idx = proxi.ProxiFlat(self.k, self.num_threads, "l2")
        with self.assertRaises(
            RuntimeError
        ):  # Or specific pybind11 exception if mapped
            idx.index_data([], [])

    def test_11_edge_case_mismatched_embeddings_docs(self):
        """Test indexing with mismatched embeddings and documents count."""
        idx = proxi.ProxiFlat(self.k, self.num_threads, "l2")
        with self.assertRaises(RuntimeError):
            idx.index_data(self.embeddings.tolist(), self.documents[:1])

    def test_12_edge_case_query_before_index(self):
        """Test querying before indexing."""
        idx = proxi.ProxiFlat(self.k, self.num_threads, "l2")
        # Behavior might depend on C++ implementation: could error or return empty/undefined.
        # Assuming it should raise an error or return empty and not crash.
        # If ProxiFlat internally checks m_is_indexed for queries:
        with self.assertRaises(RuntimeError):  # Adjust if a different error is expected
            idx.find_indices(self.query_vector_exact)
        # If it doesn't error but returns empty:
        # results = idx.find_indices(self.query_vector_exact)
        # self.assertEqual(len(results), 0) # or self.k if it returns k empty slots

    def test_13_edge_case_inconsistent_feature_dims_index(self):
        """Test indexing data with inconsistent feature dimensions."""
        idx = proxi.ProxiFlat(self.k, self.num_threads, "l2")
        bad_embeddings = self.embeddings.tolist()
        bad_embeddings[1] = [1.0, 2.0]  # Incorrect dimension for one embedding
        with self.assertRaises(RuntimeError):
            idx.index_data(bad_embeddings, self.documents)

    def test_14_edge_case_inconsistent_feature_dims_query(self):
        """Test querying with inconsistent feature dimensions."""
        idx = proxi.ProxiFlat(self.k, self.num_threads, "l2")
        idx.index_data(self.embeddings.tolist(), self.documents)
        bad_query = [1.0, 2.0]  # Incorrect dimension
        with self.assertRaises(RuntimeError):
            idx.find_indices(bad_query)

        bad_batched_queries = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0]], dtype=object
        )  # Mixed dims
        # Batched query check might be at numpy conversion or deeper in C++
        with self.assertRaises(
            RuntimeError
        ):  # Or TypeError from pybind11 if conversion fails early
            idx.find_indices_batched(bad_batched_queries)

    def test_15_save_before_index(self):
        """Test saving the model before any data is indexed."""
        idx = proxi.ProxiFlat(self.k, self.num_threads, "l2")
        with self.assertRaises(RuntimeError):  # Expecting an error as per C++ checks
            idx.save(self.save_directory)  # Pass the directory to save

    def test_16_load_invalid_path(self):
        """Test loading from a non-existent or invalid file path."""
        non_existent_file = os.path.join(self.temp_dir, "non_existent_file.bin")
        with self.assertRaises(RuntimeError):
            proxi.ProxiFlat(non_existent_file)

        idx = proxi.ProxiFlat(self.k, self.num_threads, "l2")
        with self.assertRaises(RuntimeError):
            idx.load(non_existent_file)

    def test_17_constructor_invalid_objective(self):
        """Test constructor with an invalid objective function string."""
        with self.assertRaises(RuntimeError):  # Or std::invalid_argument if mapped
            proxi.ProxiFlat(self.k, self.num_threads, "invalid_function")


if __name__ == "__main__":
    unittest.main()
