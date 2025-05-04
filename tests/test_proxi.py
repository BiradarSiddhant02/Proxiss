import unittest
import numpy as np
from proxi import Proxi

class TestProxi(unittest.TestCase):
    def setUp(self):
        # Create a small dataset
        self.data = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0]
        ]
        self.doc_ids = ["a", "b", "c", "d"]  # must match data length

        self.query = [1.5, 1.5]
        self.query_batched = np.array([
            [1.5, 1.5],
            [0.0, 0.0],
            [3.0, 3.0]
        ], dtype=np.float32)

        self.k = 2
        self.proxi = Proxi(k=self.k, num_threads=1, objective_function="l2")
        self.proxi.index_data(self.data, self.doc_ids)

    def test_find_indices(self):
        indices = self.proxi.find_indices(self.query)
        self.assertEqual(len(indices), self.k)
        self.assertIsInstance(indices[0], int)

    def test_find_docs(self):
        docs = self.proxi.find_docs(self.query)
        self.assertEqual(len(docs), self.k)
        self.assertTrue(all(isinstance(doc, str) for doc in docs))

    def test_find_indices_batched(self):
        indices_batch = self.proxi.find_indices_batched(self.query_batched)
        self.assertEqual(len(indices_batch), len(self.query_batched))
        for result in indices_batch:
            self.assertEqual(len(result), self.k)

    def test_find_docs_batched(self):
        docs_batch = self.proxi.find_docs_batched(self.query_batched)
        self.assertEqual(len(docs_batch), len(self.query_batched))
        for doc in docs_batch:
            self.assertEqual(len(doc), self.k)

    def test_invalid_shape(self):
        with self.assertRaises(RuntimeError):
            self.proxi.find_indices_batched(np.array([1.0, 2.0], dtype=np.float32))

if __name__ == "__main__":
    unittest.main()
