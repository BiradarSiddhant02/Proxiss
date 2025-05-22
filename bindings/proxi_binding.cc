/*
 * Copyright 2025 Siddhant Biradar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "proxi_flat.h"
#include <cstring>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(proxi_cpp, m) {
    m.doc() = "Proxi: A C++ library for fast approximate nearest neighbour search, with Python bindings.";

    py::class_<ProxiFlat>(m, "ProxiFlat",
                         "Main class for ProxiFlat, providing functionality for indexing embeddings and performing nearest neighbour searches.")
        // Parameterized Constructor
        .def(py::init<size_t, size_t, const std::string &>(),
             py::arg("k"),
             py::arg("num_threads"),
             py::arg("objective_function") = "l2",
             "Constructs a ProxiFlat instance.\n\n"
             "Args:\n"
             "    k (int): Number of nearest neighbours to find.\n"
             "    num_threads (int): Number of threads for parallel operations.\n"
             "    objective_function (str): Distance metric ('l1', 'l2', 'cos'). Defaults to 'l2'.")

        // Load Constructor
        .def(py::init<const std::string &>(),
             py::arg("file_path"),
             "Constructs a ProxiFlat instance by loading data from a saved file.\n\n"
             "Args:\n"
             "    file_path (str): Path to the serialized ProxiFlat data file (e.g., 'data.bin').")

        // Updated index_data binding (reverted to not take num_features explicitly)
        .def("index_data",
             [](ProxiFlat &self,
                py::array_t<float, py::array::c_style | py::array::forcecast> embeddings_arr,
                const std::vector<std::string> &documents) {

                 std::vector<std::vector<float>> cpp_embeddings;

                 if (embeddings_arr.size() == 0) {
                     // Handles np.array([]) or np.empty((0, D))
                     // cpp_embeddings remains empty. The C++ method will handle this.
                     // If documents are provided, C++ method must handle empty embeddings with non-empty docs.
                 } else {
                     // If embeddings are provided, they must be a 2D array.
                     if (embeddings_arr.ndim() != 2) {
                         throw py::type_error("Embeddings: Expected a 2D NumPy array (N, D) if not empty.");
                     }

                     const size_t num_embeddings_rows = embeddings_arr.shape(0);
                     const size_t dim_cols = embeddings_arr.shape(1); // Dimension is inferred here

                     // Validate consistency between number of embeddings and documents if documents are provided
                     // and embeddings are also provided.
                     if (num_embeddings_rows > 0 && !documents.empty() && num_embeddings_rows != documents.size()) {
                        throw py::value_error(
                            "Number of embeddings rows must match the number of documents if both are non-empty and embeddings are provided.");
                     }

                     const float *embeddings_data_ptr = embeddings_arr.data();
                     cpp_embeddings.resize(num_embeddings_rows, std::vector<float>(dim_cols));
                     for (size_t i = 0; i < num_embeddings_rows; ++i) {
                         std::memcpy(cpp_embeddings[i].data(), embeddings_data_ptr + i * dim_cols, dim_cols * sizeof(float));
                     }
                 }

                 self.index_data(cpp_embeddings, documents);
             },
             py::arg("embeddings"), py::arg("documents"),
             "Indexes the provided embeddings and documents.\n\n"
             "The embedding dimension is inferred from the shape of the 'embeddings' NumPy array.\n"
             "Args:\n"
             "    embeddings (numpy.ndarray): A 2D NumPy array of floats (N, D), where N is the number of samples\n"
             "                              and D is the embedding dimension. Can be an empty array if documents is also empty.\n"
             "    documents (list[str]): A list of N strings, corresponding to each embedding.")

        // --- Single-query methods ---
        .def("find_indices",
             py::overload_cast<const std::vector<float> &>(&ProxiFlat::find_indices),
             py::arg("query"),
             "Finds the indices of the K nearest neighbours for a single query embedding.\n\n"
             "Args:\n"
             "    query (list[float]): The query embedding.\n\n"
             "Returns:\n"
             "    list[int]: Indices of the K nearest neighbours.")
        .def("find_docs", py::overload_cast<const std::vector<float> &>(&ProxiFlat::find_docs),
             py::arg("query"),
             "Finds the K nearest documents for a single query embedding.\n\n"
             "Args:\n"
             "    query (list[float]): The query embedding.\n\n"
             "Returns:\n"
             "    list[str]: The K nearest documents.")

        // --- Batched-query: NumPy to vector<vector<float>> via memcpy ---
        .def("find_indices_batched",
            [](ProxiFlat &self, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                if (arr.ndim() != 2) {
                    throw std::runtime_error("Expected 2D array (N, D) for batched input");
                }

                const size_t n = arr.shape(0);
                const size_t d = arr.shape(1);
                const float *data = arr.data();

                std::vector<std::vector<float>> vecs(n, std::vector<float>(d));
                for (size_t i = 0; i < n; ++i) {
                    std::memcpy(vecs[i].data(), data + i * d, d * sizeof(float));
                }

                return self.find_indices(vecs);
            },
            py::arg("queries"),
            "Finds indices of K nearest neighbours for a batch of query embeddings.\n\n"
            "Args:\n"
            "    queries (numpy.ndarray): A 2D NumPy array of query embeddings (M, D).\n\n"
            "Returns:\n"
            "    list[list[int]]: For each query, a list of K nearest neighbour indices.")

        .def("find_docs_batched",
            [](ProxiFlat &self, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                if (arr.ndim() != 2) {
                    throw std::runtime_error("Expected 2D array (N, D) for batched input");
                }

                const size_t n = arr.shape(0);
                const size_t d = arr.shape(1);
                const float *data = arr.data();

                std::vector<std::vector<float>> vecs(n, std::vector<float>(d));
                for (size_t i = 0; i < n; ++i) {
                    std::memcpy(vecs[i].data(), data + i * d, d * sizeof(float));
                }

                return self.find_docs(vecs);
            },
            py::arg("queries"),
            "Finds K nearest documents for a batch of query embeddings.\n\n"
            "Args:\n"
            "    queries (numpy.ndarray): A 2D NumPy array of query embeddings (M, D).\n\n"
            "Returns:\n"
            "    list[list[str]]: For each query, a list of K nearest documents.")

        .def("insert_data", &ProxiFlat::insert_data,
             py::arg("embedding"), py::arg("text"),
             "Inserts a new embedding and its corresponding document into the index.\n\n"
             "Args:\n"
             "    embedding (list[float]): The embedding to insert.\n"
             "    text (str): The document text to insert.")

        .def("save", &ProxiFlat::save,
             py::arg("path"),
             "Saves the ProxiFlat index and data to a directory.\n\n"
             "A file named 'data.bin' will be created in the specified directory.\n"
             "Args:\n"
             "    path (str): The directory path to save the data file.")

        .def("load", &ProxiFlat::load,
             py::arg("path"),
             "Loads the ProxiFlat index and data from a file.\n\n"
             "Args:\n"
             "    path (str): Path to the serialized ProxiFlat data file (e.g., 'data.bin').");
}
