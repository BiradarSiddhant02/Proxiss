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
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(proxi_flat_cpp, m) {
    m.doc() = "ProxiFlat: A C++ library for fast nearest neighbour search";

    py::class_<ProxiFlat>(m, "ProxiFlat",
                          "Main class for ProxiFlat, providing functionality for indexing "
                          "embeddings and performing nearest neighbour searches.")
        // Parameterized Constructor
        .def(py::init<size_t, size_t, const std::string &>(), py::arg("k"), py::arg("num_threads"),
             py::arg("objective_function") = "l2",
             "Constructs a ProxiFlat instance.\n\n"
             "Args:\n"
             "    k (int): Number of nearest neighbours to find.\n"
             "    num_threads (int): Number of threads for parallel operations.\n"
             "    objective_function (str): Distance metric ('l1', 'l2', 'cos'). Defaults to 'l2'.")

        .def(
            "index_data",
            [](ProxiFlat &self,
               py::array_t<float, py::array::c_style | py::array::forcecast> embeddings_arr) {
                std::vector<std::vector<float>> cpp_embeddings;
                if (embeddings_arr.ndim() == 2) {
                    cpp_embeddings.resize(embeddings_arr.shape(0));
                    const float *embeddings_data_ptr = embeddings_arr.data();
                    for (size_t i = 0; i < embeddings_arr.shape(0); ++i) {
                        cpp_embeddings[i].resize(embeddings_arr.shape(1));
                        if (embeddings_arr.shape(1) >
                            0) { // Ensure there's something to copy for this row
                            std::memcpy(cpp_embeddings[i].data(),
                                        embeddings_data_ptr + i * embeddings_arr.shape(1),
                                        embeddings_arr.shape(1) * sizeof(float));
                        }
                    }
                } else if (embeddings_arr.ndim() == 1 && embeddings_arr.shape(0) == 0) {
                    // Handle empty np.array([]) case, cpp_embeddings remains empty.
                } else if (embeddings_arr.ndim() == 2 && embeddings_arr.shape(0) == 0 &&
                           embeddings_arr.shape(1) >= 0) { // Allow (0,D)
                    // Handle empty np.array([[]]) or np.empty((0,D)) case, cpp_embeddings remains
                    // empty.
                }
                // Removed explicit error throw for embeddings_arr dimensions

                self.index_data(cpp_embeddings);
            },
            py::arg("embeddings"),
            "Indexes the provided embeddings.\n\n"
            "Args:\n"
            "    embeddings (numpy.ndarray): A 2D NumPy array of floats (N, D).\n")

        // --- Single-query methods ---
        .def(
            "find_indices",
            [](ProxiFlat &self,
               py::array_t<float, py::array::c_style | py::array::forcecast> query_arr) {
                if (query_arr.ndim() != 1) {
                    throw std::runtime_error("Input query array must be 1D");
                }
                std::vector<float> cpp_query(query_arr.shape(0));
                if (query_arr.shape(0) > 0) {
                    std::memcpy(cpp_query.data(), query_arr.data(),
                                query_arr.shape(0) * sizeof(float));
                }
                return self.find_indices(cpp_query);
            },
            py::arg("query"),
            "Finds the indices of the K nearest neighbours for a single query embedding.\n\n"
            "Args:\n"
            "    query (numpy.ndarray): The 1D query embedding.\n\n"
            "Returns:\n"
            "    list[int]: Indices of the K nearest neighbours.")
        // --- Batched-query methods ---
        .def(
            "find_indices_batched",
            [](ProxiFlat &self,
               py::array_t<float, py::array::c_style | py::array::forcecast> queries_arr) {
                if (queries_arr.ndim() != 2) {
                    throw std::runtime_error("Input queries array must be 2D (M, D)");
                }
                std::vector<std::vector<float>> cpp_queries(queries_arr.shape(0));
                const float *queries_data_ptr = queries_arr.data();
                for (size_t i = 0; i < queries_arr.shape(0); ++i) {
                    cpp_queries[i].resize(queries_arr.shape(1));
                    if (queries_arr.shape(1) > 0) { // Ensure there's something to copy for this row
                        std::memcpy(cpp_queries[i].data(),
                                    queries_data_ptr + i * queries_arr.shape(1),
                                    queries_arr.shape(1) * sizeof(float));
                    }
                }
                return self.find_indices(cpp_queries); // Calls overloaded ProxiFlat::find_indices
            },
            py::arg("queries"),
            "Finds indices of K nearest neighbours for a batch of query embeddings.\n\n"
            "Args:\n"
            "    queries (numpy.ndarray): A 2D NumPy array of query embeddings (M, D).\n\n"
            "Returns:\n"
            "    list[list[int]]: For each query, a list of K nearest neighbour indices.")

        .def(
            "insert_data",
            [](ProxiFlat &self,
               py::array_t<float, py::array::c_style | py::array::forcecast> embedding_arr) {
                if (embedding_arr.ndim() != 1) {
                    throw std::runtime_error("Input embedding array must be 1D");
                }
                std::vector<float> cpp_embedding(embedding_arr.shape(0));
                if (embedding_arr.shape(0) > 0) {
                    std::memcpy(cpp_embedding.data(), embedding_arr.data(),
                                embedding_arr.shape(0) * sizeof(float));
                }
                self.insert_data(cpp_embedding);
            },
            py::arg("embedding"),
            "Inserts a new embedding into the index.\n\n"
            "Args:\n"
            "    embedding (numpy.ndarray): The 1D embedding to insert.\n")

        .def("save_state", &ProxiFlat::save_state, py::arg("path"),
             "Saves the ProxiFlat index and data to a directory.\n\n"
             "A file named 'data.bin' will be created in the specified directory.\n"
             "Args:\n"
             "    path (str): The directory path to save the data file.")

        .def("load_state", &ProxiFlat::load_state, py::arg("path"),
             "Loads the ProxiFlat index and data from a file.\n\n"
             "Args:\n"
             "    path (str): Path to the serialized ProxiFlat data file (e.g., 'data.bin').");
}