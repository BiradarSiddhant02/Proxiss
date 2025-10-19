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

#include "proxi_pca.h"
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(proxi_pca_cpp, m) {
    m.doc() = "ProxiPCA: A library combining PCA dimensionality reduction with fast nearest neighbor search";

    py::class_<ProxiPCA>(m, "ProxiPCA",
                         "Main class for ProxiPCA, combining PCA with ProxiFlat for efficient "
                         "dimensionality reduction and nearest neighbor search.")
        // Parameterized Constructor
        .def(py::init<size_t, size_t, size_t, const std::string &>(),
             py::arg("n_components"),
             py::arg("k"),
             py::arg("num_threads"),
             py::arg("objective_function") = "l2",
             "Constructs a ProxiPCA instance.\n\n"
             "Args:\n"
             "    n_components (int): Number of PCA components (reduced dimensions).\n"
             "    k (int): Number of nearest neighbours to find.\n"
             "    num_threads (int): Number of threads for parallel operations.\n"
             "    objective_function (str): Distance metric ('l1', 'l2', 'cos'). Defaults to 'l2'.")

        .def(
            "fit_transform_index",
            [](ProxiPCA &self,
               py::array_t<float, py::array::c_style | py::array::forcecast> embeddings_arr) {
                std::vector<std::vector<float>> cpp_embeddings;
                if (embeddings_arr.ndim() == 2) {
                    cpp_embeddings.resize(embeddings_arr.shape(0));
                    const float *embeddings_data_ptr = embeddings_arr.data();
                    for (size_t i = 0; i < embeddings_arr.shape(0); ++i) {
                        cpp_embeddings[i].resize(embeddings_arr.shape(1));
                        if (embeddings_arr.shape(1) > 0) {
                            std::memcpy(cpp_embeddings[i].data(),
                                        embeddings_data_ptr + i * embeddings_arr.shape(1),
                                        embeddings_arr.shape(1) * sizeof(float));
                        }
                    }
                } else if (embeddings_arr.ndim() == 1 && embeddings_arr.shape(0) == 0) {
                    // Handle empty array
                } else if (embeddings_arr.ndim() == 2 && embeddings_arr.shape(0) == 0 &&
                           embeddings_arr.shape(1) >= 0) {
                    // Handle empty (0, D) array
                }

                self.fit_transform_index(cpp_embeddings);
            },
            py::arg("embeddings"),
            "Fits PCA on embeddings, transforms them to reduced dimensions, and indexes them.\n\n"
            "This method performs three operations:\n"
            "1. Fits PCA model on the input embeddings\n"
            "2. Transforms embeddings to reduced dimensions (n_components)\n"
            "3. Indexes the reduced embeddings for fast nearest neighbor search\n\n"
            "Args:\n"
            "    embeddings (numpy.ndarray): A 2D NumPy array of floats (N, D) where N is the "
            "number of samples and D is the original dimensionality.\n")

        // --- Single-query methods ---
        .def(
            "find_indices",
            [](ProxiPCA &self,
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
            "Finds the indices of the K nearest neighbours for a single query.\n\n"
            "The query is automatically reduced to n_components dimensions using the fitted PCA "
            "model before searching.\n\n"
            "Args:\n"
            "    query (numpy.ndarray): The 1D query vector with original dimensionality D.\n\n"
            "Returns:\n"
            "    list[int]: Indices of the K nearest neighbours.")

        // --- Batched-query methods ---
        .def(
            "find_indices_batched",
            [](ProxiPCA &self,
               py::array_t<float, py::array::c_style | py::array::forcecast> queries_arr) {
                if (queries_arr.ndim() != 2) {
                    throw std::runtime_error("Input queries array must be 2D (M, D)");
                }
                std::vector<std::vector<float>> cpp_queries(queries_arr.shape(0));
                const float *queries_data_ptr = queries_arr.data();
                for (size_t i = 0; i < queries_arr.shape(0); ++i) {
                    cpp_queries[i].resize(queries_arr.shape(1));
                    if (queries_arr.shape(1) > 0) {
                        std::memcpy(cpp_queries[i].data(),
                                    queries_data_ptr + i * queries_arr.shape(1),
                                    queries_arr.shape(1) * sizeof(float));
                    }
                }
                return self.find_indices_batched(cpp_queries);
            },
            py::arg("queries"),
            "Finds indices of K nearest neighbours for a batch of queries.\n\n"
            "All queries are automatically reduced to n_components dimensions using the fitted "
            "PCA model before searching.\n\n"
            "Args:\n"
            "    queries (numpy.ndarray): A 2D NumPy array of query vectors (M, D) where M is "
            "the number of queries and D is the original dimensionality.\n\n"
            "Returns:\n"
            "    list[list[int]]: For each query, a list of K nearest neighbour indices.")

        .def(
            "insert_data",
            [](ProxiPCA &self,
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
            "The embedding is automatically reduced to n_components dimensions using the fitted "
            "PCA model before insertion.\n\n"
            "Args:\n"
            "    embedding (numpy.ndarray): The 1D embedding to insert with original "
            "dimensionality D.\n")

        .def("save_state", &ProxiPCA::save_state, py::arg("directory_path"),
             "Saves the ProxiPCA state (PCA model and index) to a directory.\n\n"
             "Args:\n"
             "    directory_path (str): The directory path where state will be saved.")

        .def("load_state", &ProxiPCA::load_state, py::arg("directory_path"),
             "Loads the ProxiPCA state (PCA model and index) from a directory.\n\n"
             "Args:\n"
             "    directory_path (str): The directory path from where state will be loaded.")

        // Getters
        .def("get_k", &ProxiPCA::get_k,
             "Gets the current number of nearest neighbors.\n\n"
             "Returns:\n"
             "    int: Current value of k.")

        .def("get_num_threads", &ProxiPCA::get_num_threads,
             "Gets the current number of threads.\n\n"
             "Returns:\n"
             "    int: Current number of threads.")

        .def("get_n_components", &ProxiPCA::get_n_components,
             "Gets the number of PCA components (reduced dimensions).\n\n"
             "Returns:\n"
             "    int: Number of PCA components.")

        .def("is_fitted", &ProxiPCA::is_fitted,
             "Checks if the PCA model has been fitted.\n\n"
             "Returns:\n"
             "    bool: True if fitted, False otherwise.")

        // Setters
        .def("set_k", &ProxiPCA::set_k, py::arg("k"),
             "Sets the number of nearest neighbors to retrieve.\n\n"
             "Args:\n"
             "    k (int): The new number of neighbors (must be > 0).")

        .def("set_num_threads", &ProxiPCA::set_num_threads, py::arg("num_threads"),
             "Sets the number of threads for parallel operations.\n\n"
             "Args:\n"
             "    num_threads (int): The new number of threads (must be > 0).");
}
