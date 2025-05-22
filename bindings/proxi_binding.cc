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
    m.doc() =
        "Proxi: A C++ library for fast approximate nearest neighbour search, with Python bindings.";

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

        // Updated index_data binding to accept std::vector<std::vector<float>> directly
        .def("index_data",
             py::overload_cast<const std::vector<std::vector<float>> &,
                               const std::vector<std::string> &>(&ProxiFlat::index_data),
             py::arg("embeddings"), py::arg("documents"),
             "Indexes the provided embeddings and documents.\n\n"
             "Args:\n"
             "    embeddings (list[list[float]]): A list of N embeddings, where each embedding is "
             "a list of D floats.\n"
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

        // --- Batched-query methods ---
        // Modified to accept std::vector<std::vector<float>> directly
        .def("find_indices_batched",
             py::overload_cast<const std::vector<std::vector<float>> &>(&ProxiFlat::find_indices),
             py::arg("queries"),
             "Finds indices of K nearest neighbours for a batch of query embeddings.\n\n"
             "Args:\n"
             "    queries (list[list[float]]): A 2D list of query embeddings (M, D).\n\n"
             "Returns:\n"
             "    list[list[int]]: For each query, a list of K nearest neighbour indices.")

        // Modified to accept std::vector<std::vector<float>> directly
        .def("find_docs_batched",
             py::overload_cast<const std::vector<std::vector<float>> &>(&ProxiFlat::find_docs),
             py::arg("queries"),
             "Finds K nearest documents for a batch of query embeddings.\n\n"
             "Args:\n"
             "    queries (list[list[float]]): A 2D list of query embeddings (M, D).\n\n"
             "Returns:\n"
             "    list[list[str]]: For each query, a list of K nearest documents.")

        .def("insert_data", &ProxiFlat::insert_data, py::arg("embedding"), py::arg("text"),
             "Inserts a new embedding and its corresponding document into the index.\n\n"
             "Args:\n"
             "    embedding (list[float]): The embedding to insert.\n"
             "    text (str): The document text to insert.")

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
