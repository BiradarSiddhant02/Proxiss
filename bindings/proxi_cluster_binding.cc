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

#include "proxi_cluster.h"
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(proxi_cluster_cpp, m) {
    m.doc() = "ProxiCluster: A C++ library for optimized neighbour search";

    py::class_<ProxiCluster>(m, "ProxiCluster")
        .def(py::init<size_t, size_t, size_t, size_t, std::string>(), py::arg("k"), py::arg("n"),
             py::arg("t"), py::arg("it"), py::arg("distance_function") = "l2")
        .def("index_data",
             [](ProxiCluster &self,
                py::array_t<float, py::array::c_style | py::array::forcecast> embeddings,
                py::list documents,
                py::array_t<float, py::array::c_style | py::array::forcecast> lower_bounds,
                py::array_t<float, py::array::c_style | py::array::forcecast> upper_bounds) {
                 std::vector<std::vector<float>> cpp_embeddings;
                 if (embeddings.ndim() == 2) {
                     cpp_embeddings.resize(embeddings.shape(0));
                     const float *embeddings_data_ptr = embeddings.data();
                     for (size_t i = 0; i < embeddings.shape(0); ++i) {
                         cpp_embeddings[i].resize(embeddings.shape(1));
                         if (embeddings.shape(1) >
                             0) { // Ensure there's something to copy for this row
                             std::memcpy(cpp_embeddings[i].data(),
                                         embeddings_data_ptr + i * embeddings.shape(1),
                                         embeddings.shape(1) * sizeof(float));
                         }
                     }
                 } else if (embeddings.ndim() == 1 && embeddings.shape(0) == 0) {
                     // Handle empty np.array([]) case, cpp_embeddings remains empty.
                 } else if (embeddings.ndim() == 2 && embeddings.shape(0) == 0 &&
                            embeddings.shape(1) >= 0) { // Allow (0,D)
                     // Handle empty np.array([[]]) or np.empty((0,D)) case, cpp_embeddings
                     // remains empty.
                 }
                 // Removed explicit error throw for embeddings dimensions

                 std::vector<std::string> cpp_documents;
                 cpp_documents.reserve(documents.size());
                 for (const auto &item : documents) { // Iterate over py::list
                     cpp_documents.push_back(item.cast<std::string>());
                 }

                 // Convert lower_bounds numpy array to std::vector<float>
                 std::vector<float> cpp_lower_bounds;
                 if (lower_bounds.size() > 0) {
                     cpp_lower_bounds.resize(lower_bounds.size());
                     const float *lower_bounds_ptr = lower_bounds.data();
                     std::memcpy(cpp_lower_bounds.data(), lower_bounds_ptr,
                                 lower_bounds.size() * sizeof(float));
                 }

                 // Convert upper_bounds numpy array to std::vector<float>
                 std::vector<float> cpp_upper_bounds;
                 if (upper_bounds.size() > 0) {
                     cpp_upper_bounds.resize(upper_bounds.size());
                     const float *upper_bounds_ptr = upper_bounds.data();
                     std::memcpy(cpp_upper_bounds.data(), upper_bounds_ptr,
                                 upper_bounds.size() * sizeof(float));
                 }

                 self.index_data(cpp_embeddings, cpp_documents, cpp_lower_bounds, cpp_upper_bounds);
             });
}