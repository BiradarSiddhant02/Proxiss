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

namespace py = pybind11;

PYBIND11_MODULE(proxi, m) {
    py::class_<ProxiFlat>(m, "ProxiFlat")
        // Parameterized Constructor
        .def(py::init<size_t, size_t, const std::string &>(), py::arg("k"),
             py::arg("num_threads"), py::arg("objective_function") = "l2")

        // Load Constructor
        .def(py::init<const std::string &>(), py::arg("file_path"),
             "Construct ProxiFlat by loading serialised data.")

        .def("index_data", &ProxiFlat::index_data)

        // --- Single-query methods ---
        .def("find_indices", py::overload_cast<const std::vector<float> &>(
                                 &ProxiFlat::find_indices))
        .def("find_docs", py::overload_cast<const std::vector<float> &>(
                              &ProxiFlat::find_docs))

        // --- Batched-query: NumPy to vector<vector<float>> via memcpy ---
        .def("find_indices_batched",
             [](ProxiFlat &self,
                py::array_t<float, py::array::c_style | py::array::forcecast>
                    arr) {
                 if (arr.ndim() != 2) {
                     throw std::runtime_error(
                         "Expected 2D array (N, D) for batched input");
                 }

                 const ssize_t n = arr.shape(0);
                 const ssize_t d = arr.shape(1);
                 const float *data = arr.data();

                 std::vector<std::vector<float>> vecs(n, std::vector<float>(d));
                 for (ssize_t i = 0; i < n; ++i) {
                     std::memcpy(vecs[i].data(), data + i * d,
                                 d * sizeof(float));
                 }

                 return self.find_indices(vecs);
             })

        .def("find_docs_batched",
             [](ProxiFlat &self,
                py::array_t<float, py::array::c_style | py::array::forcecast>
                    arr) {
                 if (arr.ndim() != 2) {
                     throw std::runtime_error(
                         "Expected 2D array (N, D) for batched input");
                 }

                 const ssize_t n = arr.shape(0);
                 const ssize_t d = arr.shape(1);
                 const float *data = arr.data();

                 std::vector<std::vector<float>> vecs(n, std::vector<float>(d));
                 for (ssize_t i = 0; i < n; ++i) {
                     std::memcpy(vecs[i].data(), data + i * d,
                                 d * sizeof(float));
                 }

                 return self.find_docs(vecs);
             })

        .def("insert_data", &ProxiFlat::insert_data)

        .def("save", &ProxiFlat::save)

        .def("load", &ProxiFlat::load);
}
