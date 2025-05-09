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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>
#include "proxi_flat.h"

namespace py = pybind11;

PYBIND11_MODULE(proxi, m) {
    py::class_<ProxiFlat>(m, "ProxiFlat")
        .def(
            py::init<size_t, size_t, const std::string&>(), 
            py::arg("k"), 
            py::arg("num_threads"), 
            py::arg("objective_function") = "l2"
        )

        .def("index_data", &ProxiFlat::index_data, py::call_guard<py::gil_scoped_release>())

        // --- Single-query methods ---
        .def("find_indices", py::overload_cast<const std::vector<float>&>(&ProxiFlat::find_indices), py::call_guard<py::gil_scoped_release>())
        .def("find_docs", py::overload_cast<const std::vector<float>&>(&ProxiFlat::find_docs), py::call_guard<py::gil_scoped_release>())

        // --- Batched-query: NumPy to vector<vector<float>> via memcpy ---
        .def("find_indices_batched", [](ProxiFlat &self, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            if (arr.ndim() != 2) {
                throw std::runtime_error("Expected 2D array (N, D) for batched input");
            }

            const ssize_t n = arr.shape(0);
            const ssize_t d = arr.shape(1);
            const float* data = arr.data();

            // Call the new method that accepts a raw pointer
            return self.find_indices_batched_from_ptr(data, n, d);
        }, py::call_guard<py::gil_scoped_release>())

        .def("find_docs_batched", [](ProxiFlat &self, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            if (arr.ndim() != 2) {
                throw std::runtime_error("Expected 2D array (N, D) for batched input");
            }

            const ssize_t n = arr.shape(0);
            const ssize_t d = arr.shape(1);
            const float* data = arr.data();

            // Call the new method that accepts a raw pointer
            return self.find_docs_batched_from_ptr(data, n, d);
        }, py::call_guard<py::gil_scoped_release>())

        .def("insert_data", &ProxiFlat::insert_data, py::call_guard<py::gil_scoped_release>());
}
