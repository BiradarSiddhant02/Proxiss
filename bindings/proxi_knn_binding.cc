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

#include "proxi_knn.h"
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(proxi_knn_cpp, m) {
    m.doc() = "ProxiKNN: A library for fast KNN algorithm.";

    py::class_<ProxiKNN>(m, "ProxiKNN",
                         "Main class for ProxiKNN, providing functionality for KNN classification.")
        // Parameterized Constructor
        .def(py::init<size_t, size_t, const std::string &>(), py::arg("n_neighbours"),
             py::arg("n_jobs"), py::arg("distance_function") = "l2",
             "Constructs a ProxiKNN instance.\n\n"
             "Args:\n"
             "    n_neighbours (int): Number of nearest neighbours to find.\n"
             "    n_jobs (int): Number of threads for parallel operations.\n"
             "    distance_function (str): Distance metric ('l1', 'l2', 'cos'). Defaults to 'l2'.")

        .def(
            "fit",
            [](ProxiKNN &self,
               py::array_t<float, py::array::c_style | py::array::forcecast> features_arr,
               py::array_t<float, py::array::c_style | py::array::forcecast> labels_arr) {
                std::vector<std::vector<float>> cpp_features;
                if (features_arr.ndim() == 2) {
                    cpp_features.resize(features_arr.shape(0));
                    const float *features_data_ptr = features_arr.data();
                    for (size_t i = 0; i < features_arr.shape(0); ++i) {
                        cpp_features[i].resize(features_arr.shape(1));
                        if (features_arr.shape(1) > 0) {
                            std::memcpy(cpp_features[i].data(),
                                        features_data_ptr + i * features_arr.shape(1),
                                        features_arr.shape(1) * sizeof(float));
                        }
                    }
                } else if (features_arr.ndim() == 1 && features_arr.shape(0) == 0) {
                    // Handle empty np.array([]) case
                } else if (features_arr.ndim() == 2 && features_arr.shape(0) == 0 &&
                           features_arr.shape(1) >= 0) {
                    // Handle empty np.array([[]]) or np.empty((0,D)) case
                }

                std::vector<float> cpp_labels;
                if (labels_arr.ndim() == 1) {
                    cpp_labels.resize(labels_arr.shape(0));
                    const float *labels_data_ptr = labels_arr.data();
                    if (labels_arr.shape(0) > 0) {
                        std::memcpy(cpp_labels.data(), labels_data_ptr,
                                    labels_arr.shape(0) * sizeof(float));
                    }
                } else {
                    throw std::runtime_error("Labels array must be 1D");
                }

                self.fit(cpp_features, cpp_labels);
            },
            py::arg("features"), py::arg("labels"),
            "Trains the model on the provided feature vectors and their respective labels.\n\n"
            "Args:\n"
            "    features (numpy.ndarray): A 2D NumPy array of floats (N, D).\n"
            "    labels (numpy.ndarray): A 1D NumPy array of float labels (N,).")

        .def(
            "predict",
            [](ProxiKNN &self,
               py::array_t<float, py::array::c_style | py::array::forcecast> feature_arr) {
                if (feature_arr.ndim() != 1) {
                    throw std::runtime_error("Input feature array must be 1D");
                }
                std::vector<float> cpp_feature(feature_arr.shape(0));
                if (feature_arr.shape(0) > 0) {
                    std::memcpy(cpp_feature.data(), feature_arr.data(),
                                feature_arr.shape(0) * sizeof(float));
                }
                return self.predict(cpp_feature);
            },
            py::arg("feature"),
            "Predicts the class label for a single feature vector.\n\n"
            "Args:\n"
            "    feature (numpy.ndarray): The 1D feature vector.\n\n"
            "Returns:\n"
            "    float: The predicted class label.")

        .def(
            "predict_batch",
            [](ProxiKNN &self,
               py::array_t<float, py::array::c_style | py::array::forcecast> features_arr) {
                if (features_arr.ndim() != 2) {
                    throw std::runtime_error("Input features array must be 2D (M, D)");
                }
                std::vector<std::vector<float>> cpp_features(features_arr.shape(0));
                const float *features_data_ptr = features_arr.data();
                for (size_t i = 0; i < features_arr.shape(0); ++i) {
                    cpp_features[i].resize(features_arr.shape(1));
                    if (features_arr.shape(1) > 0) {
                        std::memcpy(cpp_features[i].data(),
                                    features_data_ptr + i * features_arr.shape(1),
                                    features_arr.shape(1) * sizeof(float));
                    }
                }
                return self.predict(cpp_features);
            },
            py::arg("features"),
            "Predicts class labels for a batch of feature vectors.\n\n"
            "Args:\n"
            "    features (numpy.ndarray): A 2D NumPy array of feature vectors (M, D).\n\n"
            "Returns:\n"
            "    list[float]: Predicted class labels for each feature vector.");
}