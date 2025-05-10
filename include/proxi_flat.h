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

// --- proxi_flat.h --- //

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <queue>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

class ProxiFlat {
protected:
  // Helper function to serialize an object of this class
  std::vector<std::uint8_t> serialise();

  template <typename T>
  static std::vector<std::uint8_t> to_bytes(const T value) noexcept;

  std::vector<std::uint8_t>
  strings_to_bytes(const std::vector<std::string> &strs) noexcept;

private:
  // Embeddings flattened
  std::vector<float> m_embeddings_flat;

  // Document Chunks
  std::vector<std::string> m_documents;

  // Dimensions
  size_t m_num_samples;
  size_t m_num_features;

  // K
  size_t m_K;

  // Flag to represent if the data indexed
  bool m_is_indexed;

  // Number of threads to be used
  size_t m_num_threads;

  std::function<float(std::span<const float>, std::span<const float>)>
      m_objective_function;

  std::string m_objective_function_id;

  std::vector<size_t>
  m_get_neighbours(const std::vector<float> &query) noexcept;

public:
  ProxiFlat(const size_t k, const size_t num_threads,
            const std::string objective_function = "l2");

  ProxiFlat(const std::string &path);

  // Method to index the data
  void index_data(const std::vector<std::vector<float>> &embeddings,
                  const std::vector<std::string> &documents);

  // Methods to return indices of neighbours
  std::vector<size_t> find_indices(const std::vector<float> &query);
  std::vector<std::vector<size_t>>
  find_indices(const std::vector<std::vector<float>> &queries);

  // Method to return document chunk neighbours
  std::vector<std::string> find_docs(const std::vector<float> &query);
  std::vector<std::vector<std::string>>
  find_docs(const std::vector<std::vector<float>> &queries);

  // Method to add new data
  void insert_data(const std::vector<float> &embedding,
                   const std::string &text);

  // Methods to save and load an object of this class
  void save(const std::string &path);

  void load(const std::string &path);
};