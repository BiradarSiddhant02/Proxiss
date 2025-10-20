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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pca.hpp"
#include "proxi_flat.h"

/**
 * @class ProxiPCA
 * @brief A wrapper that combines PCA dimensionality reduction with ProxiFlat nearest neighbor search.
 *
 * ProxiPCA reduces the dimensionality of embeddings using PCA, then indexes the reduced
 * embeddings using ProxiFlat for efficient nearest neighbor search. For queries, it first
 * reduces the query dimensions using the fitted PCA model, then searches in the ProxiFlat index.
 */
class ProxiPCA {
public:
    /**
     * @brief Constructor for ProxiPCA.
     * @param n_components Number of PCA components (reduced dimensions).
     * @param k Number of nearest neighbors to find.
     * @param num_threads Number of threads for parallel operations.
     * @param objective_function Distance metric ("l2", "l1", or "cos").
     */
    ProxiPCA(size_t n_components, size_t k, size_t num_threads,
             const std::string &objective_function = "l2");

    /**
     * @brief Fit PCA on embeddings and index the reduced embeddings.
     * @param embeddings Input embeddings (N x D matrix).
     */
    void fit_transform_index(const std::vector<std::vector<float>> &embeddings);

    /**
     * @brief Find k nearest neighbors for a single query.
     * @param query Query vector (D dimensions, will be reduced to n_components).
     * @return Vector of indices of nearest neighbors.
     */
    std::vector<size_t> find_indices(const std::vector<float> &query);

    /**
     * @brief Find k nearest neighbors for multiple queries.
     * @param queries Batch of query vectors (M x D matrix).
     * @return Vector of vectors containing indices of nearest neighbors for each query.
     */
    std::vector<std::vector<size_t>> find_indices_batched(
        const std::vector<std::vector<float>> &queries);

    /**
     * @brief Insert a new embedding into the index.
     * @param embedding New embedding vector (D dimensions).
     */
    void insert_data(const std::vector<float> &embedding);

    /**
     * @brief Save the ProxiPCA state to disk.
     * @param directory_path Directory where state will be saved.
     */
    void save_state(const std::string &directory_path);

    /**
     * @brief Load the ProxiPCA state from disk.
     * @param directory_path Directory where state is stored.
     */
    void load_state(const std::string &directory_path);

    // Getters
    size_t get_k() const { return m_proxi_flat.get_k(); }
    size_t get_num_threads() const { return m_proxi_flat.get_num_threads(); }
    size_t get_n_components() const { return m_n_components; }
    bool is_fitted() const { return m_is_fitted; }
    
    // Get PCA information
    const PCAResult& get_pca_result() const { return m_pca_result; }

    // Setters
    void set_k(size_t k) { m_proxi_flat.set_k(k); }
    void set_num_threads(size_t num_threads) { m_proxi_flat.set_num_threads(num_threads); }

private:
    PCAResult m_pca_result;     ///< PCA result containing components, mean, variance
    ProxiFlat m_proxi_flat;     ///< ProxiFlat index for reduced embeddings
    size_t m_n_components;      ///< Number of PCA components
    bool m_is_fitted;           ///< Whether PCA has been fitted
};
