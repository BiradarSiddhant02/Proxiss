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
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <cstring>

ProxiPCA::ProxiPCA(size_t n_components, size_t k, size_t num_threads,
                   const std::string &objective_function)
    : m_pca_result(0, 0),  // Will be initialized in fit_transform_index
      m_proxi_flat(k, num_threads, objective_function),
      m_n_components(n_components),
      m_is_fitted(false) {}

void ProxiPCA::fit_transform_index(const std::vector<std::vector<float>> &embeddings) {
    if (embeddings.empty()) {
        throw std::runtime_error("Embeddings cannot be empty.");
    }

    size_t n_samples = embeddings.size();
    size_t n_features = embeddings[0].size();

    // Convert input to Matrix format
    Matrix data(n_samples, n_features);
    for (size_t i = 0; i < n_samples; ++i) {
        if (embeddings[i].size() != n_features) {
            throw std::runtime_error("All embeddings must have the same dimension.");
        }
        for (size_t j = 0; j < n_features; ++j) {
            data(i, j) = embeddings[i][j];
        }
    }

    // Fit PCA on the embeddings
    m_pca_result = PCA::fit(data, m_n_components);
    m_is_fitted = true;

    // Transform embeddings to reduced dimensions
    Matrix reduced = PCA::transform(data, m_pca_result);

    // Convert Matrix back to vector<vector<float>> for ProxiFlat
    std::vector<std::vector<float>> reduced_embeddings(reduced.rows());
    for (size_t i = 0; i < reduced.rows(); ++i) {
        reduced_embeddings[i].resize(reduced.cols());
        for (size_t j = 0; j < reduced.cols(); ++j) {
            reduced_embeddings[i][j] = reduced(i, j);
        }
    }

    // Index the reduced embeddings in ProxiFlat
    m_proxi_flat.index_data(reduced_embeddings);
}

std::vector<size_t> ProxiPCA::find_indices(const std::vector<float> &query) {
    if (!m_is_fitted) {
        throw std::runtime_error("ProxiPCA must be fitted before searching. Call fit_transform_index() first.");
    }

    // Convert query to Matrix (1 x D)
    Matrix query_mat(1, query.size());
    for (size_t i = 0; i < query.size(); ++i) {
        query_mat(0, i) = query[i];
    }

    // Reduce query dimensions using PCA
    Matrix reduced_query = PCA::transform(query_mat, m_pca_result);

    // Convert reduced query back to vector
    std::vector<float> reduced_query_vec(reduced_query.cols());
    for (size_t j = 0; j < reduced_query.cols(); ++j) {
        reduced_query_vec[j] = reduced_query(0, j);
    }

    // Search in ProxiFlat with reduced query
    return m_proxi_flat.find_indices(reduced_query_vec);
}

std::vector<std::vector<size_t>> ProxiPCA::find_indices_batched(
    const std::vector<std::vector<float>> &queries) {
    
    if (!m_is_fitted) {
        throw std::runtime_error("ProxiPCA must be fitted before searching. Call fit_transform_index() first.");
    }

    size_t n_queries = queries.size();
    if (n_queries == 0) {
        return {};
    }

    size_t n_features = queries[0].size();

    // Convert queries to Matrix format
    Matrix queries_mat(n_queries, n_features);
    for (size_t i = 0; i < n_queries; ++i) {
        if (queries[i].size() != n_features) {
            throw std::runtime_error("All queries must have the same dimension.");
        }
        for (size_t j = 0; j < n_features; ++j) {
            queries_mat(i, j) = queries[i][j];
        }
    }

    // Transform all queries using PCA
    Matrix reduced_queries = PCA::transform(queries_mat, m_pca_result);

    // Convert to vector<vector<float>>
    std::vector<std::vector<float>> reduced_queries_vec(reduced_queries.rows());
    for (size_t i = 0; i < reduced_queries.rows(); ++i) {
        reduced_queries_vec[i].resize(reduced_queries.cols());
        for (size_t j = 0; j < reduced_queries.cols(); ++j) {
            reduced_queries_vec[i][j] = reduced_queries(i, j);
        }
    }

    // Search in ProxiFlat with reduced queries (batch method)
    return m_proxi_flat.find_indices(reduced_queries_vec);
}

void ProxiPCA::insert_data(const std::vector<float> &embedding) {
    if (!m_is_fitted) {
        throw std::runtime_error("ProxiPCA must be fitted before inserting data. Call fit_transform_index() first.");
    }

    // Convert to Matrix (1 x D)
    Matrix emb_mat(1, embedding.size());
    for (size_t i = 0; i < embedding.size(); ++i) {
        emb_mat(0, i) = embedding[i];
    }

    // Reduce dimensions using PCA
    Matrix reduced = PCA::transform(emb_mat, m_pca_result);

    // Convert to vector
    std::vector<float> reduced_vec(reduced.cols());
    for (size_t j = 0; j < reduced.cols(); ++j) {
        reduced_vec[j] = reduced(0, j);
    }

    // Insert into ProxiFlat
    m_proxi_flat.insert_data(reduced_vec);
}

void ProxiPCA::save_state(const std::string &directory_path) {
    if (!m_is_fitted) {
        throw std::runtime_error("ProxiPCA must be fitted before saving. Call fit_transform_index() first.");
    }

    try {
        std::filesystem::path path(directory_path);
        if (!std::filesystem::exists(path)) {
            std::filesystem::create_directories(path);
        }
        if (!std::filesystem::is_directory(path)) {
            throw std::runtime_error("Path is not a directory: " + directory_path);
        }

        // Save ProxiFlat index first
        m_proxi_flat.save_state(directory_path);

        // Save PCA state
        std::filesystem::path pca_file_path = path / "pca_data.bin";
        std::ofstream out_file(pca_file_path, std::ios::binary);
        if (!out_file.is_open()) {
            throw std::runtime_error("Failed to open PCA file for writing: " + pca_file_path.string());
        }

        // Save metadata
        out_file.write(reinterpret_cast<const char*>(&m_n_components), sizeof(size_t));
        
        // Save PCA components matrix dimensions and data
        size_t components_rows = m_pca_result.components.rows();
        size_t components_cols = m_pca_result.components.cols();
        out_file.write(reinterpret_cast<const char*>(&components_rows), sizeof(size_t));
        out_file.write(reinterpret_cast<const char*>(&components_cols), sizeof(size_t));
        out_file.write(reinterpret_cast<const char*>(m_pca_result.components.data()), 
                      components_rows * components_cols * sizeof(float));

        // Save mean matrix dimensions and data
        size_t mean_rows = m_pca_result.mean.rows();
        size_t mean_cols = m_pca_result.mean.cols();
        out_file.write(reinterpret_cast<const char*>(&mean_rows), sizeof(size_t));
        out_file.write(reinterpret_cast<const char*>(&mean_cols), sizeof(size_t));
        out_file.write(reinterpret_cast<const char*>(m_pca_result.mean.data()), 
                      mean_rows * mean_cols * sizeof(float));

        // Save explained variance vectors
        size_t var_size = m_pca_result.explained_variance.size();
        out_file.write(reinterpret_cast<const char*>(&var_size), sizeof(size_t));
        out_file.write(reinterpret_cast<const char*>(m_pca_result.explained_variance.data()), 
                      var_size * sizeof(float));

        size_t var_ratio_size = m_pca_result.explained_variance_ratio.size();
        out_file.write(reinterpret_cast<const char*>(&var_ratio_size), sizeof(size_t));
        out_file.write(reinterpret_cast<const char*>(m_pca_result.explained_variance_ratio.data()), 
                      var_ratio_size * sizeof(float));

        if (!out_file.good()) {
            throw std::runtime_error("Failed to write PCA data to file: " + pca_file_path.string());
        }

        out_file.close();
    } catch (const std::filesystem::filesystem_error &error) {
        throw std::runtime_error("Filesystem error: " + std::string(error.what()));
    } catch (const std::exception &e) {
        throw std::runtime_error("Save failed: " + std::string(e.what()));
    }
}

void ProxiPCA::load_state(const std::string &directory_path) {
    try {
        std::filesystem::path path(directory_path);
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("Directory does not exist: " + directory_path);
        }
        if (!std::filesystem::is_directory(path)) {
            throw std::runtime_error("Path is not a directory: " + directory_path);
        }

        // Load ProxiFlat index first
        std::filesystem::path proxi_flat_file = path / "data.bin";
        if (!std::filesystem::exists(proxi_flat_file)) {
            throw std::runtime_error("ProxiFlat data file not found: " + proxi_flat_file.string());
        }
        m_proxi_flat.load_state(proxi_flat_file.string());

        // Load PCA state
        std::filesystem::path pca_file_path = path / "pca_data.bin";
        if (!std::filesystem::exists(pca_file_path)) {
            throw std::runtime_error("PCA data file not found: " + pca_file_path.string());
        }

        std::ifstream in_file(pca_file_path, std::ios::binary);
        if (!in_file.is_open()) {
            throw std::runtime_error("Failed to open PCA file for reading: " + pca_file_path.string());
        }

        // Load metadata
        in_file.read(reinterpret_cast<char*>(&m_n_components), sizeof(size_t));

        // Load PCA components matrix
        size_t components_rows, components_cols;
        in_file.read(reinterpret_cast<char*>(&components_rows), sizeof(size_t));
        in_file.read(reinterpret_cast<char*>(&components_cols), sizeof(size_t));
        
        m_pca_result.components = Matrix(components_rows, components_cols);
        in_file.read(reinterpret_cast<char*>(m_pca_result.components.data()), 
                    components_rows * components_cols * sizeof(float));

        // Load mean matrix
        size_t mean_rows, mean_cols;
        in_file.read(reinterpret_cast<char*>(&mean_rows), sizeof(size_t));
        in_file.read(reinterpret_cast<char*>(&mean_cols), sizeof(size_t));
        
        m_pca_result.mean = Matrix(mean_rows, mean_cols);
        in_file.read(reinterpret_cast<char*>(m_pca_result.mean.data()), 
                    mean_rows * mean_cols * sizeof(float));

        // Load explained variance vectors
        size_t var_size;
        in_file.read(reinterpret_cast<char*>(&var_size), sizeof(size_t));
        m_pca_result.explained_variance.resize(var_size);
        in_file.read(reinterpret_cast<char*>(m_pca_result.explained_variance.data()), 
                    var_size * sizeof(float));

        size_t var_ratio_size;
        in_file.read(reinterpret_cast<char*>(&var_ratio_size), sizeof(size_t));
        m_pca_result.explained_variance_ratio.resize(var_ratio_size);
        in_file.read(reinterpret_cast<char*>(m_pca_result.explained_variance_ratio.data()), 
                    var_ratio_size * sizeof(float));

        if (!in_file.good()) {
            throw std::runtime_error("Failed to read PCA data from file: " + pca_file_path.string());
        }

        in_file.close();
        m_is_fitted = true;
    } catch (const std::filesystem::filesystem_error &error) {
        throw std::runtime_error("Filesystem error: " + std::string(error.what()));
    } catch (const std::exception &e) {
        throw std::runtime_error("Load failed: " + std::string(e.what()));
    }
}
