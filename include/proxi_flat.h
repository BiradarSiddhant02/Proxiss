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

/**
 * @class ProxiFlat
 * @brief A class for performing fast approximate nearest neighbour searches.
 *
 * ProxiFlat stores embeddings and their corresponding documents (text).
 * It allows indexing this data and then querying for the K nearest neighbours
 * of a given query embedding. Embeddings are stored internally as a flat
 * vector for efficiency, but are handled as std::vector<std::vector<float>>
 * in the public interface for `index_data`.
 *
 * Supports L1, L2 (Euclidean), and Cosine distance/similarity metrics.
 * Data can be saved to and loaded from disk.
 */
class ProxiFlat {
protected:
    /**
     * @brief Serializes the ProxiFlat object into a byte vector.
     * @return std::vector<std::uint8_t> Byte vector of the serialized object.
     * @throws std::runtime_error if data has not been indexed.
     */
    std::vector<std::uint8_t> serialise();

    /**
     * @brief Converts a trivially copyable type to a byte vector.
     * @tparam T Type of the value to convert.
     * @param value The value to convert.
     * @return std::vector<std::uint8_t> Byte vector representation.
     */
    template <typename T> static std::vector<std::uint8_t> to_bytes(const T value) noexcept;

    /**
     * @brief Converts a vector of strings to a byte vector, with a delimiter.
     * @param strs Vector of strings to convert.
     * @return std::vector<std::uint8_t> Byte vector representation.
     * @deprecated This method is not currently used in the serialization process.
     */
    std::vector<std::uint8_t> strings_to_bytes(const std::vector<std::string> &strs) noexcept;

private:
    // Embeddings flattened for internal storage
    std::vector<float> m_embeddings_flat;

    // Document Chunks corresponding to embeddings
    std::vector<std::string> m_documents;

    // Number of data samples (embeddings/documents)
    size_t m_num_samples;
    // Dimensionality of the embeddings
    size_t m_num_features;

    // Number of nearest neighbours to retrieve (K)
    size_t m_K;

    // Flag indicating if data has been indexed
    bool m_is_indexed;

    // Number of threads for parallel operations (e.g., batched queries)
    size_t m_num_threads;

    // Function pointer for the chosen distance/similarity metric
    std::function<float(std::span<const float>, std::span<const float>)> m_objective_function;

    // String identifier for the objective function (e.g., "l2", "cos")
    std::string m_objective_function_id;

    /**
     * @brief Internal helper to find K nearest neighbour indices for a single query.
     * @param query The query embedding.
     * @return std::vector<size_t> Indices of the K nearest neighbours.
     * @note This method is marked noexcept but relies on m_objective_function which might throw.
     *       Consider removing noexcept or ensuring m_objective_function cannot throw.
     */
    std::vector<size_t> m_get_neighbours(const std::vector<float> &query) noexcept;

public:
    /**
     * @brief Constructs a ProxiFlat object.
     * @param k Number of nearest neighbours to find.
     * @param num_threads Number of threads for parallel operations.
     * @param objective_function Name of the distance metric ("l1", "l2", "cos"). Defaults to "l2".
     * @throws std::invalid_argument if an unsupported objective_function is provided.
     */
    ProxiFlat(const size_t k, const size_t num_threads,
              const std::string objective_function = "l2");

    /**
     * @brief Indexes the provided embeddings and their corresponding documents.
     * The number of features (embedding dimension) is inferred from the first embedding.
     * @param embeddings A vector of embeddings (each embedding is a vector of floats).
     * @param documents A vector of document strings, corresponding to each embedding.
     * @throws std::invalid_argument if embeddings or documents are empty, or if their sizes
     * mismatch.
     * @throws std::runtime_error if embeddings have inconsistent dimensions.
     */
    void index_data(const std::vector<std::vector<float>> &embeddings,
                    const std::vector<std::string> &documents);

    /**
     * @brief Finds the indices of the K nearest neighbours for a single query embedding.
     * @param query The query embedding.
     * @return std::vector<size_t> Indices of the K nearest neighbours.
     * @throws std::runtime_error if data has not been indexed or if query dimension mismatches.
     */
    std::vector<size_t> find_indices(const std::vector<float> &query);

    /**
     * @brief Finds the indices of the K nearest neighbours for a batch of query embeddings.
     * @param queries A vector of query embeddings.
     * @return std::vector<std::vector<size_t>> For each query, a vector of K nearest neighbour
     * indices.
     * @throws std::runtime_error if data has not been indexed or if any query dimension mismatches.
     * @note The noexcept specifier might be violated if the single find_indices call throws.
     */
    std::vector<std::vector<size_t>>
    find_indices(const std::vector<std::vector<float>> &queries) noexcept;

    /**
     * @brief Finds the K nearest documents for a single query embedding.
     * @param query The query embedding.
     * @return std::vector<std::string> The K nearest documents.
     * @throws std::runtime_error if data has not been indexed or if query dimension mismatches.
     * @note The noexcept specifier might be violated if m_get_neighbours or find_indices throws.
     */
    std::vector<std::string> find_docs(const std::vector<float> &query) noexcept;

    /**
     * @brief Finds the K nearest documents for a batch of query embeddings.
     * @param queries A vector of query embeddings.
     * @return std::vector<std::vector<std::string>> For each query, a vector of K nearest
     * documents.
     * @throws std::runtime_error if data has not been indexed or if any query dimension mismatches.
     * @note The noexcept specifier might be violated if the single find_docs call throws.
     */
    std::vector<std::vector<std::string>>
    find_docs(const std::vector<std::vector<float>> &queries) noexcept;

    /**
     * @brief Inserts a new embedding and its corresponding document into the index.
     * @param embedding The embedding to insert.
     * @param text The document text to insert.
     * @throws std::invalid_argument if the embedding dimension mismatches the existing data.
     * @throws std::runtime_error if data was not indexed prior to insertion (if m_num_features is
     * 0).
     */
    void insert_data(const std::vector<float> &embedding, const std::string &text);

    /**
     * @brief Saves the current ProxiFlat object (index and data) to a directory.
     * A file named "data.bin" will be created in the specified directory.
     * @param path The directory path where the data file will be saved.
     * @throws std::runtime_error if data has not been indexed or if saving fails (e.g., path
     * issues, write errors).
     */
    void save_state(const std::string &path);

    /**
     * @brief Loads a ProxiFlat object from a saved data file.
     * Replaces the current object's state with the loaded data.
     * @param path Path to the serialized ProxiFlat data file ("data.bin").
     * @throws std::runtime_error if loading fails (e.g., file not found, corrupted data, version
     * mismatch).
     */
    void load_state(const std::string &path);
};