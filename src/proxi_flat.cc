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

// --- proxi_flat.cc --- //

#include "proxi_flat.h"
#include "distance.hpp"

// PROTECTED

std::vector<std::uint8_t> ProxiFlat::serialise() {
    /**
     * @brief Serialises the ProxiFlat object into a contiguous array of bytes.
     * This method is called by the public `save` method.
     *
     * Serialization Format (in order):
     * 1.  `m_num_samples` (size_t, 8 bytes)
     * 2.  `m_num_features` (size_t, 8 bytes)
     * 3.  `m_K` (size_t, 8 bytes)
     * 4.  `m_is_indexed` (converted to size_t, 8 bytes)
     * 5.  `m_num_threads` (size_t, 8 bytes)
     * 6.  `m_objective_function_id` (char[8], 8 bytes, null-padded if shorter)
     * 7.  `m_embeddings_flat` (float array, `m_num_samples * m_num_features * sizeof(float)` bytes)
     * 8.  Document lengths (array of size_t, `m_num_samples * sizeof(size_t)` bytes)
     * 9.  Document contents (concatenated char arrays, total length sum of individual doc lengths)
     *
     * @return std::vector<std::uint8_t> A byte vector representing the serialised state.
     * @throws std::runtime_error if the data is not indexed (`m_is_indexed` is false).
     */

    if (!m_is_indexed)
        throw std::runtime_error("Data is not indexed. Index data before saving");

    std::vector<std::uint8_t> object; // Final output

    // ---- Serialize metadata (6 words = 48 bytes) ----
    auto insert_bytes = [&](const std::vector<std::uint8_t> &bytes) {
        object.insert(object.end(), bytes.begin(), bytes.end());
    };

    insert_bytes(ProxiFlat::to_bytes<size_t>(m_num_samples));                     // Word 1
    insert_bytes(ProxiFlat::to_bytes<size_t>(m_num_features));                    // Word 2
    insert_bytes(ProxiFlat::to_bytes<size_t>(m_K));                               // Word 3
    insert_bytes(ProxiFlat::to_bytes<size_t>(static_cast<size_t>(m_is_indexed))); // Word 4
    insert_bytes(ProxiFlat::to_bytes<size_t>(m_num_threads));                     // Word 5

    // Word 6: m_objective_function_id, padded to 8 bytes
    std::vector<std::uint8_t> fnc_id_serial(8, 0);
    for (size_t i = 0; i < std::min<size_t>(8, m_objective_function_id.size()); ++i)
        fnc_id_serial[i] = static_cast<std::uint8_t>(m_objective_function_id[i]);
    insert_bytes(fnc_id_serial);

    // ---- Serialize embeddings ----
    for (float val : m_embeddings_flat) {
        insert_bytes(ProxiFlat::to_bytes<float>(val));
    }

    // ---- Serialize text documents ----
    // Insert string lengths
    for (const std::string &doc : m_documents)
        insert_bytes(ProxiFlat::to_bytes<size_t>(doc.length()));

    // Insert strings
    for (const std::string &doc : m_documents) {
        insert_bytes(std::vector<std::uint8_t>{doc.begin(), doc.end()});
    }

    return object;
}

template <typename T> std::vector<std::uint8_t> ProxiFlat::to_bytes(const T value) noexcept {
    /**
     * @brief Serializes a trivially copyable object into a vector of bytes.
     * This is a utility function for converting fundamental types (like size_t, float)
     * into their byte representations for serialization.
     *
     * @tparam T The type of the value to serialize. Must be trivially copyable.
     * @param value The value to serialize.
     * @return std::vector<std::uint8_t> A vector containing the byte-wise representation of `value`.
     * @note Assumes native endianness. For cross-platform compatibility, endian conversion might be needed.
     */

    std::vector<std::uint8_t> bytes(sizeof(T));
    std::memcpy(bytes.data(), &value, sizeof(T));
    return bytes;
}

std::vector<std::uint8_t>
ProxiFlat::strings_to_bytes(const std::vector<std::string> &strs) noexcept {
    /**
     * @brief Serializes a vector of strings into a single byte vector, with a delimiter.
     * @deprecated This function is not currently used in the main serialization logic.
     *             The current serialization stores string lengths and then raw string data.
     *
     * @param strs The vector of strings to serialize.
     * @return std::vector<std::uint8_t> A byte vector containing the concatenated strings,
     *                                   each followed by a '\xFF' delimiter.
     */

    const unsigned char delim = '\xFF';

    std::vector<std::uint8_t> bytes;
    for (const auto &str : strs) {
        bytes.insert(bytes.begin(), str.begin(), str.end());
        bytes.push_back(delim);
    }

    return bytes;
}

// PRIVATE
std::vector<size_t> ProxiFlat::m_get_neighbours(const std::vector<float> &query) noexcept {
    /**
     * @brief Finds the indices of the K nearest neighbours for a given query vector.
     * This is an internal helper method used by `find_indices` and `find_docs`.
     * It uses the objective function defined during ProxiFlat initialization (e.g., L2, Cosine).
     * The search is performed by iterating through all indexed samples and maintaining a
     * min-priority queue (simulated with a max-priority queue storing negative distances or
     * by inverting comparison for similarity) of the K closest items found so far.
     *
     * @param query The query vector (embedding) for which to find neighbours.
     * @return A vector containing the indices of the K nearest neighbours in the `m_embeddings_flat` data.
     *         The order of indices is typically from furthest to closest of the K neighbours due to priority queue pop order.
     * @note The `noexcept` specifier should be reviewed as `m_objective_function` could potentially throw
     *       if it's a complex function object, though standard distance functions here are unlikely to.
     *       If `m_num_samples` is 0 or less than `m_K`, the behavior might be to return fewer than `m_K` indices.
     */
    using pair = std::pair<float, size_t>;

    std::priority_queue<pair> heap;

    long long int num_samples = static_cast<long long int>(m_num_samples);
    long long int num_features = static_cast<long long int>(m_num_features);

#pragma omp for nowait
    for (long long int i = 0; i < num_samples; i++) {
        float distance = m_objective_function(
            std::span<const float>(query),
            std::span<const float>(m_embeddings_flat).subspan(i * num_features, num_features));

        bool should_insert = heap.size() < m_K;
        bool should_replace = !should_insert && distance < heap.top().first;

        if (should_insert || should_replace) {
            if (should_replace)
                heap.pop();
            heap.emplace(distance, i);
        }
    }

    std::vector<size_t> indices;
    while (!heap.empty()) {
        indices.push_back(heap.top().second);
        heap.pop();
    }

    return indices;
}

// PUBLIC
ProxiFlat::ProxiFlat(const size_t k, const size_t num_threads, const std::string objective_function)
    : m_num_samples(0), m_num_features(0), m_K(k), m_is_indexed(false), m_num_threads(num_threads),
      m_objective_function_id(objective_function) {
    /**
     * @brief Constructs a ProxiFlat object with specified parameters.
     *
     * Initializes the object with the number of nearest neighbours (K) to find,
     * the number of threads for parallel operations, and the objective function
     * (distance metric) to use. The object is initially not indexed.
     *
     * @param k The number of nearest neighbours to retrieve in queries.
     * @param num_threads The number of threads to use for parallelizable operations
     *                    (e.g., batched queries). OpenMP is used for parallelism.
     * @param objective_function A string specifying the distance metric.
     *                           Supported values: "l2" (Euclidean), "l1" (Manhattan), "cos" (Cosine similarity/distance).
     *                           Defaults to "l2".
     * @throws std::runtime_error If an unsupported `objective_function` string is provided.
     */

    if (objective_function == "l2") {
        m_objective_function = euclidean_distance;
    } else if (objective_function == "l1") {
        m_objective_function = manhattan_distance;
    } else if (objective_function == "cos") {
        m_objective_function = cosine_similarity;
    } else {
        throw std::runtime_error("Invalid Distance function.");
    }

}

void ProxiFlat::index_data(const std::vector<std::vector<float>> &embeddings,
                           const std::vector<std::string> &documents) {
    /**
     * @brief Indexes the provided embeddings and their corresponding documents.
     *
     * This method populates the ProxiFlat instance with the given data.
     * Embeddings are flattened and stored internally. The dimensionality (number of features)
     * is inferred from the first embedding. All subsequent embeddings must have the same dimension.
     * After successful indexing, the `m_is_indexed` flag is set to true.
     *
     * @param embeddings A vector of vectors, where each inner vector is a float embedding.
     *                   The outer vector represents the dataset, and each inner vector is a data point.
     * @param documents A vector of strings, where each string is a document or label
     *                  corresponding to the embedding at the same index in the `embeddings` vector.
     * @throws std::runtime_error If `embeddings` or `documents` are empty.
     * @throws std::runtime_error If the number of embeddings does not match the number of documents.
     * @throws std::runtime_error If any embedding has a dimension inconsistent with the first embedding.
     */
    if (embeddings.empty() || documents.empty())
        throw std::runtime_error("Embeddings or Documents cannot be empty.");

    if (embeddings.size() != documents.size())
        throw std::runtime_error("Size of embeddings and corpus are unequal");

    m_documents = documents;

    m_num_samples = documents.size();
    m_num_features = embeddings[0].size();

    m_embeddings_flat.resize(m_num_features * m_num_samples);
    for (size_t i = 0; i < m_num_samples; i++) {
        if (embeddings[i].size() != m_num_features)
            throw std::runtime_error("Number of features is inconsistent.");
    }
    for (size_t i = 0; i < m_num_samples; i++) {
        std::memcpy(&m_embeddings_flat[i * m_num_features], embeddings[i].data(),
                    m_num_features * sizeof(float));
    }

    m_is_indexed = true;
}

std::vector<size_t> ProxiFlat::find_indices(const std::vector<float> &query) {
    /**
     * @brief Finds the indices of the K nearest neighbours for a single query vector.
     *
     * @param query The query vector (embedding) for which to find neighbours.
     * @return A vector containing the indices of the K nearest neighbours. The order of indices
     *         is typically from furthest to closest of the K neighbours.
     * @throws std::runtime_error If data has not been indexed yet (call `index_data` first).
     * @throws std::runtime_error If the dimension of the `query` vector does not match
     *                               the dimension of the indexed embeddings (`m_num_features`).
     */

    if (!m_is_indexed)
        throw std::runtime_error("Call index() before querying.");
    if (query.size() != m_num_features)
        throw std::runtime_error("Query vector dimensions mismatch dataset feature dimensions.");

    return m_get_neighbours(query);
}

std::vector<std::vector<size_t>>
ProxiFlat::find_indices(const std::vector<std::vector<float>> &queries) noexcept {
    /**
     * @brief Finds the indices of the K nearest neighbours for multiple query vectors (batched query).
     * This operation is parallelized using OpenMP based on `m_num_threads`.
     *
     * @param queries A vector of query vectors. Each inner vector is a query embedding.
     * @return A vector of vectors. Each inner vector contains the indices of the K nearest
     *         neighbours for the corresponding query in the input `queries` vector.
     *         Order of indices within inner vectors is typically from furthest to closest.
     * @throws std::runtime_error This method is marked `noexcept`, but it calls the single-query
     *                            `find_indices` which can throw `std::runtime_error` or `std::runtime_error`.
     *                            This can lead to `std::terminate` if an exception propagates out.
     *                            The `noexcept` should be removed or error handling improved.
     *                            Specifically, if any query has inconsistent dimensions, an exception will be thrown.
     */
    omp_set_num_threads(m_num_threads);

    std::vector<std::vector<size_t>> indices(queries.size());

    long long int num_queries = static_cast<long long int>(queries.size());

#pragma omp parallel for
    for (long long int i = 0; i < num_queries; i++) {
        indices[i] = find_indices(queries[i]);
    }

    return indices;
}

std::vector<std::string> ProxiFlat::find_docs(const std::vector<float> &query) noexcept {
    /**
     * @brief Finds the documents (text) corresponding to the K nearest neighbours for a single query vector.
     *
     * @param query The query vector (embedding).
     * @return A vector of strings, where each string is a document corresponding to one of the
     *         K nearest neighbours. The order corresponds to the order of indices from `m_get_neighbours`.
     * @throws std::runtime_error This method is marked `noexcept`, but it calls `m_get_neighbours`
     *                            and potentially relies on `find_indices` behavior which can throw.
     *                            If `query` has incorrect dimensions or data is not indexed, behavior is undefined
     *                            or may lead to crashes if underlying calls throw. `noexcept` should be reconsidered.
     *                            If K is larger than the number of indexed items, it may return fewer than K documents
     *                            or throw if indices are out of bounds.
     */
    std::vector<size_t> neighbours = m_get_neighbours(query);

    std::vector<std::string> docs(m_K);
    for (size_t i = 0; i < m_K; i++)
        docs[i] = m_documents[neighbours[i]];

    return docs;
}

std::vector<std::vector<std::string>>
ProxiFlat::find_docs(const std::vector<std::vector<float>> &queries) noexcept {
    /**
     * @brief Finds the documents (text) for the K nearest neighbours for multiple query vectors (batched query).
     * This operation is parallelized using OpenMP based on `m_num_threads`.
     *
     * @param queries A vector of query vectors. Each inner vector is a query embedding.
     * @return A vector of vectors. Each inner vector contains the K nearest documents (strings)
     *         for the corresponding query in the input `queries` vector.
     * @throws std::runtime_error This method is marked `noexcept`, but it calls the single-query
     *                            `find_docs` which itself has a problematic `noexcept` and can throw.
     *                            This can lead to `std::terminate`. `noexcept` should be removed.
     */
    omp_set_num_threads(m_num_threads);

    std::vector<std::vector<std::string>> results(queries.size());

    long long int num_queries = static_cast<long long int>(queries.size());

#pragma omp parallel for
    for (long long int i = 0; i < num_queries; i++) {
        results[i] = find_docs(queries[i]);
    }

    return results;
}

void ProxiFlat::insert_data(const std::vector<float> &embedding, const std::string &text) {
    /**
     * @brief Inserts a new data point (embedding and its corresponding text document) into the index.
     *
     * The new embedding is appended to the internal flat embedding store, and the document
     * is added to the document list. The total number of samples (`m_num_samples`) is incremented.
     * This method assumes the ProxiFlat instance has been initialized (e.g., `m_num_features` is set,
     * typically by a prior call to `index_data` or by loading an existing index).
     * If `m_num_features` is 0 (e.g. on a newly constructed, un-indexed instance), this could lead to issues.
     *
     * @param embedding The embedding vector (std::vector<float>) of the text to be inserted.
     * @param text The text document (std::string) to be inserted.
     * @throws std::runtime_error If the dimension of the provided `embedding` does not match
     *                               the `m_num_features` of the existing dataset. This check only occurs
     *                               if `m_num_features` > 0.
     * @note If called on a completely empty index where `m_num_features` hasn't been set, this method
     *       might not behave as expected or might not set `m_num_features` correctly for the first insertion.
     *       It's generally expected that `index_data` is called at least once before `insert_data`,
     *       or the index is loaded from a file, to establish `m_num_features`.
     */

    // Check if the vector is of the same length of the dataset
    if (embedding.size() != m_num_features)
        throw std::runtime_error("Invalid embedding vector size.");

    m_embeddings_flat.insert(m_embeddings_flat.end(), embedding.begin(), embedding.end());
    m_documents.push_back(text);

    m_num_samples++;
}

void ProxiFlat::save_state(const std::string &path_str) {
    /**
     * @brief Saves the serialized ProxiFlat object to a binary file named "data.bin"
     * within the specified directory.
     *
     * The method first checks if the provided path is a valid, existing directory.
     * It then calls the internal `serialise()` method to get the byte representation
     * of the object and writes these bytes to "data.bin" in the given directory.
     *
     * @param path_str The directory path where the "data.bin" file will be saved.
     * @throws std::runtime_error If `m_is_indexed` is false (data must be indexed before saving).
     * @throws std::runtime_error If `path_str` is not an existing directory.
     * @throws std::runtime_error If file opening or writing fails (e.g., permissions, disk full).
     * @throws std::filesystem::filesystem_error For other filesystem-related issues.
     */

    try {
        std::filesystem::path path(path_str);
        if (!std::filesystem::exists(path))
            throw std::runtime_error("Path does not exist: " + path_str);
        if (!std::filesystem::is_directory(path))
            throw std::runtime_error("Path is not a directory: " + path_str);

        // Serialize the object
        std::vector<std::uint8_t> bytes = serialise();

        // Create the full file path (e.g., path/data.bin)
        std::filesystem::path file_path = path / "data.bin";

        // Open the file in binary mode
        std::ofstream out_file(file_path, std::ios::binary);
        if (!out_file.is_open())
            throw std::runtime_error("Failed to open file for writing: " + file_path.string());

        // Write the bytes
        out_file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        if (!out_file.good())
            throw std::runtime_error("Failed to write data to file: " + file_path.string());

        out_file.close();
    } catch (const std::filesystem::filesystem_error &error) {
        throw std::runtime_error("Filesystem error: " + std::string(error.what()));
    } catch (const std::exception &e) {
        throw std::runtime_error("Save failed: " + std::string(e.what()));
    }
}

void ProxiFlat::load_state(const std::string &path_str) {
    /**
     * @brief Loads a ProxiFlat object from a serialized binary file.
     *
     * This method reads the byte stream from the specified file path (expected to be "data.bin"),
     * deserializes it, and populates the current ProxiFlat instance's members
     * (`m_num_samples`, `m_num_features`, `m_K`, `m_is_indexed`, `m_num_threads`,
     * `m_objective_function_id`, `m_embeddings_flat`, `m_documents`).
     * It also re-initializes the `m_objective_function` based on the loaded ID.
     *
     * The expected file format is detailed in the `serialise()` method's docstring.
     *
     * @param path_str The full file path to the serialized ProxiFlat data (e.g., "/path/to/your/data.bin").
     * @throws std::runtime_error If the file does not exist, is not a regular file,
     *                            or if any part of the reading or deserialization process fails
     *                            (e.g., file truncated, corrupted data, I/O errors, unknown objective function ID).
     * @throws std::filesystem::filesystem_error For other filesystem-related issues.
     */
    try {
        std::filesystem::path file_path(path_str);

        if (!std::filesystem::exists(file_path)) {
            throw std::runtime_error("File does not exist: " + path_str);
        }
        if (!std::filesystem::is_regular_file(file_path)) {
            throw std::runtime_error("Path is not a file: " + path_str);
        }

        std::ifstream in_file(file_path, std::ios::binary | std::ios::ate);
        if (!in_file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path.string());
        }

        std::streamsize size = in_file.tellg();
        in_file.seekg(0, std::ios::beg);

        std::vector<std::uint8_t> buffer(size);
        if (!in_file.read(reinterpret_cast<char *>(buffer.data()), size)) {
            throw std::runtime_error("Failed to read data from file: " + file_path.string());
        }
        in_file.close();

        size_t offset = 0;

        auto read_size_t = [&](size_t &out) {
            out = *reinterpret_cast<const size_t *>(buffer.data() + offset);
            offset += sizeof(size_t);
        };

        // ---- Metadata ----
        read_size_t(m_num_samples);
        read_size_t(m_num_features);
        read_size_t(m_K);
        size_t indexed_flag;
        read_size_t(indexed_flag);
        m_is_indexed = static_cast<bool>(indexed_flag);
        read_size_t(m_num_threads);

        char fnc_id[9] = {0};
        std::memcpy(fnc_id, buffer.data() + offset, 8);
        offset += 8;
        m_objective_function_id = std::string(fnc_id);

        if (m_objective_function_id.find("l2") != std::string::npos) {
            m_objective_function = euclidean_distance;
        } else if (m_objective_function_id.find("l1") != std::string::npos) {
            m_objective_function = manhattan_distance;
        } else if (m_objective_function_id.find("cos") != std::string::npos) {
            m_objective_function = cosine_similarity;
        } else {
            throw std::runtime_error("Unknown objective function: " + m_objective_function_id);
        }

        // ---- Embeddings ----
        size_t num_floats = m_num_samples * m_num_features;
        size_t embeddings_size = num_floats * sizeof(float);

        if (buffer.size() < offset + embeddings_size) {
            throw std::runtime_error("File truncated: embeddings data incomplete");
        }

        m_embeddings_flat.resize(num_floats);
        std::memcpy(m_embeddings_flat.data(), buffer.data() + offset, embeddings_size);
        offset += embeddings_size;

        // ---- Documents ----
        m_documents.clear();
        std::vector<size_t> doc_lengths(m_num_samples);

        // Read lengths
        for (size_t i = 0; i < m_num_samples; ++i) {
            if (offset + sizeof(size_t) > buffer.size()) {
                throw std::runtime_error("File truncated: missing document length data");
            }
            doc_lengths[i] = *reinterpret_cast<const size_t *>(buffer.data() + offset);
            offset += sizeof(size_t);
        }

        // Read strings
        for (size_t len : doc_lengths) {
            if (offset + len > buffer.size()) {
                throw std::runtime_error("File truncated: missing document content");
            }
            m_documents.emplace_back(reinterpret_cast<const char *>(buffer.data() + offset), len);
            offset += len;
        }

    } catch (const std::filesystem::filesystem_error &error) {
        throw std::runtime_error("Filesystem error: " + std::string(error.what()));
    } catch (const std::exception &e) {
        throw std::runtime_error("Load failed: " + std::string(e.what()));
    }
}
