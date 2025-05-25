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

    std::vector<std::uint8_t> bytes(sizeof(T));
    std::memcpy(bytes.data(), &value, sizeof(T));
    return bytes;
}

std::vector<std::uint8_t>
ProxiFlat::strings_to_bytes(const std::vector<std::string> &strs) noexcept {

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

    if (!m_is_indexed)
        throw std::runtime_error("Call index() before querying.");
    if (query.size() != m_num_features)
        throw std::runtime_error("Query vector dimensions mismatch dataset feature dimensions.");

    return m_get_neighbours(query);
}

std::vector<std::vector<size_t>>
ProxiFlat::find_indices(const std::vector<std::vector<float>> &queries) noexcept {

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

    std::vector<size_t> neighbours = m_get_neighbours(query);

    std::vector<std::string> docs(m_K);
    for (size_t i = 0; i < m_K; i++)
        docs[i] = m_documents[neighbours[i]];

    return docs;
}

std::vector<std::vector<std::string>>
ProxiFlat::find_docs(const std::vector<std::vector<float>> &queries) noexcept {

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

    if (embedding.size() != m_num_features)
        throw std::runtime_error("Invalid embedding vector size.");

    m_embeddings_flat.insert(m_embeddings_flat.end(), embedding.begin(), embedding.end());
    m_documents.push_back(text);

    m_num_samples++;
}

void ProxiFlat::save_state(const std::string &path_str) {

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