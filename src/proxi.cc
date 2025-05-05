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

// --- proxi.cc --- //

#include "proxi.h"

// PROTECTED
float Proxi::euclidean(std::span<const float> A, std::span<const float> B) noexcept {
    /**
     * @brief Computes the Euclidean distance (L2 norm) between two vectors A and B.
     * 
     * @param A First vector.
     * @param B Second vector.
     * @return The Euclidean distance between A and B.
     */
    float distance = 0.0f;
    
    size_t length = A.size();
    for (size_t i = 0; i < length; i++) {
        float diff = A[i] - B[i];
        distance += diff * diff;
    }

    return std::sqrt(distance);
}

float Proxi::manhattan(std::span<const float> A, std::span<const float> B) noexcept {
    /**
     * @brief Computes the Manhattan distance (L1 norm) between two vectors A and B.
     * 
     * @param A First vector.
     * @param B Second vector.
     * @return The Manhattan distance between A and B.
     */
    float distance = 0.0f;

    size_t length = A.size();
    for (size_t i = 0; i < length; i++) {
        float diff = A[i] - B[i];
        distance += std::abs(diff);
    }

    return distance;
}

float Proxi::cosine(std::span<const float> A, std::span<const float> B) noexcept {
    /**
     * @brief Computes the cosine similarity between two vectors A and B.
     * Computes A * B / ||A|| ||B||
     * 
     * @param A First vector
     * @param B Second vector
     * 
     * @return Cosine similarity (float value in [-1, 1])
     */
    const float dot_product = dot(A, B);
    const float norm_A = l2_norm(A);
    const float norm_B = l2_norm(B);

    // Handle case where norm of one of the vector is 0
    if (norm_A == 0.0f || norm_B == 0.0f) 
        return std::numeric_limits<float>::quiet_NaN();

    return dot_product / (norm_A * norm_B);
}

float Proxi::l2_norm(std::span<const float> vec) noexcept {
    /**
     * @brief Computes the L2 norm (Euclidean norm) of a vector.
     * 
     * @param vec The input vector.
     * @return The L2 norm of the vector.
     */
    float norm = 0.0f;

    for (const float ele : vec) { norm += ele * ele; }

    return std::sqrt(norm);
}

float Proxi::dot(std::span<const float> A, std::span<const float> B) noexcept {
    /**
     * @brief Computes the dot product of two vectors A and B.
     * 
     * @param A First vector.
     * @param B Second vector.
     * @return The dot product of A and B.
     */
    float sum = 0.0f;

    size_t length = A.size();
    for (size_t i = 0; i < length; i++) { sum += A[i] * B[i]; }

    return sum;
}

// PRIVATE
std::vector<size_t> Proxi::m_get_neighbours(const std::vector<float>& query) noexcept {
    /**
     * @brief Finds the indices of the K nearest neighbours for a given query vector.
     * Uses the objective function defined during Proxi initialization.
     * 
     * @param query The query vector.
     * @return A vector containing the indices of the K nearest neighbours.
     */
    using pair = std::pair<float, size_t>;

    std::priority_queue<pair> heap;

    long long int num_samples = static_cast<long long int>(m_num_samples);
    long long int num_features = static_cast<long long int>(m_num_features);

    #pragma omp for nowait
    for (long long int i = 0; i < num_samples; i++) {
        float distance = m_objective_function(
            std::span<const float>(query),
            std::span<const float>(m_embeddings_flat).subspan(i * num_features, num_features)
        );

        bool should_insert = heap.size() < m_K;
        bool should_replace = !should_insert && distance < heap.top().first;

        if (should_insert || should_replace) {
            if(should_replace) heap.pop();
            heap.emplace(distance, i);
        }
    }

    std::vector<size_t> indices;
    while(!heap.empty()) {
        indices.push_back(heap.top().second);
        heap.pop();
    }

    return indices;
}

// PUBLIC
Proxi::Proxi(const size_t k, const size_t num_threads, const std::string objective_function) 
:   m_num_samples(0),
    m_num_features(0),
    m_K(k), 
    m_is_indexed(false),
    m_num_threads(num_threads) {

    if (objective_function == "l2"){
        m_objective_function = euclidean;
    } else if (objective_function == "l1") {
        m_objective_function = manhattan;
    } else if (objective_function == "cos") {
        m_objective_function = cosine;
    } else {
        throw std::invalid_argument("Invalid Distance function.");
    }
}

void Proxi::index_data(const std::vector<std::vector<float>>& embeddings, const std::vector<std::string>& documents) {
    /**
     * @brief Indexes the provided embeddings and corresponding documents.
     * Stores the data internally for subsequent nearest neighbour searches.
     * 
     * @param embeddings A vector of vectors, where each inner vector is an embedding.
     * @param documents A vector of strings, where each string is a document corresponding to an embedding.
     * @throws std::invalid_argument If embeddings or documents are empty, or if their sizes are unequal.
     * @throws std::runtime_error If the number of features is inconsistent across embeddings.
     */
    if (embeddings.empty() || documents.empty())
        throw std::invalid_argument("Embeddings or Documents cannot be empty.");

    if (embeddings.size() != documents.size()) 
        throw std::invalid_argument("Size of embeddings and corpus are unequal");

    m_documents = documents;

    m_num_samples = documents.size();
    m_num_features = embeddings[0].size();

    m_embeddings_flat.resize(m_num_features * m_num_samples);
    for (size_t i = 0; i < m_num_samples; i++) {
        if (embeddings[i].size() != m_num_features)
            throw std::runtime_error("Number of features is inconsistent.");

        std::memcpy(
            &m_embeddings_flat[i * m_num_features],
            embeddings[i].data(),
            m_num_features * sizeof(float)
        );
    }

    m_is_indexed = true;
}

std::vector<size_t> Proxi::find_indices(const std::vector<float>& query) {
    /**
     * @brief Finds the indices of the K nearest neighbours for a single query vector.
     * 
     * @param query The query vector.
     * @return A vector containing the indices of the K nearest neighbours.
     */
    return m_get_neighbours(query);
}

std::vector<std::vector<size_t>> Proxi::find_indices(const std::vector<std::vector<float>>& queries) {
    /**
     * @brief Finds the indices of the K nearest neighbours for multiple query vectors in parallel.
     * 
     * @param queries A vector of query vectors.
     * @return A vector of vectors, where each inner vector contains the indices of the K nearest neighbours for the corresponding query.
     * @throws std::runtime_error If a query vector has an inconsistent number of features.
     */
    omp_set_num_threads(m_num_threads);

    std::vector<std::vector<size_t>> indices(queries.size());

    long long int num_queries = static_cast<long long int>(queries.size());

    #pragma omp parallel for
    for (long long int i = 0; i < num_queries; i++) {
        if (queries[i].size() != m_num_features)
            throw std::runtime_error("Inconsistent number of features.");

        indices[i] = find_indices(queries[i]);
    }

    return indices;
}

std::vector<std::string> Proxi::find_docs(const std::vector<float>& query) {
    /**
     * @brief Finds the documents corresponding to the K nearest neighbours for a single query vector.
     * 
     * @param query The query vector.
     * @return A vector of strings, where each string is a document corresponding to a nearest neighbour.
     */
    std::vector<size_t> neighbours = m_get_neighbours(query);

    std::vector<std::string> docs;
    for (const size_t index : neighbours) { docs.push_back(m_documents[index]); }

    return docs;
}

std::vector<std::vector<std::string>> Proxi::find_docs(const std::vector<std::vector<float>>& queries) {
    /**
     * @brief Finds the documents corresponding to the K nearest neighbours for multiple query vectors in parallel.
     * 
     * @param queries A vector of query vectors.
     * @return A vector of vectors, where each inner vector contains the documents corresponding to the K nearest neighbours for the corresponding query.
     * @throws std::runtime_error If a query vector has an inconsistent number of features.
     */
    omp_set_num_threads(m_num_threads);

    std::vector<std::vector<std::string>> results(queries.size());

    long long int num_queries = static_cast<long long int>(queries.size());

    #pragma omp parallel for
    for (long long int i = 0; i < num_queries; i++) {
        if (queries[i].size() != m_num_features)
            throw std::runtime_error("Inconsistent number of features.");

        results[i] = find_docs(queries[i]);
    }

    return results;
}

