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

// --- proxi_cluster.cc --- //
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <execution>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <queue>
#include <random>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <proxi_flat.h>
#include <distance.hpp>
#include <proxi_cluster.h>

// PROTECTED

// PRIVATE
void ProxiCluster::map_to_heads(const std::vector<std::vector<float>> &embeddings) {
    /**
     * @brief Assigns each embedding to the nearest cluster head.
     *
     * Algorithm:
     *   for each sample i in [0, m_num_samples):
     *       for each cluster head j in [0, m_num_clusters):
     *           compute distance(embedding[i], cluster_head[j])
     *       find j with minimum distance
     *       store mapping: m_index_head_map[i] = j
     *
     * Parallelization strategy:
     *   - Parallelize over 'i' (outer loop) to avoid races on shared variables.
     *   - 'distances' is private to each thread.
     */

    // Parallelize over samples, each thread works on one embedding at a time
#pragma omp parallel for
    for (size_t i = 0; i < m_num_samples; i++) {
        std::vector<float> distances(m_num_clusters);
        auto embedding = std::span<const float>(embeddings[i]);

        for (size_t j = 0; j < m_num_clusters; j++) {
            distances[j] =
                m_distance_function(embedding, std::span<const float>(m_cluster_heads[j]));
        }

        // Find index of minimum distance
        m_index_head_map[i] =
            std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));
    }
}

void ProxiCluster::recalculate_heads(const std::vector<std::vector<float>> &embeddings) {
    /**
     * @brief Recomputes each cluster head as the mean of all assigned embeddings.
     *
     * Parallelization strategy:
     * 1. Use thread-local accumulation to avoid races when summing embeddings per head.
     * 2. Reduce thread-local results into global accumulation.
     * 3. Parallelize averaging of heads (no dependencies).
     */

    // Thread-local storage for partial sums and counts
    int nthreads = 1;
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }

    std::vector<std::vector<std::vector<double>>> local_accum(
        nthreads,
        std::vector<std::vector<double>>(m_num_clusters, std::vector<double>(m_num_features, 0.0)));

    std::vector<std::vector<size_t>> local_counts(nthreads, std::vector<size_t>(m_num_clusters, 0));

    // Phase 1: Parallel accumulation into thread-local buffers
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp for
        for (size_t idx = 0; idx < m_index_head_map.size(); ++idx) {
            const auto &[index, head] = *(std::next(m_index_head_map.begin(), idx));
            auto &sum_vec = local_accum[tid][head];
            const auto &embed = embeddings[index];

            for (size_t f = 0; f < m_num_features; ++f) {
                sum_vec[f] += static_cast<double>(embed[f]);
            }
            local_counts[tid][head]++;
        }
    }

    // Phase 2: Reduce thread-local sums into global accumulation
    std::vector<std::vector<double>> accumulation(m_num_clusters,
                                                  std::vector<double>(m_num_features, 0.0));
    std::vector<size_t> embedding_counts(m_num_clusters, 0);

    for (int t = 0; t < nthreads; ++t) {
        for (size_t c = 0; c < m_num_clusters; ++c) {
            embedding_counts[c] += local_counts[t][c];
            for (size_t f = 0; f < m_num_features; ++f) {
                accumulation[c][f] += local_accum[t][c][f];
            }
        }
    }

    // Phase 3: Parallel averaging into m_cluster_heads
#pragma omp parallel for
    for (size_t i = 0; i < m_num_clusters; i++) {
        if (embedding_counts[i] > 0) {
            float inv_count = 1.0f / static_cast<float>(embedding_counts[i]);
            for (size_t f = 0; f < m_num_features; ++f) {
                m_cluster_heads[i][f] = static_cast<float>(accumulation[i][f] * inv_count);
            }
        }
    }
}

std::vector<size_t> ProxiCluster::get_neighbours(const std::vector<float> &query) {
    // Find the cloest cluster head
    std::vector<float> distances(m_num_clusters, 0.0f);

#pragma omp for
    for (size_t i = 0; i < m_num_clusters; i++)
        distances[i] = m_distance_function(std::span<const float>(query),
                                           std::span<const float>(m_cluster_heads[i]));

    size_t closest =
        std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));

    // Get indices of neighbours
    return m_flat_indexes[closest].find_indices(query);
}

// PUBLIC
ProxiCluster::ProxiCluster(const size_t k, size_t n, const size_t t, const size_t it,
                           const std::string distance_function)
    : m_num_clusters(n), m_num_samples(0), m_num_features(0), m_K(k), m_num_jobs(t),
      m_num_iterations(it), m_is_indexed(false), m_distance_function_name(distance_function) {

    if (distance_function == "l2") {
        m_distance_function = euclidean_distance;
    } else if (distance_function == "l1") {
        m_distance_function = manhattan_distance;
    } else if (distance_function == "cos") {
        m_distance_function = cosine_similarity;
    } else {
        throw std::runtime_error("Invalid Distance function.");
    }
}

void ProxiCluster::index_data(const std::vector<std::vector<float>> &embeddings,
                              const std::vector<std::string> &documents,
                              const std::vector<float> &lower_bounds,
                              const std::vector<float> &upper_bounds) {
    // Find out dimensionality
    m_num_samples = embeddings.size();
    m_num_features = embeddings[0].size();

    m_documents = documents;

    // Generate random-number-generators
    for (size_t i = 0; i < m_num_features; i++)
        m_rngs.push_back(RandomFloat(lower_bounds[i], upper_bounds[i]));
    m_rngs.shrink_to_fit();

    // initialize random cluster heads
    for (size_t i = 0; i < m_num_clusters; i++) {
        std::vector<float> head(m_num_features, 0.0f);
        for (size_t j = 0; j < m_num_features; j++) {
            head[j] = m_rngs[j]();
        }
        m_cluster_heads.push_back(head);
    }
    m_cluster_heads.shrink_to_fit();

    // Fit the index
    for (size_t i = 0; i < m_num_iterations; i++) {
        map_to_heads(embeddings);
        recalculate_heads(embeddings);
    }

    // Collect all embeddings per cluster and build flat indices
    std::vector<size_t> num_embeddings_per_head(m_num_clusters, 0);
    for (const auto &[_, head] : m_index_head_map)
        num_embeddings_per_head[head] += 1;

    std::vector<std::vector<size_t>> indices_per_cluster(m_num_clusters);
    for (const auto &[index, head] : m_index_head_map) {
        indices_per_cluster[head].push_back(index);
    }

    m_flat_indexes.clear();
    m_flat_indexes.reserve(m_num_clusters);
    for (size_t i = 0; i < m_num_clusters; i++) {
        std::vector<std::vector<float>> embeddings_local;
        std::vector<std::string> documents_local;
        for (const auto &idx : indices_per_cluster[i]) {
            embeddings_local.push_back(embeddings[idx]);
            documents_local.push_back("");
        }
        m_flat_indexes.emplace_back(m_K, m_num_jobs, m_distance_function_name);
        m_flat_indexes[i].index_data(embeddings_local, documents_local);
    }

    m_is_indexed = true;
}

std::vector<size_t> ProxiCluster::find_indices(const std::vector<float> &query) {
    return get_neighbours(query);
}

std::vector<std::vector<size_t>>
ProxiCluster::find_indices(const std::vector<std::vector<float>> &queries) {

    omp_set_num_threads(m_num_jobs);

    std::vector<std::vector<size_t>> indices(queries.size());

    long long int num_queries = static_cast<long long int>(queries.size());

#pragma omp parallel for
    for (long long int i = 0; i < num_queries; i++) {
        indices[i] = find_indices(queries[i]);
    }

    return indices;
}

std::vector<std::string> ProxiCluster::find_docs(const std::vector<float> &query) {
    std::vector<std::string> docs;
    auto indices = find_indices(query);
    for (const auto& index : indices) {
        docs.push_back(m_documents[index]);
    }
    return docs;
}

std::vector<std::vector<std::string>> ProxiCluster::find_docs(const std::vector<std::vector<float>> &queries) {

    omp_set_num_threads(m_num_jobs);

    std::vector<std::vector<std::string>> results(queries.size());

    long long int num_queries = static_cast<long long int>(queries.size());

#pragma omp parallel for
    for (long long int i = 0; i < num_queries; i++) {
        results[i] = find_docs(queries[i]);
    }

    return results;
}