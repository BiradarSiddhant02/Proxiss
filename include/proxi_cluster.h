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

// --- proxi_cluster.h --- //

#pragma once

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

class RandomFloat {
private:
    const float m_lower;
    const float m_upper;
    std::mt19937 m_gen;
    std::uniform_real_distribution<float> dist;

public:
    RandomFloat(const float lower = -1.0f, const float upper = 1.0f)
        : m_lower(lower), m_upper(upper), m_gen(std::random_device{}()), dist(lower, upper) {}

    float operator()() { return dist(m_gen); }
};

class ProxiCluster {
protected:
private:
    // List of all cluster heads
    std::vector<std::vector<float>> m_cluster_heads;

    // Map between index of embeddings to index of cluster heads
    std::unordered_map<std::size_t, std::size_t> m_index_head_map;

    // List of all embedding vectors
    // std::vector<std::vector<float>> m_embeddings;

    // List of all documents associated with the embeddings
    std::vector<std::string> m_documents;

    // Random Number generators
    std::vector<RandomFloat> m_rngs;

    // ProxiFlat indexes of all the clusters
    std::vector<ProxiFlat> m_flat_indexes;

    // Number of cluster heads
    std::size_t m_num_clusters;

    // Number of embedding-documents pairs
    std::size_t m_num_samples;

    // Dimensionality of embeddings
    std::size_t m_num_features;

    // Number of neighbours to retreive
    std::size_t m_K;

    // Number of parallel jobs to run
    std::size_t m_num_jobs;

    // Number of iterations to run k-means algorithm for
    std::size_t m_num_iterations;

    // Flag to determine if the data is already indexed
    bool m_is_indexed;

    // Name of the distance function
    std::string m_distance_function_name;

    // Distance Function
    std::function<float(std::span<const float>, std::span<const float>)> m_distance_function;

    /**
     * @brief Private method to map individual embeddings to their respective
     * cluster heads based on their distances from the heads. Does exactly one iteration
     */
    void map_to_heads(const std::vector<std::vector<float>> &embeddings);

    /**
     * @brief Private method to recalculate the cluster heads. Must be called after mapping all
     * the embeddings to their respective heads.
     */
    void recalculate_heads(const std::vector<std::vector<float>> &embeddings);

    /**
     * @brief Finds the neighbours of the given query
     * @param query Embedding vector, whose neighbours are to be calculated
     * @return List of indices of neighbours
     */
    std::vector<size_t> get_neighbours(const std::vector<float> &query);

public:
    /**
     * @brief Constructor for the ProxiCluster class.
     *
     * @param k                 Number of neighbours to retreive
     * @param n                 Number of cluster heads. Defaults to sqrt(num_samples) if set to 0
     * @param t                 Number of parallel threads to run
     * @param it                Number of iterations. Defaults to 10
     * @param distance_function Distance Function to be used. Defaults to `l2`
     */
    ProxiCluster(const size_t k, const size_t n, const size_t t, const size_t it = 10,
                 const std::string distance_function = "l2");

    /**
     * @brief Public method to index the data. Classifies embeddings into
     * their respective clusters and builds a ProxiFlat index for all of them
     *
     * @param embeddings    List of embedding vectors
     * @param documents     List of test associated with the corresponsing embedding
     * @param lower_bound   Lower bound of the RNG used for generating random cluster heads
     * @param upper_bound   Upper bound of the RNG used for generating random cluster heads
     */
    void index_data(const std::vector<std::vector<float>> &embeddings,
                    const std::vector<std::string> &documents,
                    const std::vector<float> &lower_bounds, const std::vector<float> &upper_bounds);

    /**
     * @brief Public method to find indices of the embeddings cloest to the qeury
     * @param query The query embedding vector
     * @return List of indices of all enarby embedding vectors
     */
    std::vector<size_t> find_indices(const std::vector<float> &query);

    /**
     * @brief Public method to find indices of the embeddings cloest to the qeury
     * @param queries List of the query embedding vector
     * @return List of indices of all enarby embedding vectors
     */
    std::vector<std::vector<size_t>> find_indices(const std::vector<std::vector<float>> &queries);

    /**
     * @brief Public method to find docs of the embeddings cloest to the qeury
     * @param query The query embedding vector
     * @return List of indices of all enarby embedding vectors
     */
    std::vector<std::string> find_docs(const std::vector<float> &query);

    /**
     * @brief Public method to find docs of the embeddings cloest to the qeury
     * @param query List of the query embedding vector
     * @return List of indices of all enarby embedding vectors
     */
    std::vector<std::vector<std::string>> find_docs(const std::vector<std::vector<float>> &queries);
};