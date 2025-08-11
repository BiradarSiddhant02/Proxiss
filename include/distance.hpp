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

// --- distance.hpp --- //

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <immintrin.h>
#include <span>

inline float euclidean_distance(std::span<const float> A, std::span<const float> B) {
    /**
     * @brief Calculates Euclidean distance between the given input vectors.
     *
     * @param A Input vector A
     * @param B Input vector B
     *
     * @returns A float which is the euclidean distance between the two input vectors.
     */

    float distance = 0.0f;
    size_t len = A.size();

#pragma omp simd reduction(+:distance)
    for (size_t i = 0; i < len; i++) {
        float diff = A[i] - B[i];
        distance += diff * diff;
    }

    return std::sqrtf(distance);
}

inline float manhattan_distance(std::span<const float> A, std::span<const float> B) {
    /**
     * @brief Calculates Manhattan distance between the given input vectors.
     *
     * @param A Input vector A
     * @param B Input vector B
     *
     * @returns A float which is the euclidean distance between the two input vectors.
     */

    float distance = 0.0f;
    size_t len = A.size();

#pragma omp simd reduction(+:distance)
    for (size_t i = 0; i < len; i++) {
        distance += std::fabs(A[i] - B[i]);
    }

    return distance;

}

inline float dot(std::span<const float> A, std::span<const float> B) {
    /**
     * @brief Calculates Dot Product between the given input vectors.
     *
     * @param A Input vector A
     * @param B Input vector B
     *
     * @returns A float which is the euclidean distance between the two input vectors.
     */

    float dot = 0.0f;
    size_t len = A.size();

#pragma omp simd reduction(+:dot)
    for (size_t i = 0; i < len; i++)
        dot += A[i] * B[i];

    return dot;
}

inline float l2_norm(std::span<const float> A) {
    /**
     * @brief Calculates L2-Norm of the given input vector.
     *
     * @param A Input vector A
     * @param B Input vector B
     *
     * @returns A float which is the euclidean distance between the two input vectors.
     */

    return euclidean_distance(A, A);
}

inline float cosine_similarity(std::span<const float> A, std::span<const float> B) {
    /**
     * @brief Calculates Cosine Similarity between the given input vectors.
     *
     * @param A Input vector A
     * @param B Input vector B
     *
     * @returns A float which is the euclidean distance between the two input vectors.
     */

    float dotAB = dot(A, B);
    float normA = l2_norm(A);
    float normB = l2_norm(B);
    return dotAB / (normA * normB);
}
