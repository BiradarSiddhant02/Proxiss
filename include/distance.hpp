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

    size_t length = A.size();
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= length; i += 8) {
        __m256 va = _mm256_loadu_ps(A.data() + i);
        __m256 vb = _mm256_loadu_ps(B.data() + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
    }

    float buffer[8];
    _mm256_storeu_ps(buffer, sum);
    float distance = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
                     buffer[6] + buffer[7];

    for (; i < length; i++) {
        float diff = A[i] - B[i];
        distance += diff * diff;
    }

    return std::sqrt(distance);
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

    size_t length = A.size();
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= length; i += 8) {
        __m256 va = _mm256_loadu_ps(A.data() + i);
        __m256 vb = _mm256_loadu_ps(B.data() + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 abs_diff = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff); // abs(x)
        sum = _mm256_add_ps(sum, abs_diff);
    }

    float buffer[8];
    _mm256_storeu_ps(buffer, sum);
    float distance = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
                     buffer[6] + buffer[7];

    for (; i < length; i++) {
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

    size_t length = A.size();
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= length; i += 8) {
        __m256 va = _mm256_loadu_ps(A.data() + i);
        __m256 vb = _mm256_loadu_ps(B.data() + i);
        __m256 prod = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, prod);
    }

    float buffer[8];
    _mm256_storeu_ps(buffer, sum);
    float dot_sum = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
                    buffer[6] + buffer[7];

    for (; i < length; i++) {
        dot_sum += A[i] * B[i];
    }

    return dot_sum;
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

    size_t length = A.size();
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= length; i += 8) {
        __m256 v = _mm256_loadu_ps(A.data() + i);
        sum = _mm256_fmadd_ps(v, v, sum);
    }

    float buffer[8];
    _mm256_storeu_ps(buffer, sum);
    float sum_of_squares = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] +
                           buffer[6] + buffer[7];

    for (; i < length; i++) {
        sum_of_squares += A[i] * A[i];
    }

    return std::sqrt(sum_of_squares);
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
