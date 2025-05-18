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

#include <cmath>
#include <span>
#include <vector>

namespace distance {

// Euclidean Distance
template <typename T> T euclidean(std::span<const T> A, std::span<const T> B) noexcept {
    /**
     * @brief Templated class to find euclidean distance between two input
     * vectors
     *
     * @tparam T The data type of the vector elements.
     * @param A Vector A
     * @param B Vector B
     *
     * @return euclidean distance between A and B of type T
     */

    T distance = static_cast<T>(0.0f);

    size_t length = A.size();
    for (size_t i = 0; i < length; i++) {
        T diff = A[i] - B[i];
        distance += diff * diff;
    }

    return static_cast<T>(std::sqrt(distance));
}

// Manhattan Distance
template <typename T> T manhattan(std::span<const T> A, std::span<const T> B) noexcept {
    /**
     * @brief Computes the Manhattan distance (L1 norm) between two vectors A
     * and B.
     *
     * @tparam T The data type of the vector elements.
     * @param A First vector.
     * @param B Second vector.
     * @return The Manhattan distance between A and B.
     */

    T distance = static_cast<T>(0.0f);

    size_t length = A.size();
    for (size_t i = 0; i < length; i++) {
        T diff = A[i] - B[i];
        distance += std::abs(diff);
    }

    return distance;
}

// Helper Function to calculate Dot Product
template <typename T> T dot(std::span<const T> A, std::span<const T> B) noexcept {
    /**
     * @brief Computes the dot product of two vectors A and B.
     *
     * @tparam T The data type of the vector elements.
     * @param A First vector.
     * @param B Second vector.
     * @return The dot product of A and B.
     */

    T result = static_cast<T>(0.0f);

    size_t length = A.size();
    for (size_t i = 0; i < length; i++) {
        result += A[i] * B[i];
    }

    return result;
}

// Helper function to calculate l2 norm
template <typename T> T l2_norm(std::span<const T> A) noexcept {
    /**
     * @brief Computes the L2 norm (Euclidean norm) of a vector A.
     *
     * @tparam T The data type of the vector elements.
     * @param A The input vector.
     * @return The L2 norm of A.
     */

    T norm = static_cast<T>(0.0f);

    for (const T ele : A) {
        norm += ele * ele;
    }

    return static_cast<T>(std::sqrt(norm));
}

// Cosine Similarity
template <typename T> T cosine(std::span<const T> A, std::span<const T> B) noexcept {
    /**
     * @brief Computes the cosine similarity between two vectors A and B.
     * Cosine similarity is defined as (A . B) / (||A|| * ||B||).
     *
     * @tparam T The data type of the vector elements.
     * @param A First vector.
     * @param B Second vector.
     * @return The cosine similarity between A and B. Returns NaN if either norm
     * is zero.
     */

    const T dotAB = distance::dot(A, B);
    const T normA = distance::l2_norm(A);
    const T normB = distance::l2_norm(B);

    return dotAB / (normA * normB);
}

}; // namespace distance