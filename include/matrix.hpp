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

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <omp.h>
#include <stdexcept>

class Matrix {
private:
    std::unique_ptr<float[]> m_data;
    size_t m_rows;
    size_t m_cols;

    static inline size_t min(size_t a, size_t b) { return a < b ? a : b; }

public:
    // Constructors
    Matrix(size_t rows, size_t cols)
        : m_data(new float[rows * cols]()), m_rows(rows), m_cols(cols) {}

    Matrix(const Matrix &other)
        : m_data(new float[other.m_rows * other.m_cols]), m_rows(other.m_rows),
          m_cols(other.m_cols) {
        std::memcpy(m_data.get(), other.m_data.get(), m_rows * m_cols * sizeof(float));
    }

    Matrix(Matrix &&other) noexcept
        : m_data(std::move(other.m_data)), m_rows(other.m_rows), m_cols(other.m_cols) {
        other.m_rows = 0;
        other.m_cols = 0;
    }

    ~Matrix() = default;

    Matrix &operator=(const Matrix &other) {
        if (this != &other) {
            m_data.reset(new float[other.m_rows * other.m_cols]);
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            std::memcpy(m_data.get(), other.m_data.get(), m_rows * m_cols * sizeof(float));
        }
        return *this;
    }

    Matrix &operator=(Matrix &&other) noexcept {
        if (this != &other) {
            m_data = std::move(other.m_data);
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            other.m_rows = 0;
            other.m_cols = 0;
        }
        return *this;
    }

    // Accessors
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
    float *data() { return m_data.get(); }
    const float *data() const { return m_data.get(); }

    float &operator()(size_t i, size_t j) { return m_data[i * m_cols + j]; }

    const float &operator()(size_t i, size_t j) const { return m_data[i * m_cols + j]; }

    // Optimized matrix multiplication
    Matrix operator*(const Matrix &other) const {
        if (m_cols != other.m_rows) {
            throw std::invalid_argument("Matrix dimensions incompatible");
        }

        Matrix result(m_rows, other.m_cols);
        const size_t k_dim = m_cols;
        const size_t block_i = 64;
        const size_t block_j = 64;
        const size_t block_k = 64;

        float *result_data = result.m_data.get();
        const float *this_data = m_data.get();
        const float *other_data = other.m_data.get();

#pragma omp parallel for collapse(2)
        for (size_t ii = 0; ii < m_rows; ii += block_i) {
            for (size_t jj = 0; jj < other.m_cols; jj += block_j) {
                for (size_t kk = 0; kk < k_dim; kk += block_k) {
                    size_t i_end = min(ii + block_i, m_rows);
                    size_t j_end = min(jj + block_j, other.m_cols);
                    size_t k_end = min(kk + block_k, k_dim);

                    for (size_t i = ii; i < i_end; i++) {
                        for (size_t k = kk; k < k_end; k++) {
                            float a_ik = this_data[i * k_dim + k];
#pragma omp simd
                            for (size_t j = jj; j < j_end; j++) {
                                result_data[i * other.m_cols + j] +=
                                    a_ik * other_data[k * other.m_cols + j];
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    // Transpose
    Matrix transpose() const {
        Matrix result(m_cols, m_rows);
        float *result_data = result.m_data.get();
        const float *this_data = m_data.get();
#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < m_rows; i++) {
            for (size_t j = 0; j < m_cols; j++) {
                result_data[j * m_rows + i] = this_data[i * m_cols + j];
            }
        }
        return result;
    }

    // Scalar operations
    Matrix operator*(float scalar) const {
        Matrix result(m_rows, m_cols);
        float *result_data = result.m_data.get();
        const float *this_data = m_data.get();
#pragma omp parallel for
        for (size_t i = 0; i < m_rows * m_cols; i++) {
            result_data[i] = this_data[i] * scalar;
        }
        return result;
    }

    Matrix operator+(const Matrix &other) const {
        if (m_rows != other.m_rows || m_cols != other.m_cols) {
            throw std::invalid_argument("Matrix dimensions don't match");
        }
        Matrix result(m_rows, m_cols);
        float *result_data = result.m_data.get();
        const float *this_data = m_data.get();
        const float *other_data = other.m_data.get();
#pragma omp parallel for
        for (size_t i = 0; i < m_rows * m_cols; i++) {
            result_data[i] = this_data[i] + other_data[i];
        }
        return result;
    }

    Matrix operator-(const Matrix &other) const {
        if (m_rows != other.m_rows || m_cols != other.m_cols) {
            throw std::invalid_argument("Matrix dimensions don't match");
        }
        Matrix result(m_rows, m_cols);
        float *result_data = result.m_data.get();
        const float *this_data = m_data.get();
        const float *other_data = other.m_data.get();
#pragma omp parallel for
        for (size_t i = 0; i < m_rows * m_cols; i++) {
            result_data[i] = this_data[i] - other_data[i];
        }
        return result;
    }

    void fill(float value) {
        float *this_data = m_data.get();
#pragma omp parallel for
        for (size_t i = 0; i < m_rows * m_cols; i++) {
            this_data[i] = value;
        }
    }

    void randomize(float min_val = 0.0f, float max_val = 1.0f) {
        float *this_data = m_data.get();
        for (size_t i = 0; i < m_rows * m_cols; i++) {
            this_data[i] = min_val + (max_val - min_val) * (rand() / (float)RAND_MAX);
        }
    }
};

#endif
