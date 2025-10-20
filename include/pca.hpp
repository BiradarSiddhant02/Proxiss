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

#ifndef PCA_HPP
#define PCA_HPP

#include "matrix.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

struct PCAResult {
    Matrix components; // Principal components (n_components x n_features)
    std::vector<float> explained_variance;
    std::vector<float> explained_variance_ratio;
    Matrix mean; // Mean of each feature (1 x n_features)

    PCAResult(size_t n_components, size_t n_features)
        : components(n_components, n_features), mean(1, n_features) {
        explained_variance.reserve(n_components);
        explained_variance_ratio.reserve(n_components);
    }
};

class PCA {
public:
    // Compute mean of each column (feature)
    static Matrix compute_mean(const Matrix &x) {
        size_t n_samples = x.rows();
        size_t n_features = x.cols();
        Matrix mean(1, n_features);

        for (size_t j = 0; j < n_features; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < n_samples; i++) {
                sum += x(i, j);
            }
            mean(0, j) = sum / n_samples;
        }
        return mean;
    }

    // Center the data (subtract mean)
    static Matrix center_data(const Matrix &x, const Matrix &mean) {
        size_t n_samples = x.rows();
        size_t n_features = x.cols();
        Matrix x_centered(n_samples, n_features);

#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < n_samples; i++) {
            for (size_t j = 0; j < n_features; j++) {
                x_centered(i, j) = x(i, j) - mean(0, j);
            }
        }
        return x_centered;
    }

    // Compute covariance matrix: Cov = (X^T * X) / (n - 1)
    static Matrix compute_covariance(const Matrix &x_centered) {
        size_t n_samples = x_centered.rows();
        Matrix x_t = x_centered.transpose();
        Matrix cov = x_t * x_centered;

        // Divide by (n - 1) for unbiased estimate
        float scale = 1.0f / (n_samples - 1);
        return cov * scale;
    }

    // Power iteration to find largest eigenvector
    static void power_iteration(const Matrix &a, Matrix &eigenvec, float &eigenval,
                                size_t max_iter = 100, float tol = 1e-6f) {
        size_t n = a.rows();
        eigenvec = Matrix(n, 1);
        eigenvec.randomize(-1.0f, 1.0f);

        // Normalize
        float norm = 0.0f;
        for (size_t i = 0; i < n; i++) {
            norm += eigenvec(i, 0) * eigenvec(i, 0);
        }
        norm = sqrtf(norm);
        for (size_t i = 0; i < n; i++) {
            eigenvec(i, 0) /= norm;
        }

        for (size_t iter = 0; iter < max_iter; iter++) {
            // v_new = A * v
            Matrix v_new(n, 1);
            for (size_t i = 0; i < n; i++) {
                float sum = 0.0f;
                for (size_t j = 0; j < n; j++) {
                    sum += a(i, j) * eigenvec(j, 0);
                }
                v_new(i, 0) = sum;
            }

            // Compute eigenvalue (Rayleigh quotient)
            float num = 0.0f, den = 0.0f;
            for (size_t i = 0; i < n; i++) {
                num += v_new(i, 0) * eigenvec(i, 0);
                den += eigenvec(i, 0) * eigenvec(i, 0);
            }
            eigenval = num / den;

            // Normalize
            norm = 0.0f;
            for (size_t i = 0; i < n; i++) {
                norm += v_new(i, 0) * v_new(i, 0);
            }
            norm = sqrtf(norm);

            if (norm < 1e-10f)
                break;

            for (size_t i = 0; i < n; i++) {
                v_new(i, 0) /= norm;
            }

            // Check convergence
            float diff = 0.0f;
            for (size_t i = 0; i < n; i++) {
                float d = v_new(i, 0) - eigenvec(i, 0);
                diff += d * d;
            }

            eigenvec = v_new;

            if (sqrtf(diff) < tol)
                break;
        }
    }

    // Fit PCA model
    static PCAResult fit(const Matrix &x, size_t n_components) {
        size_t n_samples = x.rows();
        size_t n_features = x.cols();

        if (n_components > n_features) {
            n_components = n_features;
        }

        PCAResult result(n_components, n_features);

        // 1. Compute and store mean
        result.mean = compute_mean(x);

        // 2. Center the data
        Matrix x_centered = center_data(x, result.mean);

        // 3. Compute covariance matrix
        Matrix cov = compute_covariance(x_centered);

        // 4. Find principal components using power iteration with deflation
        Matrix cov_work = cov;
        float total_variance = 0.0f;

        // Compute total variance (trace of covariance matrix)
        for (size_t i = 0; i < n_features; i++) {
            total_variance += cov(i, i);
        }

        for (size_t k = 0; k < n_components; k++) {
            Matrix eigenvec(n_features, 1);
            float eigenval = 0.0f;

            power_iteration(cov_work, eigenvec, eigenval);

            // Store component
            for (size_t j = 0; j < n_features; j++) {
                result.components(k, j) = eigenvec(j, 0);
            }

            // Store variance
            result.explained_variance.push_back(eigenval);
            result.explained_variance_ratio.push_back(eigenval / total_variance);

            // Deflate: subtract rank-1 approximation
            for (size_t i = 0; i < n_features; i++) {
                for (size_t j = 0; j < n_features; j++) {
                    cov_work(i, j) -= eigenval * eigenvec(i, 0) * eigenvec(j, 0);
                }
            }
        }

        return result;
    }

    // Transform data to principal component space
    static Matrix transform(const Matrix &x, const PCAResult &pca) {
        // Center the data
        Matrix x_centered = center_data(x, pca.mean);

        // Project: X_transformed = X_centered * components^T
        Matrix components_t = pca.components.transpose();
        return x_centered * components_t;
    }

    // Inverse transform (reconstruct original space)
    static Matrix inverse_transform(const Matrix &x_transformed, const PCAResult &pca) {
        // X_reconstructed = X_transformed * components + mean
        Matrix x_recon = x_transformed * pca.components;

        // Add back the mean
        size_t n_samples = x_recon.rows();
        size_t n_features = x_recon.cols();

#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < n_samples; i++) {
            for (size_t j = 0; j < n_features; j++) {
                x_recon(i, j) += pca.mean(0, j);
            }
        }

        return x_recon;
    }
};

#endif
