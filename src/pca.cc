#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

#include "pca.h"

// PUBLIC
PCA::PCA(const size_t num_components) : m_num_components(num_components), m_is_fitted(false) {}

void PCA::fit(const matrix &mat) {

    // Center the data
    m_mean = mat.colwise().mean();
    matrix x_centered = mat.rowwise() - m_mean;

    // Calculate covariance matrix
    m_variance = (x_centered.adjoint() * x_centered) / float(mat.rows() - 1);

    // Eigen decomposition of covariance matrix
    Eigen::SelfAdjointEigenSolver<matrix> eig_solver(m_variance);

    if (eig_solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed.");
    }

    // Extract and sort eigenvalues and eigenvectors (descending order)
    vector eigen_vals = eig_solver.eigenvalues();
    matrix eigen_vecs = eig_solver.eigenvectors();

    // Reverse to get descending order (largest eigenvalues first)
    m_eigen_values = eigen_vals.reverse();
    m_eigen_vectors = eigen_vecs.rowwise().reverse();

    // Store the top m_num_components principal components
    m_components = m_eigen_vectors.leftCols(m_num_components);

    // Mark as fitted
    m_is_fitted = true;
}

void PCA::fit(const std::vector<std::vector<float>> &mat) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();

    matrix e_mat(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            e_mat(i, j) = mat[i][j];
        }
    }

    fit(e_mat);
}

matrix PCA::transform(const matrix &mat) {
    if (!m_is_fitted) {
        throw std::runtime_error("PCA must be fitted before transform. Call fit() first.");
    }

    // Center the data using the mean from fit
    matrix x_centered = mat.rowwise() - m_mean;

    // Transform: X_transformed = X_centered * components
    return x_centered * m_components;
}

matrix PCA::transform(const std::vector<std::vector<float>> &mat) {
    size_t rows = mat.size();
    size_t cols = mat[0].size();

    matrix e_mat(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            e_mat(i, j) = mat[i][j];
        }
    }

    return transform(e_mat);
}

matrix PCA::inverse_transform(const matrix &reduced_mat) const {
    if (!m_is_fitted) {
        throw std::runtime_error("PCA must be fitted before inverse_transform. Call fit() first.");
    }

    // Inverse transform: X_reconstructed = reduced_mat * components^T + mean
    matrix reconstructed = reduced_mat * m_components.transpose();
    reconstructed = reconstructed.rowwise() + m_mean;

    return reconstructed;
}

matrix PCA::inverse_transform(const std::vector<std::vector<float>> &reduced_mat) const {
    size_t rows = reduced_mat.size();
    size_t cols = reduced_mat[0].size();

    matrix e_mat(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            e_mat(i, j) = reduced_mat[i][j];
        }
    }

    return inverse_transform(e_mat);
}

// Accessor implementations
const matrix &PCA::components() const {
    if (!m_is_fitted) {
        throw std::runtime_error("PCA must be fitted before accessing components.");
    }
    return m_components;
}

const vector &PCA::mean() const {
    if (!m_is_fitted) {
        throw std::runtime_error("PCA must be fitted before accessing mean.");
    }
    // Return as a static member to avoid returning reference to temporary
    static vector mean_vec;
    mean_vec = m_mean.transpose();
    return mean_vec;
}

const matrix &PCA::variance() const {
    if (!m_is_fitted) {
        throw std::runtime_error("PCA must be fitted before accessing variance.");
    }
    return m_variance;
}

const size_t PCA::num_components() const {
    return m_num_components;
}

const bool PCA::is_fitted() const {
    return m_is_fitted;
}