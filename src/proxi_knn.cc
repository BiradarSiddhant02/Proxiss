#include "proxi_knn.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <omp.h>
#include <stdexcept>
#include <unordered_map>

ProxiKNN::ProxiKNN(const size_t n_neighbours, const size_t n_jobs,
                   const std::string &distance_function)
    : m_K(n_neighbours), m_num_jobs(n_jobs), m_objective_function_id(distance_function),
      base_model(std::make_unique<ProxiFlat>(n_neighbours, n_jobs, distance_function)),
      m_is_fitted(false) {}

void ProxiKNN::fit(const std::vector<std::vector<float>> &feature_vectors,
                   const std::vector<float> &labels) {

    if (feature_vectors.size() != labels.size())
        throw std::runtime_error("Size of feature vectors and labels must be equal");

    m_labels = labels;
    base_model->index_data(feature_vectors);

    m_is_fitted = true;
}

float ProxiKNN::predict(const std::vector<float> &feature_vector) {
    std::vector<size_t> neighbour_indices = base_model->find_indices(feature_vector);
    std::unordered_map<float, size_t> votes;

    for (const auto &index : neighbour_indices) {
        votes[m_labels[index]]++;
    }
    float best_value = -1.0f;
    size_t max_votes = 0;

    for (const auto &[label, count] : votes) {
        if (count > max_votes) {
            best_value = label;
            max_votes = count;
        }
    }

    return best_value;
}

std::vector<float> ProxiKNN::predict(const std::vector<std::vector<float>> &feature_vectors) {
    omp_set_num_threads(m_num_jobs);
    std::vector<float> results(feature_vectors.size());

#pragma omp parallel for
    for (int i = 0; i < feature_vectors.size(); i++) {
        results[i] = predict(feature_vectors[i]);
    }

    return results;
}

void ProxiKNN::save_state(const std::string &path) {
    if (!m_is_fitted)
        throw std::runtime_error("Model has not been fitted. Fit before saving state.");

    base_model->save_state(path);
    
    // Save labels separately
    std::filesystem::path dir(path);
    std::filesystem::path labels_path = dir / "knn_labels.bin";
    std::ofstream labels_file(labels_path, std::ios::binary);
    if (!labels_file.is_open())
        throw std::runtime_error("Failed to open labels file for writing: " + labels_path.string());
    
    size_t num_labels = m_labels.size();
    labels_file.write(reinterpret_cast<const char*>(&num_labels), sizeof(size_t));
    labels_file.write(reinterpret_cast<const char*>(m_labels.data()), num_labels * sizeof(float));
    
    if (!labels_file.good())
        throw std::runtime_error("Failed to write labels file: " + labels_path.string());
}

void ProxiKNN::load_state(const std::string &path) {
    // ProxiFlat expects a file path, so construct data.bin path
    std::filesystem::path dir(path);
    std::filesystem::path data_file = dir / "data.bin";
    base_model->load_state(data_file.string());
    
    // Load labels separately
    std::filesystem::path labels_path = dir / "knn_labels.bin";
    std::ifstream labels_file(labels_path, std::ios::binary);
    if (!labels_file.is_open())
        throw std::runtime_error("Failed to open labels file for reading: " + labels_path.string());
    
    size_t num_labels = 0;
    labels_file.read(reinterpret_cast<char*>(&num_labels), sizeof(size_t));
    if (!labels_file.good())
        throw std::runtime_error("Failed to read labels count: " + labels_path.string());
    
    m_labels.resize(num_labels);
    labels_file.read(reinterpret_cast<char*>(m_labels.data()), num_labels * sizeof(float));
    if (!labels_file.good())
        throw std::runtime_error("Failed to read labels data: " + labels_path.string());
    
    m_is_fitted = true;
}