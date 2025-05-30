#include "proxi_knn.h"
#include <unordered_map>
#include <algorithm>
#include <omp.h>

ProxiKNN::ProxiKNN(const size_t n_neighbours, const size_t n_jobs,
                   const std::string &distance_function)
    : m_K(n_neighbours), m_num_jobs(n_jobs), m_objective_function_id(distance_function),
      base_model(std::make_unique<ProxiFlat>(n_neighbours, n_jobs, distance_function)), m_is_fitted(false) {}

void ProxiKNN::fit(const std::vector<std::vector<float>> &feature_vectors,
              const std::vector<float> &labels) {

    std::vector<std::string> label_strings(feature_vectors.size());
    std::transform(labels.begin(), labels.end(), label_strings.begin(),
                   [](const float label) { return std::to_string(label); });

    base_model->index_data(feature_vectors, label_strings);

    m_is_fitted = true;
}

float ProxiKNN::predict(const std::vector<float> &feature_vector) {
    std::vector<std::string> neighbours = base_model->find_docs(feature_vector);
    std::unordered_map<float, size_t> votes;
    for (const auto &neighbour : neighbours) {
        votes[std::strtof(neighbour.c_str(), nullptr)]++;
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