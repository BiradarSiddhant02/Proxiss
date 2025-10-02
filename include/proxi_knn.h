#ifndef PROXI_KNN_H
#define PROXI_KNN_H

#include "proxi_flat.h"
#include <memory>
#include <string>
#include <vector>

class ProxiKNN {
private:
    size_t m_K;
    size_t m_num_jobs;
    const std::string m_objective_function_id;
    std::unique_ptr<ProxiFlat> base_model;
    std::vector<float> m_labels;
    bool m_is_fitted;

public:
    /**
     * @brief Constructor for the class ProxiKNN. A module KNN algorithm
     * @param n_neighbours The number of neighbours to use
     * @param n_jobs number of parallel jobs to run during prediction
     * @param distance_function Distance function to use {L2, L1, cos}
     */
    ProxiKNN(const size_t n_neighbours, const size_t n_jobs, const std::string &distance_function);

    /**
     * @brief Trains the model in the provided feature vectors and their respective labels
     * @param feature_vectors list of vectors i.e., training data
     * @param labels list of labels
     */
    void fit(const std::vector<std::vector<float>> &feature_vectors,
             const std::vector<float> &labels);

    /**
     * @brief Predicts the class labels of the provided data
     * @param feature_vector Test sample
     * @return class of the input sample
     */
    float predict(const std::vector<float> &feature_vector);
    std::vector<float> predict(const std::vector<std::vector<float>> &feature_vectors);

    /**
     * @brief Persists the trained model state to the specified directory.
     * @param path Directory where the state should be saved.
     * @throws std::runtime_error if the model has not been fitted or if saving fails.
     */
    void save_state(const std::string &path);

    /**
     * @brief Loads a previously persisted model state from the specified directory.
     * @param path Directory containing the saved state.
     * @throws std::runtime_error if loading fails or if metadata is incompatible with the
     * current configuration.
     */
    void load_state(const std::string &path);
};

#endif // PROXI_KNN_H