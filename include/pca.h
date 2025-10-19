#include <Eigen/Dense>
#include <vector>

using matrix = Eigen::MatrixXf;
using vector = Eigen::VectorXf;
using row_vector = Eigen::RowVectorXf;

class PCA {
protected:
public:

    PCA(const size_t num_components);

    void fit(const matrix &mat);
    void fit(const std::vector<std::vector<float>> &mat);

    matrix transform(const matrix &mat);
    matrix transform(const std::vector<std::vector<float>> &mat);

    matrix inverse_transform(const matrix &reduced_mat) const;
    matrix inverse_transform(const std::vector<std::vector<float>> &reduced_mat) const;

    // accessors
    const matrix &components() const;
    const vector &mean() const;
    const matrix &variance() const;
    const size_t num_components() const;
    const bool is_fitted() const;

private:
    row_vector m_mean;
    matrix m_components;

    matrix m_variance;
    const size_t m_num_components;

    vector m_eigen_values;
    matrix m_eigen_vectors;

    bool m_is_fitted;
};