#pragma once

#include "solver_engine.h"
#include "math_utils.h"

namespace gcransac::estimator::solver
{

class RectifyingHomographyTwoSIFTSolver : public SolverEngine
{

public:
    RectifyingHomographyTwoSIFTSolver() {}
    ~RectifyingHomographyTwoSIFTSolver() {}

    /// @brief Determines if there is a possibility for the method
    /// `estimateModel` to return multiple models.
    /// @return True if there may be multiple models. False, otherwise.
    static OLGA_INLINE constexpr bool returnMultipleModels()
    {
        return maximumSolutions() > 1;
    }

    /// @brief 
    /// @return The maximum number of models returned by the estimator. 
    static OLGA_INLINE constexpr size_t maximumSolutions() { return 1; }

    /// @brief
    /// @return The minimum number of samples required for the estimation.
    static OLGA_INLINE constexpr size_t sampleSize() { return 2; }

    /// @brief
    /// @return True if the solver requires the direction of gravity.
    /// False, otherwise. 
    static OLGA_INLINE constexpr bool needsGravity() { return false; }

    /// @brief Estimates the models parameters from the given sample set.
    /// @param data_  the set of samples.
    /// @param sample_ the indices of the samples used for the estimation.
    /// @param sample_number_ the size of the sample set used for the estimation.
    /// @param models_ the estimated parameters of the model(s).
    /// @param weights_ the weights corresponding to each sample. Uniform 
    /// weights are used by default if the input is a null pointer.
    /// @return True if the model parameters estimation was successful.
    /// False, otherwise.
    bool estimateModel(
        const cv::Mat& data_,
        const size_t *sample_,
        size_t sample_number_,
        std::vector<Model> &models_,
        const double *weights_ = nullptr
    ) const;

    static double residual(const cv::Mat& feature, const Eigen::MatrixXd& descriptor);

protected:
    bool estimateNonMinimalModel(
        const cv::Mat &data_,
        const size_t *sample_,
        size_t sample_number_,
        std::vector<Model> &models_,
        const double *weights_
    ) const;

    bool estimateMinimalModel(
        const cv::Mat &data_,
        const size_t *sample_,
        size_t sample_number_,
        std::vector<Model> &models_
    ) const;
};

void lineFromSIFT(double x, double y, double theta, Eigen::Vector3d& line)
{
    const auto c = std::cos(theta);
    const auto s = std::sin(theta);
    line = Eigen::Vector3d(s, -c, y * c - x * s);
}

void makeHomography(double h7, double h8, Eigen::Matrix3d& H)
{
    H = Eigen::Matrix3d::Identity();
    H(2, 0) = h7;
    H(2, 1) = h8;
}

void orthogonalVanishingPoint(const Eigen::Vector3d& vp, const Eigen::Matrix3d& H, Eigen::Vector3d& result)
{
    const Eigen::Vector3d z_hat(0.0, 0.0, 1.0);
    result = H.inverse() * ((H * vp).cross(z_hat));
}

bool RectifyingHomographyTwoSIFTSolver::estimateMinimalModel(
    const cv::Mat &data_,
    const size_t *sample_,
    size_t sample_number_,
    std::vector<Model> &models_
) const
{
    constexpr double kScalePower = -1.0 / 3.0;
    constexpr double Eps = 1e-9;
    // helper function to fetch correct sample
    auto get_sample_ptr = [sample_, &data_](size_t i) {
        const auto *data_ptr = reinterpret_cast<double*>(data_.data);
        const size_t idx = sample_ == nullptr ? i : sample_[i];
        return data_ptr + idx * data_.cols;
    };

    if (sample_number_ != sampleSize())
    {
        fprintf(
            stderr,
            "Minimal model requires exactly %d samples (received %d).\n",
            sampleSize(),
            sample_number_
        );
        return false;
    }    

    Eigen::Matrix<double, 3, 4> coeffs;

    const auto* sample1 = get_sample_ptr(0); // first sample
    const auto x1 = sample1[0]; // first x-coordinate
    const auto y1 = sample1[1]; // first y-coordinate
    const auto t1 = sample1[2]; // first orientation
    const auto s1 = sample1[3]; // first scale

    const auto* sample2 = get_sample_ptr(1); // second sample
    const auto x2 = sample2[0]; // second x-coordinate
    const auto y2 = sample2[1]; // second y-coordinate
    const auto t2 = sample2[2]; // second orientation
    const auto s2 = sample2[3]; // second scale

    // first line in coefficient matrix is constructed from the first sample only
    coeffs(0, 0) = x1;
    coeffs(0, 1) = y1;
    coeffs(0, 2) = -pow(s1, kScalePower);
    coeffs(0, 3) = -1.0;

    // second line in coefficient matrix is constructed from the second sample only
    coeffs(1, 0) = x2;
    coeffs(1, 1) = y2;
    coeffs(1, 2) = -pow(s2, kScalePower);
    coeffs(1, 3) = -1.0;

    // third line in the coefficient matrix is constructed from both samples
    Eigen::Vector3d l1;
    lineFromSIFT(x1, y1, t1, l1);
    Eigen::Vector3d l2;
    lineFromSIFT(x2, y2, t2, l2);
    auto vp = l1.cross(l2); // intersection of lines
    if (std::abs(vp(2)) > Eps)
    {
        vp /= vp(2);
    }

    coeffs(2, 0) = vp(0);
    coeffs(2, 1) = vp(1);
    coeffs(2, 2) = 0;
    coeffs(2, 3) = -vp(2);

    Eigen::Matrix<double, 3, 1> x;
    gcransac::utils::gaussElimination<3>(coeffs, x);
    if (x.hasNaN())
    {
        return false;
    }
    // if (std::abs(x(2)) > Eps)
    // {
    //     x /= x(2);
    // }
    
    // compute vanishing point orthogonal to the one used to estimate the model
    Eigen::Matrix3d H;
    makeHomography(x(0), x(1), H);
    Eigen::Vector3d vp_perp;
    orthogonalVanishingPoint(vp, H, vp_perp);
    // construct model
    RectifyingHomography model;
    model.descriptor << x(0), x(1), x(2),
                        vp(0), vp(1), vp(2),
                        vp_perp(0), vp_perp(1), vp_perp(2);
    models_.emplace_back(model);
    return true;
}

bool RectifyingHomographyTwoSIFTSolver::estimateNonMinimalModel(
    const cv::Mat &data_,
    const size_t *sample_,
    size_t sample_number_,
    std::vector<Model> &models_,
    const double *weights_
) const
{
    constexpr double kScalePower = -1.0 / 3.0;
    constexpr double Eps = 1e-9;
    // helper function to fetch correct sample
    auto get_sample_ptr = [sample_, &data_](size_t i) {
        const auto *data_ptr = reinterpret_cast<double*>(data_.data);
        const size_t idx = sample_ == nullptr ? i : sample_[i];
        return data_ptr + idx * data_.cols;
    };

    const size_t n_rows = (sample_number_ * (sample_number_ + 1)) / 2;
    Eigen::MatrixXd coeffs(n_rows, 3);
    Eigen::MatrixXd rhs(n_rows, 1);
    // populate first sample_number_ rows of coeffs and rhs matrices with the
    // constraints derived from positions and scales
    size_t curr_idx = 0;
    for (size_t i = 0; i < sample_number_; ++i)
    {
        const auto* sample = get_sample_ptr(i); // first sample
        const auto x = sample[0]; // first x-coordinate
        const auto y = sample[1]; // first y-coordinate
        const auto s = sample[3]; // first scale

        coeffs(curr_idx, 0) = x;
        coeffs(curr_idx, 1) = y;
        coeffs(curr_idx, 2) = -pow(s, kScalePower);
        rhs(curr_idx) = -1.0;

        curr_idx++;
    }

    // populate last "sample_number_ choose 2" rows of coeffs and rhs matrices
    // with the constraints derived from positions and orientations
    Eigen::Vector3d li;
    Eigen::Vector3d lj;
    for (size_t i = 0; i < sample_number_ - 1; ++i)
    {
        const auto* sample_i = get_sample_ptr(i); // first sample
        const auto xi = sample_i[0]; // first x-coordinate
        const auto yi = sample_i[1]; // first y-coordinate
        const auto ti = sample_i[2];
        const auto si = sample_i[3]; // first scale
        lineFromSIFT(xi, yi, ti, li);

        for (size_t j = i + 1; j < sample_number_; ++j)
        {
            const auto* sample_j = get_sample_ptr(j); // first sample
            const auto xj = sample_j[0]; // first x-coordinate
            const auto yj = sample_j[1]; // first y-coordinate
            const auto tj = sample_j[2];
            const auto sj = sample_j[3]; // first scale

            lineFromSIFT(xj, yj, tj, lj);
            auto vp = li.cross(lj); // intersection of lines
            if (std::abs(vp(2)) > Eps)
            {
                vp /= vp(2);
            }

            coeffs(curr_idx, 0) = vp(0);
            coeffs(curr_idx, 1) = vp(1);
            coeffs(curr_idx, 2) = 0.0;
            rhs(curr_idx) = -vp(2);

            curr_idx++;
        }
    }
    // solve linear least squares system
    Eigen::Matrix<double, 3, 1> x = coeffs.colPivHouseholderQr().solve(rhs);
    // verify validity of solution
    if (x.hasNaN())
    {
        fprintf(stderr, "Invalid solution for the non-minimal model");
        return false;
    }
    // if (std::abs(x(2)) > Eps)
    // {
    //     x /= x(2);
    // }

    // compute vanishing point orthogonal to the one used to estimate the model
    Eigen::Matrix3d H;
    makeHomography(x(0), x(1), H);
    // TODO compute a vanishing point and its orthogonal counterpart in the non-minimal solution.
    Eigen::Vector3d vp_perp;
    orthogonalVanishingPoint(vp, H, vp_perp);
    // construct model
    RectifyingHomography model;
    model.descriptor << x(0), x(1), x(2),
                        vp(0), vp(1), vp(2),
                        vp_perp(0), vp_perp(1), vp_perp(2);

    RectifyingHomography model;
    model.descriptor << x(0), x(1), x(2);
    models_.emplace_back(model);
    return true;
}

bool RectifyingHomographyTwoSIFTSolver::estimateModel(
    const cv::Mat& data_,
    const size_t *sample_,
    size_t sample_number_,
    std::vector<Model> &models_,
    const double *weights_
) const
{
    if (sample_number_ < sampleSize())
    {
        fprintf(stderr,
            "There weren't enough SIFT features provided for the solver (%d < %d).\n",
            sample_number_,
            sampleSize()
        );
        return false;
    }
    if (sample_number_ == sampleSize())
    {
        return estimateMinimalModel(data_, sample_, sample_number_, models_);
    }
    return estimateNonMinimalModel(data_, sample_, sample_number_, models_, weights_);
}

double RectifyingHomographyTwoSIFTSolver::residual(
    const cv::Mat& feature,
    const Eigen::MatrixXd& descriptor
)
{
    if (descriptor.rows() != 1 || descriptor.cols() != 9)
    {
        std::stringstream error_msg;
        error_msg << "Expected a 1x9 matrix as a descriptor. " 
                  << "Received the following " << descriptor.rows() << "x"
                  << descriptor.cols() << " matrix instead:\n" << descriptor 
                  << "\n";
        throw std::runtime_error(error_msg.str());
    }
    const auto h7 = descriptor(0);
    const auto h8 = descriptor(1);
    const auto alpha = descriptor(2);
    const Eigen::RowVector3d vp1 = descriptor.block<1, 3>(0, 3);
    const Eigen::RowVector3d vp2 = descriptor.block<1, 3>(0, 6);

    const auto* feature_ptr = reinterpret_cast<double*>(feature.data);
    const auto& x = feature_ptr[0];
    const auto& y = feature_ptr[1];
    const auto& t = feature_ptr[2];
    const auto& s = feature_ptr[3];
    // the scale change which fits the model
    const auto model_s = pow(alpha / (h7 * x + h8 * y + 1), 3.0);
    // scale-based residual: the deviation between the input scale and the model scale
    const auto r_scale = 1.0 - (s / model_s);
    // line induced by coordinates and orientation
    Eigen::Vector3d line;
    lineFromSIFT(x, y, t, line);
    // orientation-based residual: the minimal Euclidean distance between the
    // line and the two vanishing points
    const auto d1 = line.dot(vp1);
    const auto d2 = line.dot(vp2);
    const auto r_orientation = (std::abs(d1) < std::abs(d2)) ? d1 : d2; 
    // TODO currently it is only possible to return a scalar residual, so we
    // are forced to return a mix of the two residuals, which loses the geometric
    // meaning of each one.
    return r_scale * r_orientation;
}

}
