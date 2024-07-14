#pragma once

#include "solver_engine.h"
#include "math_utils.h"

namespace gcransac::estimator::solver
{

class RectifyingHomographyTwoSIFTSolver : public SolverEngine<SIFTRectifyingHomography>
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
        std::vector<SIFTRectifyingHomography> &models_,
        const double *weights_ = nullptr
    ) const;

    static Eigen::Vector2d residual(
        const cv::Mat& feature,
        const SIFTRectifyingHomography& model
    );

    bool normalizePoints(
        const cv::Mat& data,
        const size_t* sample,
        const size_t& sample_number,
        cv::Mat& normalized_features,
        NormalizingTransform& normalizing_transform
    ) const;

protected:
    static constexpr double kScalePower = -1.0 / 3.0;
    static constexpr double kEpsilon = 1e-9;

    bool estimateNonMinimalModel(
        const cv::Mat &data_,
        const size_t *sample_,
        size_t sample_number_,
        std::vector<SIFTRectifyingHomography> &models_,
        const double *weights_
    ) const;

    bool estimateMinimalModel(
        const cv::Mat &data_,
        const size_t *sample_,
        size_t sample_number_,
        std::vector<SIFTRectifyingHomography> &models_
    ) const;
};

void lineFromSIFT(double x, double y, double theta, Eigen::Vector3d& line)
{
    const auto c = std::cos(theta);
    const auto s = std::sin(theta);
    line(0) = s;
    line(1) = -c;
    line(2) = y * c - x * s;
}

bool orthogonalVanishingPoint(
    const Eigen::Vector3d& vp,
    const SIFTRectifyingHomography& model,
    Eigen::Vector3d& result
)
{
    Eigen::Vector3d vp_rect = vp;
    model.rectify(vp_rect);
    if (std::abs(vp_rect(2)) > 1e-6)
    {
        fprintf(
            stderr,
            "Rectified vanishing point should be at infinity (homogeneous coordinate should be zero, but instead its value is %f).\n",
            vp_rect(2)
        );
        return false;
    }
    result(0) = vp_rect(1); // x <-- y
    result(1) = -vp_rect(0); // y <-- -x
    result(2) = 0.0;
    model.unrectify(result);
    return true;
}

bool RectifyingHomographyTwoSIFTSolver::estimateMinimalModel(
    const cv::Mat &data_,
    const size_t *sample_,
    size_t sample_number_,
    std::vector<SIFTRectifyingHomography> &models_
) const
{
    // helper function to fetch correct sample
    auto get_sample_ptr = [sample_, &data_](size_t i) {
        const auto *data_ptr = reinterpret_cast<double*>(data_.data);
        const size_t idx = (sample_ == nullptr) ? i : sample_[i];
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
    // construct model
    SIFTRectifyingHomography model;
    model.h7 = x(0);
    model.h8 = x(1);
    model.alpha = x(2);
    if (std::abs(model.alpha) < kEpsilon)
    {
        fprintf(stderr, "Invalid solution for the minimal case: alpha is zero\n");
        return false;
    }
    model.vp1 = vp;
    if (!orthogonalVanishingPoint(model.vp1, model, model.vp2))
    {
        return false;
    }
    models_.emplace_back(model);
    return true;
}

bool RectifyingHomographyTwoSIFTSolver::estimateNonMinimalModel(
    const cv::Mat& data_,
    const size_t* sample_,
    size_t sample_number_,
    std::vector<SIFTRectifyingHomography>& models_,
    const double* weights_
) const
{
    // helper function to fetch correct sample
    auto get_sample_ptr = [sample_, &data_](size_t i) {
        const auto *data_ptr = reinterpret_cast<double*>(data_.data);
        const size_t idx = (sample_ == nullptr) ? i : sample_[i];
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
        const auto xi = sample_i[0]; // i-th x-coordinate
        const auto yi = sample_i[1]; // i-th y-coordinate
        const auto ti = sample_i[2]; // i-th orientation
        lineFromSIFT(xi, yi, ti, li);

        for (size_t j = i + 1; j < sample_number_; ++j)
        {
            const auto* sample_j = get_sample_ptr(j); // first sample
            const auto xj = sample_j[0]; // j-th x-coordinate
            const auto yj = sample_j[1]; // j-th y-coordinate
            const auto tj = sample_j[2]; // j-th orientation

            lineFromSIFT(xj, yj, tj, lj);
            auto vp = li.cross(lj); // intersection of lines
            // rescale homogeneous vanishing point to scale coefficients to
            // range [-1, 1]
            const auto max_abs_value = vp.cwiseAbs().maxCoeff();
            if (max_abs_value > 1.0)
            {
                vp /= max_abs_value;
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
        fprintf(stderr, "Invalid solution for the non-minimal model\n");
        return false;
    }
    // construct model
    SIFTRectifyingHomography model;
    model.h7 = x(0);
    model.h8 = x(1);
    model.alpha = x(2);
    if (std::abs(model.alpha) < kEpsilon)
    {
        fprintf(stderr, "Invalid solution for the non-minimal case: alpha is zero\n");
        // std::cout << "\nNon-minimal solution with zero alpha:\n"
        //           << "Number of samples: " << sample_number_ << "\n"
        //           << "coefficient matrix:\n" << coeffs << "\n"
        //           << "RHS: " << rhs.transpose() << "\n"
        //           << "x: " << x.transpose() << "\n";
        return false;
    }
    // TODO update vanishing points in model
    model.vp1 << 0, 0, 0;
    model.vp2 << 0, 0, 0;
    models_.emplace_back(model);
    return true;
}

bool RectifyingHomographyTwoSIFTSolver::estimateModel(
    const cv::Mat& data_,
    const size_t* sample_,
    size_t sample_number_,
    std::vector<SIFTRectifyingHomography> &models_,
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

Eigen::Vector2d RectifyingHomographyTwoSIFTSolver::residual(
    const cv::Mat& feature,
    const SIFTRectifyingHomography& model
)
{
    Eigen::Vector2d residuals;
    // Normalized model: parameters are in the normalized image coordinate system
    const auto alpha = model.alpha;
    const auto& vp1 = model.vp1;
    const auto& vp2 = model.vp2;
    // Unnormalized feature: parameters are in the unnormalized input image coordinate system
    const auto* feature_ptr = reinterpret_cast<double*>(feature.data);
    const double orientation = feature_ptr[2];
    double s = feature_ptr[3];
    // the scale change which fits the model
    Eigen::Vector3d point(feature_ptr[0], feature_ptr[1], 1.0);
    model.normalize(point);
    s *= model.s;
    // line induced by coordinates and orientation
    Eigen::Vector3d line;
    lineFromSIFT(point(0), point(1), orientation, line); // compute before applying the homography to the point
    model.rectify(point);
    // point(2) = h7 * x' + h8 * y' + 1, where (x', y') are normalized coordinates
    if (std::abs(point(2)) < kEpsilon)
    {
        // an estimated rectifying homography which sends a detected feature 
        // to infinity must be wrong
        residuals(0) = DBL_MAX;
    }
    const auto model_s = std::pow(alpha / point(2), 3.0);
    // scale-based residual: the deviation between the input scale and the model scale
    if (std::abs(model_s) < kEpsilon)
    {
        // an estimated rectifying homography should not allow a point with
        // zero scale change
        residuals(0) = DBL_MAX;
    }
    residuals(0) = std::abs(1.0 - (s / model_s));
    // orientation-based residual: the minimal Euclidean distance between the
    // line and the two vanishing points
    const auto d1 = std::abs(line.dot(vp1));
    const auto d2 = std::abs(line.dot(vp2));
    residuals(1) = std::min(d1, d2);
    
    return residuals;
}

bool RectifyingHomographyTwoSIFTSolver::normalizePoints(
    const cv::Mat& data, // The data points
    const size_t* sample, // The points to which the model will be fit
    const size_t& sample_number,// The number of points
    cv::Mat& normalized_features, // The normalized features
    NormalizingTransform& normalizing_transform // the normalization transformation model
) const
{
    constexpr size_t x_pos = 0;
    constexpr size_t y_pos = 1;
    constexpr size_t t_pos = 2;
    constexpr size_t s_pos = 3;
    if (sample_number < 1)
    {
        fprintf(stderr,
            "Feature normalization failed because number of input features is zero.\n"
        );
        return false;
    }
    const auto cols = data.cols;
    const double* data_ptr = reinterpret_cast<double*>(data.data);
    // compute mean position of features
    normalizing_transform.x0 = 0.0;
    normalizing_transform.y0 = 0.0;
    for (size_t i = 0; i < sample_number; i++)
    {
        const double* feature = data_ptr + cols * sample[i];
        normalizing_transform.x0 += feature[x_pos]; // x-coordinate
        normalizing_transform.y0 += feature[y_pos]; // y-coordinate
    }
    const auto inv_n = 1.0 / static_cast<double>(sample_number);
    normalizing_transform.x0 *= inv_n;
    normalizing_transform.y0 *= inv_n;
    // compute average Euclidean distance to mean position
    double avg_dist = 0.0;
    for (size_t i = 0; i < sample_number; i++)
    {
        const double* feature = data_ptr + cols * sample[i];
        const auto dx = feature[x_pos] - normalizing_transform.x0; // x-coordinate
        const auto dy = feature[y_pos] - normalizing_transform.y0; // y-coordinate
        avg_dist += sqrt(dx * dx + dy * dy);
    }
    avg_dist *= inv_n;
    if (avg_dist < kEpsilon)
    {
        fprintf(stderr,
            "Feature normalization failed because all features are located in the \
            same position (near-zero average distance from mean position), or because \
            the average distance came out negative. Average distance: %f\n",
            avg_dist
        );
        return false;
    }
    // compute scaling factor to transform all feature positions to so that averge
    // distance is sqrt(2).
    normalizing_transform.s = M_SQRT2 / avg_dist;
    // compute normalized features - normalizing is relevant only for coordinates
    // and scale as the scaling of feature positions about the origin is isotropic
    auto* norm_features_ptr = reinterpret_cast<double*>(normalized_features.data);
    for (size_t i = 0; i < sample_number; i++)
    {
        const double* feature = data_ptr + cols * sample[i];
        auto norm_x = feature[x_pos]; // x-coordinate
        auto norm_y = feature[y_pos]; // y-coordinate
        auto norm_scale = feature[s_pos]; // scale

        normalizing_transform.normalize(norm_x, norm_y);
        normalizing_transform.normalizeScale(norm_scale);
        
        norm_features_ptr[i * normalized_features.cols + x_pos] = norm_x;
        norm_features_ptr[i * normalized_features.cols + y_pos] = norm_y;
        // orientation is not affected by translation and isotropic scaling
        norm_features_ptr[i * normalized_features.cols + t_pos] = feature[t_pos];
        norm_features_ptr[i * normalized_features.cols + s_pos] = norm_scale;
    }

    return true;
}

}
