#pragma once

#include <vector>
#include <optional>
#include <cmath>
#include <unordered_map>
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

    void getInlierWeights(
        const size_t* sample,
        const size_t& sample_number,
        const double* weights,
        std::vector<double>& inlier_weights
    ) const;

protected:
    static constexpr double kScalePower = -1.0 / 3.0;
    static constexpr double kEpsilon = 1e-9;
    static constexpr double kCircularStdDevThresh = M_PI / 18.0; // 10 degrees
    static constexpr double kCircularVarThresh = kCircularStdDevThresh * kCircularStdDevThresh;
    static constexpr size_t x_pos = 0; // x-coordinate position
    static constexpr size_t y_pos = 1; // y-coordinate position
    static constexpr size_t t_pos = 2; // orientation position
    static constexpr size_t s_pos = 3; // scale position
    static constexpr size_t weight_dim = 2;
    static constexpr size_t scale_weight_pos = 0;
    static constexpr size_t orientation_weight_pos = 1;

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

Eigen::Vector3d lineFromSIFT(double x, double y, double theta)
{
    const auto c = std::cos(theta);
    const auto s = std::sin(theta);
    return {s, -c, y * c - x * s};
}

/// @brief Computes the minimal angular difference between two angles while
/// treating each angle and its opposite angle as equivalent.
/// @param angle1 first angle as angle in range [0, 2 * PI)
/// @param angle2 second angle as angle in range [0, 2 * PI)
/// @return The minimal angular difference between the two angles.
double absoluteAngleDiff(const double& angle1, const double& angle2)
{
    constexpr auto kTwoPI = 2.0 * M_PI;
    auto diff1 = std::fabs(angle1 - angle2);
    auto diff2 = std::fabs(angle1 - angle2 - M_PI); // flipping second orientation
    return std::fmin(
        std::fmin(diff1, kTwoPI - diff1),
        std::fmin(diff2, kTwoPI - diff2)
    );
}

bool RectifyingHomographyTwoSIFTSolver::estimateMinimalModel(
    const cv::Mat &data_,
    const size_t *sample_,
    size_t sample_number_,
    std::vector<SIFTRectifyingHomography> &models_
) const
{
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
    // helper function to fetch correct sample
    auto get_sample_ptr = [sample_, &data_](const size_t& i) {
        const auto *data_ptr = reinterpret_cast<double*>(data_.data);
        const size_t idx = (sample_ == nullptr) ? i : sample_[i];
        return data_ptr + idx * data_.cols;
    };

    Eigen::Matrix<double, 3, 4> coeffs;

    const auto* sample1 = get_sample_ptr(0); // first sample
    const auto x1 = sample1[x_pos]; // first x-coordinate
    const auto y1 = sample1[y_pos]; // first y-coordinate
    const auto t1 = sample1[t_pos]; // first orientation
    const auto s1 = sample1[s_pos]; // first scale

    const auto* sample2 = get_sample_ptr(1); // second sample
    const auto x2 = sample2[x_pos]; // second x-coordinate
    const auto y2 = sample2[y_pos]; // second y-coordinate
    const auto t2 = sample2[t_pos]; // second orientation
    const auto s2 = sample2[s_pos]; // second scale

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
    const auto l1 = lineFromSIFT(x1, y1, t1);
    const auto l2 = lineFromSIFT(x2, y2, t2);
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
    if (model.alpha < kEpsilon)
    {
        return false;
    }
    const auto rectified_t1 = fmod(model.rectifiedAngle(x1, y1, t1), M_PI);
    const auto rectified_t2 = fmod(model.rectifiedAngle(x2, y2, t2), M_PI);
    if (absoluteAngleDiff(rectified_t1, rectified_t2) > M_PI / 180.0)
    {
        fprintf(
            stderr, 
            "Invalid solution for the minimal case: rectified angles are not "
            "parallel (angle #1: %f, angle #2: %f).\n",
            rectified_t1, rectified_t2
        );
        return false;
    }
    model.vanishing_point_dir1 = 0.5 * (rectified_t1 + rectified_t2);
    // the second vanishing point's direction is orthogonal to the first.
    model.vanishing_point_dir2 = fmod(model.vanishing_point_dir1 + M_PI_2, M_PI);
    models_.emplace_back(model);
    return true;
} 

/// @brief Estimates the weighted-mode (most probable value) of the distribution
/// from which the angle samples where taken. This is done by placing the samples
/// in bins and finding the most frequent one. This functions treat an angle
/// theta and theta + PI as the same orientation.
/// @param angles a vector of angles in radians in range [0, 2 * PI).
/// @param weights a vector of weights in range [0, 1].
/// @param bin_width the width of the bins (must be a positive number).
/// @return a scalar represeting the estimated weighted-mode of the sampled distribution.
double findWeightedMode(
    const std::vector<double>& angles,
    const std::vector<double>& weights,
    double bin_width
)
{
    if (angles.size() != weights.size() || angles.empty()) {
        throw std::invalid_argument("Angles and weights must be of the same size and non-empty.");
    }

    if (bin_width <= 0) {
        throw std::invalid_argument("Bin width must be a positive value.");
    }

    // Calculate the total weight for each binned sample value
    std::unordered_map<int, double> weight_map;
    std::unordered_map<int, double> bin_value_map;
    for (size_t i = 0; i < angles.size(); i++)
    {
        const auto angle = fmod(angles[i], M_PI);
        const int bin = static_cast<int>(std::round(angle / bin_width));
        weight_map[bin] += weights[i];
        bin_value_map[bin] += angles[i] * weights[i];
    }

    // Find the binned sample with the maximum total weight
    int mode_bin = 0;
    double max_weight = -1;
    for (const auto& pair : weight_map) {
        if (pair.second > max_weight) {
            max_weight = pair.second;
            mode_bin = pair.first;
        }
    }

    // Calculate the weighted average within the mode bin
    double mode = bin_value_map[mode_bin] / weight_map[mode_bin];

    return mode;
}

constexpr inline size_t nChoose2(const size_t& n)
{
    return (n * (n - 1)) / 2;
}

bool RectifyingHomographyTwoSIFTSolver::estimateNonMinimalModel(
    const cv::Mat& data_,
    const size_t* sample_,
    size_t sample_number_,
    std::vector<SIFTRectifyingHomography>& models_,
    const double* weights_
) const
{
    using namespace std;
    constexpr auto kBinWidth = M_PI / 360.0; // half-degree in radians
    // helper functions to fetch correct sample and weight 
    auto get_sample_ptr = [sample_, &data_](const size_t& i) {
        const auto *data_ptr = reinterpret_cast<double*>(data_.data);
        const size_t idx = (sample_ == nullptr) ? i : sample_[i];
        return data_ptr + idx * data_.cols;
    };
    auto get_weight = [sample_, weights_](const size_t& i, const size_t& j) {
        const size_t idx = (sample_ == nullptr) ? i : sample_[i];
        return (weights_ == nullptr) ? 1.0 : weights_[weight_dim * idx + j];
    };
    // count non-zero weights to determine how many rows the coefficient matrix should have.
    std::vector<size_t> scale_inliers;
    std::vector<size_t> orientation_inliers;
    scale_inliers.reserve(sample_number_);
    orientation_inliers.reserve(sample_number_);
    if (weights_ == nullptr)
    {
        // 1-valued weights for scale- and orientation-constraints are assumed,
        // meaning maximum number of constraints is expected.
        // populate vectors so that at index i the value i will appear.
        for (size_t i = 0; i < sample_number_; i++)
        {
            scale_inliers.push_back(i);
            orientation_inliers.push_back(i);
        }
        // verify total number of rows is the maximum possible with sample_numbers_ samples.
        const auto n_max_rows = (sample_number_ * (sample_number_ + 1)) / 2;
        if (n_max_rows != (scale_inliers.size() + nChoose2(orientation_inliers.size())))
        {
            fprintf(
                stderr,
                "Received incorrect number of rows for the uniform-weights case in the non-minimal solver.\n"
            );
        }
    }
    else
    {
        // count number of non-zero weights for each type of weight
        for (size_t i = 0; i < sample_number_; ++i)
        {
            if (get_weight(i, scale_weight_pos) > 0.0)
            {
                scale_inliers.push_back(i);
            }
            if (get_weight(i, orientation_weight_pos) > 0.0)
            {
                orientation_inliers.push_back(i);
            }
        }
        scale_inliers.shrink_to_fit();
        orientation_inliers.shrink_to_fit();
    }
    const auto n_scale_constraints = scale_inliers.size();
    const auto n_orientation_constraints = nChoose2(orientation_inliers.size());
    // make sure there are enough constraints from each type to estimate the model.
    if (n_scale_constraints < 2 || n_orientation_constraints < 1)
    {
        fprintf(
            stderr,
            "Insufficient combination of scale- and orientation-based "
            "constraints to estimate the non-minimal model.\n"
            "There are %d scale-based constraints and %d "
            "orientation-based constraints.\n",
            n_scale_constraints, n_orientation_constraints
        );
        return false;
    }
    // the number of rows in the coefficient matrix is the total number of constraints.
    const auto n_rows = n_scale_constraints + n_orientation_constraints;

    Eigen::MatrixXd coeffs(n_rows, 3);
    Eigen::VectorXd rhs(n_rows, 1);
    // populate first sample_number_ rows of coeffs and rhs matrices with the
    // constraints derived from positions and scales
    size_t curr_idx = 0;
    for (const auto& i : scale_inliers)
    {
        const auto* sample = get_sample_ptr(i);
        const auto w = get_weight(i, scale_weight_pos);
        const auto x = sample[x_pos]; // x-coordinate
        const auto y = sample[y_pos]; // y-coordinate
        const auto s = sample[s_pos]; // scale

        coeffs(curr_idx, 0) = w * x;
        coeffs(curr_idx, 1) = w * y;
        coeffs(curr_idx, 2) = -w * pow(s, kScalePower);
        rhs(curr_idx) = -w;

        curr_idx++;
    }
    // populate last "sample_number_ choose 2" rows of coeffs and rhs matrices
    // with the constraints derived from positions and orientations
    const auto n_orientation_inliers = orientation_inliers.size();
    vector<optional<Eigen::Vector3d>> lines{n_orientation_inliers, nullopt};
    for (size_t i = 0; i < n_orientation_inliers - 1; ++i)
    {
        if (!lines.at(i).has_value())
        {
            const auto* sample_i = get_sample_ptr(orientation_inliers.at(i)); // i-th sample
            const auto xi = sample_i[x_pos]; // i-th x-coordinate
            const auto yi = sample_i[y_pos]; // i-th y-coordinate
            const auto ti = sample_i[t_pos]; // i-th orientation
            lines.at(i).emplace(lineFromSIFT(xi, yi, ti));
        }

        for (size_t j = i + 1; j < n_orientation_inliers; ++j)
        {
            if (!lines.at(j).has_value())
            {
                const auto* sample_j = get_sample_ptr(orientation_inliers.at(j)); // j-th sample
                const auto xj = sample_j[x_pos]; // j-th x-coordinate
                const auto yj = sample_j[y_pos]; // j-th y-coordinate
                const auto tj = sample_j[t_pos]; // j-th orientation
                lines.at(j).emplace(lineFromSIFT(xj, yj, tj));
            }

            const auto& li = lines.at(i).value();
            const auto& lj = lines.at(j).value();
            auto vp = li.cross(lj); // intersection of lines
            // rescale homogeneous vanishing point to scale coefficients to
            // range [-1, 1]
            const auto max_abs_value = vp.cwiseAbs().maxCoeff();
            if (max_abs_value > 1.0)
            {
                vp /= max_abs_value;
            }
            const auto wi = get_weight(orientation_inliers.at(i), orientation_weight_pos);
            const auto wj = get_weight(orientation_inliers.at(j), orientation_weight_pos);
            const auto w = wi * wj;

            coeffs(curr_idx, 0) = w * vp(0);
            coeffs(curr_idx, 1) = w * vp(1);
            coeffs(curr_idx, 2) = 0.0;
            rhs(curr_idx) = -w * vp(2);

            curr_idx++;
        }
    }
    // verify coefficient matrix was constructed as expected.
    if (curr_idx != n_rows)
    {
        fprintf(
            stderr,
            "Error while computing coefficient matrix in the non-minimal solver:\n"
            "The number of constraints added to the matrix (%d) is different from "
            "the number of rows of the matrix (%d).\n",
            curr_idx, n_rows
        );
        return false;
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
    if (model.alpha < kEpsilon)
    {
        return false;
    }
    std::vector<double> rectified_angles(n_orientation_inliers, 0.0);
    std::vector<double> angle_weights(n_orientation_inliers, 0.0);
    for (size_t i = 0; i < n_orientation_inliers; ++i)
    {
        const auto* sample = get_sample_ptr(orientation_inliers.at(i));
        const auto x = sample[x_pos]; // x-coordinate
        const auto y = sample[y_pos]; // y-coordinate
        const auto t = sample[t_pos]; // orientation
        rectified_angles.at(i) = model.rectifiedAngle(x, y, t);
        angle_weights.at(i) = get_weight(orientation_inliers.at(i), orientation_weight_pos);
    }
    model.vanishing_point_dir1 = findWeightedMode(
        rectified_angles, angle_weights, kBinWidth
    );
    // the second vanishing point's direction is orthogonal to the first.
    model.vanishing_point_dir2 = fmod(model.vanishing_point_dir1 + M_PI_2, M_PI);
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
    const auto* feature_ptr = reinterpret_cast<double*>(feature.data);
    Eigen::Vector3d point(feature_ptr[x_pos], feature_ptr[y_pos], 1.0);
    double scale = feature_ptr[s_pos];
    // Normalize coordinates and scale (orientation is unchanged by normalization).
    model.normalize(point);
    model.normalizeScale(scale);
    // Rectify orientation and scale.
    const auto rectified_orientation = model.rectifiedAngle(
        point(0), point(1), feature_ptr[t_pos]
    );
    const auto rectified_scale = model.rectifiedScale(
        point(0), point(1), scale
    );
    // the model's estimation of the feature's cubed-scale in the rectified image
    const auto alpha_cube = std::pow(model.alpha, 3.0);
    // scale-based residual: logarithmic scale difference between the feature's
    // rectified scale and the model's estimated rectified scale for all features.
    const auto r_scale = std::fabs(std::log(rectified_scale / alpha_cube));
    // orientation-based residual: the minimal angular distance between the
    // feature's rectified orientation and the model's two orthogonal principle
    // orientations.  
    const auto r_orientation = std::fmin(
        absoluteAngleDiff(model.vanishing_point_dir1, rectified_orientation),
        absoluteAngleDiff(model.vanishing_point_dir2, rectified_orientation)
    );

    return {r_scale, r_orientation};
}

bool RectifyingHomographyTwoSIFTSolver::normalizePoints(
    const cv::Mat& data, // The data points
    const size_t* sample, // The points to which the model will be fit
    const size_t& sample_number,// The number of points
    cv::Mat& normalized_features, // The normalized features
    NormalizingTransform& normalizing_transform // the normalization transformation model
) const
{
    if (sample_number < 1)
    {
        fprintf(stderr,
            "Feature normalization failed because number of input features is zero.\n"
        );
        return false;
    }
    // helper function to fetch correct sample
    auto get_sample_ptr = [sample, &data](size_t i) {
        const auto *data_ptr = reinterpret_cast<double*>(data.data);
        const size_t idx = (sample == nullptr) ? i : sample[i];
        return data_ptr + idx * data.cols;
    };
    // compute mean position of features
    normalizing_transform.x0 = 0.0;
    normalizing_transform.y0 = 0.0;
    for (size_t i = 0; i < sample_number; i++)
    {
        const auto* feature = get_sample_ptr(i);
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
        const auto* feature = get_sample_ptr(i);
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
        const auto* feature = get_sample_ptr(i);
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
        // ensures that if the dimension of the features is larger
        // than 4, then the normalization will still succeed.
        for (size_t j = 4; j < normalized_features.cols; j++)
        {
			norm_features_ptr[i * normalized_features.cols + j] = feature[j];
        }
    }

    return true;
}

void RectifyingHomographyTwoSIFTSolver::getInlierWeights(
    const size_t* sample,
    const size_t& sample_number,
    const double* weights,
    std::vector<double>& inlier_weights
) const
{
    auto get_weight = [sample, weights](const size_t& i, const size_t& j) {
        const size_t idx = (sample == nullptr) ? i : sample[i];
        return (weights == nullptr) ? 1.0 : weights[weight_dim * idx + j]; // weights are in row-major order
    };
    inlier_weights.clear();
    inlier_weights.reserve(sample_number);
    for (size_t i = 0; i < sample_number; i++)
    {
        inlier_weights.push_back(get_weight(i, scale_weight_pos));
        inlier_weights.push_back(get_weight(i, orientation_weight_pos));
    }
}

}
