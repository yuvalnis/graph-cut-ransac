#pragma once

#include <vector>
#include <optional>
#include <cmath>
#include "model.h"
#include "solver_engine.h"
#include "math_utils.h"

namespace gcransac::estimator::solver
{

// This is the estimator class for estimating a rectifying homography matrix of an image. A model estimation method and error calculation method are implemented
class RectifyingHomographyThreeSIFTSolver : public SolverEngine<ScaleBasedRectifyingHomography, 1>
{
public:
    using Base = SolverEngine<ScaleBasedRectifyingHomography, 1>;
    using Model = typename Base::Model;
    using InlierContainerType = typename Base::InlierContainerType;
    using ResidualType = typename Base::ResidualType;
    using WeightType = typename Base::WeightType;

    RectifyingHomographyThreeSIFTSolver() {}
    ~RectifyingHomographyThreeSIFTSolver() {}

    // The minimum number of points required for the estimation
    inline std::array<size_t, 1> sampleSize() const override { return {3}; }

    // Estimate the model parameters from the given point sample
    // using weighted fitting if possible.
    bool estimateModel(
        const cv::Mat& data, // The set of data points
        const InlierContainerType& inliers,
        std::vector<ScaleBasedRectifyingHomography>& models, // The estimated model parameters
        const WeightType& weights
    ) const;

    ResidualType residual(
        const cv::Mat& feature,
        const ScaleBasedRectifyingHomography& model
    ) const;

    ResidualType squaredResidual(
        const cv::Mat& feature,
        const ScaleBasedRectifyingHomography& model
    ) const;

    bool normalizePoints(
        const cv::Mat& data,
        const std::vector<size_t>& inliers,
        cv::Mat& normalized_features,
        NormalizingTransform& normalizing_transform
    ) const;

    void getInlierWeightsInternal(
        const std::vector<double>& weights,
        const std::vector<size_t>& inliers,
        std::vector<double>& inlier_weights
    ) const;

protected:
    static constexpr double kScalePower = -1.0 / 3.0;
    static constexpr double kEpsilon = 1e-9;
    static constexpr size_t x_pos = 0; // x-coordinate position
    static constexpr size_t y_pos = 1; // y-coordinate position
    static constexpr size_t s_pos = 2; // scale position
    static constexpr size_t feature_size = 3;

    bool estimateNonMinimalModel(
        const cv::Mat &data,
        const std::vector<size_t>& inliers,
        std::vector<ScaleBasedRectifyingHomography> &models,
        const std::vector<double>& weights
    ) const;

    bool estimateMinimalModel(
        const cv::Mat& data,
        const std::vector<size_t>& inliers,
        std::vector<ScaleBasedRectifyingHomography>& models
    ) const;

};

bool RectifyingHomographyThreeSIFTSolver::estimateMinimalModel(
    const cv::Mat& data,
    const std::vector<size_t>& inliers,
    std::vector<ScaleBasedRectifyingHomography>& models
) const
{
    const auto sample_size = sampleSize()[0];
    if (inliers.size() != sample_size)
    {
        fprintf(
            stderr,
            "Minimal model requires exactly %ld samples (received %ld).\n",
            sample_size,
            inliers.size()
        );
        return false;
    }
    // helper function to fetch correct sample
    const auto* data_ptr = reinterpret_cast<double*>(data.data);
    auto get_inlier = [&data_ptr, &inliers, &data](
        const size_t& feature_idx
    )
    {
        const size_t& idx = inliers.empty() ?
                            feature_idx :
                            inliers[feature_idx];
        return data_ptr + idx * data.cols;
    };

    Eigen::Matrix<double, 3, 4> coeffs;
    for (size_t i = 0; i < inliers.size(); i++)
    {
        const auto* feature = get_inlier(i);
        const auto &x = feature[x_pos];
        const auto &y = feature[y_pos];
        const auto &s = feature[s_pos];

        coeffs(i, 0) = x;
        coeffs(i, 1) = y;
        coeffs(i, 2) = -pow(s, kScalePower);
        coeffs(i, 3) = -1.0;
    }
    Eigen::Matrix<double, 3, 1> x;
    gcransac::utils::gaussElimination<3>(coeffs, x);
    if (x.hasNaN())
    {
        return false;
    }
    // construct model
    ScaleBasedRectifyingHomography model;
    model.h7 = x(0);
    model.h8 = x(1);
    model.alpha = x(2);
    if (model.alpha < kEpsilon)
    {
        return false;
    }
    models.emplace_back(model);
    return true;
}

bool RectifyingHomographyThreeSIFTSolver::estimateNonMinimalModel(
    const cv::Mat &data,
    const std::vector<size_t>& inliers,
    std::vector<ScaleBasedRectifyingHomography> &models,
    const std::vector<double>& weights
) const
{
    // helper functions to fetch correct sample and weight
    const auto* data_ptr = reinterpret_cast<double*>(data.data);
    auto get_inlier = [&data_ptr, &inliers, &data](const size_t& feature_idx)
    {
        const size_t& idx = inliers.empty() ?
                            feature_idx :
                            inliers[feature_idx];
        return data_ptr + idx * data.cols;
    };
    auto get_weight = [&inliers, &weights](const size_t& feature_idx)
    {
        const size_t& idx = inliers.empty() ?
                            feature_idx :
                            inliers[feature_idx];
        return weights.empty() ? 1.0 : weights[idx];
    };

    const auto n_rows = inliers.size();
    Eigen::Matrix<double, Eigen::Dynamic, 3> coeffs(n_rows, 3);
    Eigen::VectorXd rhs(n_rows, 1);

    for (size_t i = 0; i < n_rows; ++i)
    {
        const auto* sample = get_inlier(i);
        const auto w = get_weight(i);
        const auto &x = sample[x_pos];
        const auto &y = sample[y_pos];
        const auto &s = sample[s_pos];

        coeffs(i, 0) = w * x;
        coeffs(i, 1) = w * y;
        coeffs(i, 2) = -w * pow(s, kScalePower);
        rhs(i) = -w;
    }
    // solve linear least squares system
    Eigen::Matrix<double, 3, 1> x = coeffs.colPivHouseholderQr().solve(rhs);
    // verify validity of solution
    if (x.hasNaN())
    {
        fprintf(stderr, "Invalid solution for the non-minimal model");
        return false;
    }
    // construct model
    ScaleBasedRectifyingHomography model;
    model.h7 = x(0);
    model.h8 = x(1);
    model.alpha = x(2);
    if (model.alpha < kEpsilon)
    {
        return false;
    }
    models.emplace_back(model);
    return true;
}

bool RectifyingHomographyThreeSIFTSolver::estimateModel(
    const cv::Mat& data, // The set of data points
    const InlierContainerType& inliers,
    std::vector<ScaleBasedRectifyingHomography>& models, // The estimated model parameters
    const WeightType& weights = WeightType{}
) const
{
    if (inliers.size() != 1 || weights.size() != 1)
    {
        fprintf(
            stderr,
            "The wrong number of inlier sets or weights sets was given. "
            "Expected 1 inlier set and 1 weight set. "
            "Received %ld inliers sets and %ld weights sets.\n",
            inliers.size(), weights.size()
        );
        return false;
    }
    const auto& inliers_vec = inliers[0];
    const auto sample_size = sampleSize()[0];
    if (inliers_vec.size() < sample_size)
    {
        fprintf(stderr,
            "There weren't enough SIFT features provided for the solver "
            "(%ld < %ld).\n", inliers_vec.size(), sample_size
        );
        return false;
    }
    if (inliers_vec.size() == sample_size)
    {
        return estimateMinimalModel(data, inliers_vec, models);
    }
    const auto& weights_vec = weights[0];
    return estimateNonMinimalModel(data, inliers_vec, models, weights_vec);
}

RectifyingHomographyThreeSIFTSolver::ResidualType RectifyingHomographyThreeSIFTSolver::residual(
    const cv::Mat& feature,
    const ScaleBasedRectifyingHomography& model
) const
{
    const auto* feature_ptr = reinterpret_cast<double*>(feature.data);
    Eigen::Vector3d point(feature_ptr[x_pos], feature_ptr[y_pos], 1.0);
    double scale = feature_ptr[s_pos];
    // Normalize coordinates and scale
    model.normalize(point);
    model.normalizeScale(scale);
    // Rectify scale.
    const auto rectified_scale = model.rectifiedScale(
        point(0), point(1), scale
    );
    // the model's estimation of the feature's cubed-scale in the rectified image
    const auto alpha_cube = std::pow(model.alpha, 3.0);
    // scale-based residual: logarithmic scale difference between the feature's
    // rectified scale and the model's estimated rectified scale for all features.
    const auto r_scale = std::fabs(std::log(rectified_scale / alpha_cube));

    return ResidualType{r_scale};
}

RectifyingHomographyThreeSIFTSolver::ResidualType RectifyingHomographyThreeSIFTSolver::squaredResidual(
    const cv::Mat& feature,
    const ScaleBasedRectifyingHomography& model
) const
{
    const auto r = residual(feature, model);
    return r * r;
}

bool RectifyingHomographyThreeSIFTSolver::normalizePoints(
    const cv::Mat& data, // The data points
    const std::vector<size_t>& inliers,
    cv::Mat& normalized_features, // The normalized features
    NormalizingTransform& normalizing_transform // the normalization transformation model
) const
{
    if (inliers.size() < 1)
    {
        fprintf(stderr,
            "Feature normalization failed because number of input features is zero.\n"
        );
        return false;
    }
    // helper function to fetch correct sample
    const auto* data_ptr = reinterpret_cast<double*>(data.data);
    auto get_inlier = [&data_ptr, &data, &inliers](const size_t& i) {
        const size_t idx = inliers.empty() ? i : inliers[i];
        return data_ptr + idx * data.cols;
    };
    // compute mean position of features
    normalizing_transform.x0 = 0.0;
    normalizing_transform.y0 = 0.0;
    for (size_t i = 0; i < inliers.size(); i++)
    {
        const auto* feature = get_inlier(i);
        normalizing_transform.x0 += feature[x_pos]; // x-coordinate
        normalizing_transform.y0 += feature[y_pos]; // y-coordinate
    }
    const auto inv_n = 1.0 / static_cast<double>(inliers.size());
    normalizing_transform.x0 *= inv_n;
    normalizing_transform.y0 *= inv_n;
    // compute average Euclidean distance to mean position
    double avg_dist = 0.0;
    for (size_t i = 0; i < inliers.size(); i++)
    {
        const auto* feature = get_inlier(i);
        const auto dx = feature[0] - normalizing_transform.x0; // x-coordinate
        const auto dy = feature[1] - normalizing_transform.y0; // y-coordinate
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
    for (size_t i = 0; i < inliers.size(); i++)
    {
        const auto* feature = get_inlier(i);
        auto norm_x = feature[x_pos]; // x-coordinate
        auto norm_y = feature[y_pos]; // y-coordinate
        auto norm_scale = feature[s_pos]; // scale

        normalizing_transform.normalize(norm_x, norm_y);
        normalizing_transform.normalizeScale(norm_scale);

        norm_features_ptr[i * normalized_features.cols + x_pos] = norm_x;
        norm_features_ptr[i * normalized_features.cols + y_pos] = norm_y;
        norm_features_ptr[i * normalized_features.cols + s_pos] = norm_scale;
        // ensures that if the dimension of the features is larger
        // than 3, then the normalization will still succeed.
        for (size_t j = feature_size; j < normalized_features.cols; j++)
        {
			norm_features_ptr[i * normalized_features.cols + j] = feature[j];
        }
    }

    return true;
}

}
