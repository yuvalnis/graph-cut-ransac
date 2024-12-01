#pragma once

#include <vector>
#include <optional>
#include <cmath>
#include <sstream>
#include "model.h"
#include "solver_engine.h"
#include "math_utils.hpp"

namespace gcransac::estimator::solver
{

// This is the estimator class for estimating a rectifying homography matrix of an image. A model estimation method and error calculation method are implemented
class RectifyingHomographyThreeSIFTSolver : public SolverEngine<ScaleBasedRectifyingHomography, 1>
{
public:
    using Base = SolverEngine<ScaleBasedRectifyingHomography, 1>;
    using Model = typename Base::Model;
    using InlierContainerType = typename Base::InlierContainerType;
    using DataType = typename Base::DataType;
    using MutableDataType = typename Base::MutableDataType;
    using ResidualType = typename Base::ResidualType;
    using WeightType = typename Base::WeightType;

    RectifyingHomographyThreeSIFTSolver() {}
    ~RectifyingHomographyThreeSIFTSolver() {}

    // The minimum number of points required for the estimation
    inline std::array<size_t, 1> sampleSize() const { return {3}; }

    static bool areAllPointsCollinear(
        const cv::Mat& data,
		const std::vector<size_t>& inliers
    );
    
    inline bool isValidSample(
		const DataType& data,
		const InlierContainerType& inliers
	) const override
	{
        return !areAllPointsCollinear(*(data[0].get()), inliers[0]);
	}

    bool isValidModelInternal(
		const Model& model,
        const cv::Mat& scale_features,
        const std::vector<size_t>& scale_inliers
	) const;

    inline bool isValidModel(
		const Model& model,
		const DataType& data,
		const InlierContainerType& inliers,
		[[maybe_unused]] const InlierContainerType& minimal_sample,
		[[maybe_unused]] const ResidualType& threshold,
		[[maybe_unused]] bool& model_updated
	) const override
    {
        return isValidModelInternal(model, *(data[0].get()), inliers[0]);
    }

    // Estimate the model parameters from the given point sample
    // using weighted fitting if possible.
    bool estimateModel(
        const DataType& data, // The set of data points
        const InlierContainerType& inliers,
        std::vector<ScaleBasedRectifyingHomography>& models, // The estimated model parameters
        const WeightType& weights
    ) const;

    static double scaleResidual(
        double x, double y, double s, const ScaleBasedRectifyingHomography& model
    );

    double residual(
        size_t type, const cv::Mat& feature,
        const ScaleBasedRectifyingHomography& model
    ) const;

    double squaredResidual(
        size_t type, const cv::Mat& feature,
        const ScaleBasedRectifyingHomography& model
    ) const;

    static bool internalNormalizePoints(
        const cv::Mat& data,
        const std::vector<size_t>& inliers,
        cv::Mat& normalized_features,
        NormalizingTransform& normalizing_transform
    );

    bool normalizePoints(
        const DataType& data,
        const InlierContainerType& inliers,
        MutableDataType& normalized_features,
        NormalizingTransform& normalizing_transform
    ) const;

    void getInlierWeightsInternal(
        const std::vector<double>& weights,
        const std::vector<size_t>& inliers,
        std::vector<double>& inlier_weights
    ) const;

protected:
    static constexpr double kScalePower = 1.0 / 3.0;
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

bool RectifyingHomographyThreeSIFTSolver::areAllPointsCollinear(
    const cv::Mat& data,
    const std::vector<size_t>& inliers
)
{
    constexpr double kMaxSmallestAngle{5.0}; // Maximum size of smallest angle in triangle (in degrees)
    // Check that all points are collinear
    const auto n = inliers.size();
    if (n < 3)
    {
        // Less than 3 points are always collinear by definition
        return true;
    }
    // compute collinearity tolerance
    const double tolerance = std::abs(std::sin(kMaxSmallestAngle * M_PI / 180.0));
    // check every triplet of points is collinear
    for (size_t i = 0; i < n - 2; i++)
    {
        auto idx1 = inliers.at(i);
        auto idx2 = inliers.at(i + 1);
        auto idx3 = inliers.at(i + 2);
        double x1 = data.at<double>(idx1, x_pos);
        double y1 = data.at<double>(idx1, y_pos);
        double x2 = data.at<double>(idx2, x_pos);
        double y2 = data.at<double>(idx2, y_pos);
        double x3 = data.at<double>(idx3, x_pos);
        double y3 = data.at<double>(idx3, y_pos);
        if (!utils::areCollinear(x1, y1, x2, y2, x3, y3, tolerance))
        {
            return false;
        }
    }

    return true;
}

bool RectifyingHomographyThreeSIFTSolver::isValidModelInternal(
    const Model& model,
    const cv::Mat& scale_features,
    const std::vector<size_t>& scale_inliers
) const
{
    // the model should not map detected (non-zero) scales to non-positive
    // scales in the rectified image
    for (const auto& idx : scale_inliers)
    {
        auto x = scale_features.at<double>(idx, x_pos);
        auto y = scale_features.at<double>(idx, y_pos);
        auto s = scale_features.at<double>(idx, s_pos);
        if (model.rectifiedScale(x, y, s) < kEpsilon)
        {
            return false;
        }
    }
    return true;
}

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
    Eigen::Matrix<double, 3, 4> coeffs;
    for (size_t i = 0; i < inliers.size(); i++)
    {
        auto inlier_idx = inliers.at(i);
        double x = data.at<double>(inlier_idx, x_pos);
        double y = data.at<double>(inlier_idx, y_pos);
        double s = data.at<double>(inlier_idx, s_pos);

        coeffs(i, 0) = x;
        coeffs(i, 1) = y;
        coeffs(i, 2) = pow(s, kScalePower);
        coeffs(i, 3) = 1.0;
    }
    // Eigen::Matrix<double, 3, 1> solution = coeffs.colPivHouseholderQr().solve(rhs);
    Eigen::Matrix<double, 3, 1> solution;
    gcransac::utils::gaussElimination<3>(coeffs, solution);
    if (solution.hasNaN())
    {
        return false;
    }
    // construct model
    ScaleBasedRectifyingHomography model;
    model.h7 = solution(0);
    model.h8 = solution(1);
    // The solution includes an estimate pf the inverse of the alpha parameter,
    // since it uses XY coordinates in the warped image space, and not the
    // rectified image space, unlike the method in Chum's paper. 
    double inv_alpha = solution(2);
    if (inv_alpha < kEpsilon)
    {
        return false;
    }
    model.alpha = 1.0 / inv_alpha;
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
        coeffs(i, 2) = w * pow(s, kScalePower);
        rhs(i) = w;
    }
    // solve linear least squares system
    Eigen::Matrix<double, 3, 1> solution = coeffs.colPivHouseholderQr().solve(rhs);
    // verify validity of solution
    if (solution.hasNaN())
    {
        return false;
    }
    // construct model
    ScaleBasedRectifyingHomography model;
    model.h7 = solution(0);
    model.h8 = solution(1);
    model.alpha = solution(2);
    if (model.alpha < kEpsilon)
    {
        return false;
    }
    models.emplace_back(model);
    return true;
}

bool RectifyingHomographyThreeSIFTSolver::estimateModel(
    const DataType& data, // The set of data points
    const InlierContainerType& inliers,
    std::vector<ScaleBasedRectifyingHomography>& models, // The estimated model parameters
    const WeightType& weights = WeightType{}
) const
{
    if (data.size() != 1 || inliers.size() != 1 || weights.size() != 1)
    {
        fprintf(
            stderr,
            "The wrong number of data sets, inlier sets or weights sets was "
            "given. Expected 1 data set, 1 inlier set and 1 weight set. "
            "Received %ld data sets, %ld inliers sets and %ld weights sets.\n",
            data.size(), inliers.size(), weights.size()
        );
        return false;
    }
    const auto& data_set = *(data[0].get()); // TODO change to use unique_ptr directly
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
        return estimateMinimalModel(data_set, inliers_vec, models);
    }
    const auto& weights_vec = weights[0];
    return estimateNonMinimalModel(data_set, inliers_vec, models, weights_vec);
}

double RectifyingHomographyThreeSIFTSolver::scaleResidual(
    double x, double y, double s, const ScaleBasedRectifyingHomography& model
)
{
    Eigen::Vector3d point(x, y, 1.0);
    double scale = s;
    // Normalize coordinates and scale
    model.normalize(point);
    model.normalizeScale(scale);
    // Rectify  scale
    const auto rectified_scale = model.rectifiedScale(
        point(0), point(1), scale
    );
    // the model's estimation of the feature's cubed-scale in the rectified image
    const auto alpha_cube = utils::cube(model.alpha);
    // scale-based residual: logarithmic scale difference between the feature's
    // rectified scale and the model's estimated rectified scale for all features.
    return std::fabs(std::log(rectified_scale / alpha_cube));
}

double RectifyingHomographyThreeSIFTSolver::residual(
    size_t type, const cv::Mat& feature,
    const ScaleBasedRectifyingHomography& model
) const
{
    if (type != 0)
    {
        std::stringstream err_msg;
        err_msg << "Invalid type argument in class method "
                << "RectifyingHomographyThreeSIFTSolver::residual. Expected 0, "
                << "but received " << type << std::endl;
        throw std::runtime_error(err_msg.str());
    }
    const auto r_scale = scaleResidual(
        feature.at<double>(0, x_pos),
        feature.at<double>(0, y_pos),
        feature.at<double>(0, s_pos),
        model
    );
    return r_scale;
}

double RectifyingHomographyThreeSIFTSolver::squaredResidual(
    size_t type, const cv::Mat& feature,
    const ScaleBasedRectifyingHomography& model
) const
{
    const auto r = residual(type, feature, model);
    return r * r;
}

bool RectifyingHomographyThreeSIFTSolver::internalNormalizePoints(
    const cv::Mat& data,
    const std::vector<size_t>& inliers,
    cv::Mat& normalized_features,
    NormalizingTransform& normalizing_transform
)
{
    if (inliers.size() < 1)
    {
        fprintf(stderr,
            "Feature normalization failed because number of input features is zero.\n"
        );
        return false;
    }
    // compute mean position of features
    normalizing_transform.x0 = 0.0;
    normalizing_transform.y0 = 0.0;
    for (size_t i = 0; i < inliers.size(); i++)
    {
        auto inlier_idx = inliers[i];
        auto x = data.at<double>(inlier_idx, x_pos);
        auto y = data.at<double>(inlier_idx, y_pos);
        normalizing_transform.x0 += x; // x-coordinate
        normalizing_transform.y0 += y; // y-coordinate
    }
    const auto inv_n = 1.0 / static_cast<double>(inliers.size());
    normalizing_transform.x0 *= inv_n;
    normalizing_transform.y0 *= inv_n;
    // compute average Euclidean distance to mean position
    double avg_dist = 0.0;
    for (size_t i = 0; i < inliers.size(); i++)
    {
        auto inlier_idx = inliers[i];
        auto x = data.at<double>(inlier_idx, x_pos);
        auto y = data.at<double>(inlier_idx, y_pos);
        auto dx = x - normalizing_transform.x0; // x-coordinate
        auto dy = y - normalizing_transform.y0; // y-coordinate
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
    // for (size_t i = 0; i < inliers.size(); i++)
    // {
    //     auto inlier_idx = inliers[i];
    //     auto x = data.at<double>(inlier_idx, x_pos);
    //     auto y = data.at<double>(inlier_idx, y_pos);
    //     auto scale = data.at<double>(inlier_idx, s_pos);
    //     normalizing_transform.normalize(x, y);
    //     normalizing_transform.normalizeScale(scale);
    //     normalized_features.at<double>(i, x_pos) = x;
    //     normalized_features.at<double>(i, y_pos) = y;
    //     normalized_features.at<double>(i, s_pos) = scale;
    // }

    for (size_t i = 0; i < inliers.size(); i++)
    {
        auto inlier_idx = inliers[i];
        auto x = data.at<double>(inlier_idx, x_pos);
        auto y = data.at<double>(inlier_idx, y_pos);
        auto scale = data.at<double>(inlier_idx, s_pos);
        normalized_features.at<double>(i, x_pos) = x;
        normalized_features.at<double>(i, y_pos) = y;
        normalized_features.at<double>(i, s_pos) = scale;
    }
    normalizing_transform.x0 = 0.0;
    normalizing_transform.y0 = 0.0;
    normalizing_transform.s = 1.0;

    return true;
}

bool RectifyingHomographyThreeSIFTSolver::normalizePoints(
    const DataType& data, // The data points
    const InlierContainerType& inliers,
    MutableDataType& normalized_features, // The normalized features
    NormalizingTransform& normalizing_transform // the normalization transformation model
) const
{
    // TODO change to use unique_ptr directly
    return internalNormalizePoints(
        *(data[0].get()), inliers[0], *(normalized_features[0].get()),
        normalizing_transform
    );
}

}
