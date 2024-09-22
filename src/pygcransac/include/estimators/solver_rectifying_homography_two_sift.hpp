#pragma once

#include <vector>
#include <cmath>
#include <unordered_map>
#include "model.h"
#include "solver_engine.h"
#include "math_utils.hpp"

namespace gcransac::estimator::solver
{

class RectifyingHomographyTwoSIFTSolver : public SolverEngine<SIFTRectifyingHomography, 2>
{
public:
    using Base = SolverEngine<SIFTRectifyingHomography, 2>;
    using Model = typename Base::Model;
    using InlierContainerType = typename Base::InlierContainerType;
    using ResidualType = typename Base::ResidualType;
    using WeightType = typename Base::WeightType;

    RectifyingHomographyTwoSIFTSolver() {}
    ~RectifyingHomographyTwoSIFTSolver() {}

    inline std::array<size_t, 2> sampleSize() const
    {
        // 2 scale-samples amd 2 orientation-samples are required
        return {2, 2};
    }

    inline bool isValidSample(
		[[maybe_unused]] const cv::Mat& data,
		[[maybe_unused]] const InlierContainerType& inliers
	) const override
	{
        // TODO
		return true;
	}

    bool estimateModel(
        const cv::Mat& data,
        const InlierContainerType& inliers,
        std::vector<SIFTRectifyingHomography>& models,
        const WeightType& weights = WeightType{}
    ) const;

    ResidualType residual(
        const cv::Mat& feature,
        const SIFTRectifyingHomography& model
    ) const;

    ResidualType squaredResidual(
        const cv::Mat& feature,
        const SIFTRectifyingHomography& model
    ) const;

    bool normalizePoints(
        const cv::Mat& data,
        const std::vector<size_t>& inliers,
        cv::Mat& normalized_features,
        NormalizingTransform& normalizing_transform
    ) const;

protected:
    static constexpr double kScalePower = -1.0 / 3.0;
    static constexpr double kEpsilon = 1e-9;
    static constexpr size_t x_pos = 0; // x-coordinate position
    static constexpr size_t y_pos = 1; // y-coordinate position
    static constexpr size_t t_pos = 2; // orientation position
    static constexpr size_t s_pos = 3; // scale position
    static constexpr size_t feature_size = 4;
    static constexpr size_t scale_set_idx = 0;
    static constexpr size_t orient_set_idx = 1;

    static void setScaleConstraint(
        const double* feature,
        const size_t& idx,
        Eigen::Matrix<double, 3, 4>& coeffs
    );

    static void setOrientationConstraint(
        const double* feature1,
        const double* feature2,
        const size_t& idx,
        Eigen::Matrix<double, 3, 4>& coeffs
    );

    static void setScaleConstraint(
        const double* feature,
        const double& weight,
        const size_t& idx,
        Eigen::Matrix<double, Eigen::Dynamic, 3>& coeffs,
        Eigen::VectorXd& rhs
    );

    static void setOrientationConstraint(
        const double* feature1,
        const double* feature2,
        const double& weight1,
        const double& weight2,
        const size_t& idx,
        Eigen::Matrix<double, Eigen::Dynamic, 3>& coeffs,
        Eigen::VectorXd& rhs
    );

    inline static double rectifiedAngle(
        const double* feature,
        const SIFTRectifyingHomography& model
    );

    bool estimateNonMinimalModel(
        const cv::Mat &data,
        const std::vector<size_t>& scale_inliers,
        const std::vector<size_t>& orient_inliers,
        std::vector<SIFTRectifyingHomography>& models,
        const std::vector<double>& scale_weights,
        const std::vector<double>& orient_weights
    ) const;

    bool estimateMinimalModel(
        const cv::Mat& data,
        const std::vector<size_t>& scale_inliers,
        const std::vector<size_t>& orient_inliers,
        std::vector<SIFTRectifyingHomography>& models
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

void RectifyingHomographyTwoSIFTSolver::setScaleConstraint(
    const double* feature,
    const size_t& idx,
    Eigen::Matrix<double, 3, 4>& coeffs
)
{
    coeffs(idx, 0) = feature[x_pos];
    coeffs(idx, 1) = feature[y_pos];
    coeffs(idx, 2) = -pow(feature[s_pos], kScalePower);
    coeffs(idx, 3) = -1.0;
}

void RectifyingHomographyTwoSIFTSolver::setOrientationConstraint(
    const double* feature1,
    const double* feature2,
    const size_t& idx,
    Eigen::Matrix<double, 3, 4>& coeffs
)
{
    const auto l1 = lineFromSIFT(feature1[x_pos], feature1[y_pos], feature1[t_pos]);
    const auto l2 = lineFromSIFT(feature2[x_pos], feature2[y_pos], feature2[t_pos]);
    auto vp = l1.cross(l2); // intersection of lines is vanishing point
    const auto max_abs_value = vp.cwiseAbs().maxCoeff();
    if (max_abs_value > 1.0)
    {
        vp /= max_abs_value;
    }
    coeffs(idx, 0) = vp(0);
    coeffs(idx, 1) = vp(1);
    coeffs(idx, 2) = 0;
    coeffs(idx, 3) = -vp(2);
}

void RectifyingHomographyTwoSIFTSolver::setScaleConstraint(
    const double* feature,
    const double& weight,
    const size_t& idx,
    Eigen::Matrix<double, Eigen::Dynamic, 3>& coeffs,
    Eigen::VectorXd& rhs
)
{
    coeffs(idx, 0) = weight * feature[x_pos];
    coeffs(idx, 1) = weight * feature[y_pos];
    coeffs(idx, 2) = -weight * pow(feature[s_pos], kScalePower);
    rhs(idx) = -weight;
}

void RectifyingHomographyTwoSIFTSolver::setOrientationConstraint(
    const double* feature1,
    const double* feature2,
    const double& weight1,
    const double& weight2,
    const size_t& idx,
    Eigen::Matrix<double, Eigen::Dynamic, 3>& coeffs,
    Eigen::VectorXd& rhs
)
{
    const auto w = weight1 * weight2;
    const auto l1 = lineFromSIFT(feature1[x_pos], feature1[y_pos], feature1[t_pos]);
    const auto l2 = lineFromSIFT(feature2[x_pos], feature2[y_pos], feature2[t_pos]);
    auto vp = l1.cross(l2); // intersection of lines is vanishing point
    const auto max_abs_value = vp.cwiseAbs().maxCoeff();
    if (max_abs_value > 1.0)
    {
        vp /= max_abs_value;
    }
    coeffs(idx, 0) = w * vp(0);
    coeffs(idx, 1) = w * vp(1);
    coeffs(idx, 2) = 0.0;
    rhs(idx) = -w * vp(2);
}

double RectifyingHomographyTwoSIFTSolver::rectifiedAngle(
    const double* feature,
    const SIFTRectifyingHomography& model
)
{
    return fmod(model.rectifiedAngle(
        feature[x_pos], feature[y_pos], feature[t_pos]
    ), M_PI);
}

constexpr inline size_t nChoose2(const size_t& n)
{
    return (n * (n - 1)) / 2;
}

bool RectifyingHomographyTwoSIFTSolver::estimateMinimalModel(
    const cv::Mat& data,
    const std::vector<size_t>& scale_inliers,
    const std::vector<size_t>& orient_inliers,
    std::vector<SIFTRectifyingHomography>& models
) const
{
    const auto n_scale_constraints = scale_inliers.size();
    const auto n_orientation_constraints = nChoose2(orient_inliers.size());
    // make sure there are enough constraints from each type to estimate the model.
    if (n_scale_constraints != 2 || n_orientation_constraints != 1)
    {
        fprintf(
            stderr,
            "Incorrect combination of scale- and orientation-based "
            "constraints to estimate the minimal model.\n"
            "There are %ld scale-based constraints and %ld "
            "orientation-based constraints.\n",
            n_scale_constraints, n_orientation_constraints
        );
        return false;
    }
       
    // helper function to fetch correct inliers
    const auto* data_ptr = reinterpret_cast<double*>(data.data);
    auto get_scale_inlier = [&data_ptr, &scale_inliers, &data](
        const size_t& feature_idx
    )
    {
        const size_t& idx = scale_inliers.empty() ?
                            feature_idx :
                            scale_inliers[feature_idx];
        return data_ptr + idx * data.cols;
    };
    auto get_orientation_inlier = [&data_ptr, &orient_inliers, &data](
        const size_t& feature_idx
    )
    {
        const size_t& idx = orient_inliers.empty() ?
                            feature_idx :
                            orient_inliers[feature_idx];
        return data_ptr + idx * data.cols;
    };

    Eigen::Matrix<double, 3, 4> coeffs;
    size_t row_idx = 0;

    const auto* scale_inlier1 = get_scale_inlier(0);
    setScaleConstraint(scale_inlier1, row_idx++, coeffs);

    const auto* scale_inlier2 = get_scale_inlier(1);
    setScaleConstraint(scale_inlier2, row_idx++, coeffs);

    const auto* orientation_inlier1 = get_orientation_inlier(0);
    const auto* orientation_inlier2 = get_orientation_inlier(1);
    setOrientationConstraint(
        orientation_inlier1, orientation_inlier2, row_idx++, coeffs
    );

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
    const auto rectified_t1 = rectifiedAngle(orientation_inlier1, model);
    const auto rectified_t2 = rectifiedAngle(orientation_inlier2, model);
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
    models.emplace_back(model);
    return true;
} 

/// @brief Estimates the weighted-mode (most probable value) of the distribution
/// from which the angle samples where taken. This is done by placing the samples
/// in bins and finding the most frequent one. This functions treat an angle
/// theta and theta + PI as the same orientation.
/// @param angles a vector of angles in radians in range [0, PI).
/// @param weights a vector of weights in range [0, 1].
/// @param bin_width the width of the bins (must be a positive number).
/// @return a scalar represeting the estimated weighted-mode of the sampled distribution.
double findWeightedMode(
    const std::vector<double>& angles,
    const std::vector<double>& weights,
    const double& bin_width
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
        const auto& angle = angles[i];
        const auto& weight = weights[i];
        const int bin = static_cast<int>(std::round(angle / bin_width));
        weight_map[bin] += weight;
        bin_value_map[bin] += angle * weight;
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

bool RectifyingHomographyTwoSIFTSolver::estimateNonMinimalModel(
    const cv::Mat &data,
    const std::vector<size_t>& scale_inliers,
    const std::vector<size_t>& orient_inliers,
    std::vector<SIFTRectifyingHomography>& models,
    const std::vector<double>& scale_weights,
    const std::vector<double>& orient_weights
) const
{
    using namespace std;
    constexpr auto kBinWidth = M_PI / 360.0; // half-degree in radians

    const auto n_scale_constraints = scale_inliers.size();
    const auto n_orientation_constraints = nChoose2(orient_inliers.size());
    // make sure there are enough constraints from each type to estimate the model.
    if (n_scale_constraints < 2 || n_orientation_constraints < 1)
    {
        fprintf(
            stderr,
            "Insufficient combination of scale- and orientation-based "
            "constraints to estimate the non-minimal model.\n"
            "There are %ld scale-based constraints and %ld "
            "orientation-based constraints.\n",
            n_scale_constraints, n_orientation_constraints
        );
        return false;
    }
    if (!scale_weights.empty() && scale_weights.size() != scale_inliers.size())
    {
        fprintf(
            stderr,
            "Bad scale-weights container in non-minimal model. Container "
            "should be either empty (signifies uniform weights by default), or "
            "should be in the same size as the scale-inliers container. "
            "There are %ld weights and %ld inliers.\n",
            scale_weights.size(), scale_inliers.size()
        );
        return false;
    }
    if (!orient_weights.empty() && orient_weights.size() != orient_inliers.size())
    {
        fprintf(
            stderr,
            "Bad orientation-weights container in non-minimal model. Container "
            "should be either empty (signifies uniform weights by default), or "
            "should be in the same size as the orientation-inliers container. "
            "There are %ld weights and %ld inliers.\n",
            orient_weights.size(), orient_inliers.size()
        );
        return false;
    }
    // helper function to fetch correct inliers
    const auto* data_ptr = reinterpret_cast<double*>(data.data);
    auto get_scale_inlier = [&data_ptr, &scale_inliers, &data](
        const size_t& feature_idx
    )
    {
        const size_t& idx = scale_inliers.empty() ?
                            feature_idx :
                            scale_inliers[feature_idx];
        return data_ptr + idx * data.cols;
    };
    auto get_orientation_inlier = [&data_ptr, &orient_inliers, &data](
        const size_t& feature_idx
    )
    {
        const size_t& idx = orient_inliers.empty() ?
                            feature_idx :
                            orient_inliers[feature_idx];
        return data_ptr + idx * data.cols;
    };
    auto get_scale_weight = [&scale_inliers, &scale_weights](
        const size_t& feature_idx
    )
    {
        const size_t& idx = scale_inliers.empty() ?
                            feature_idx :
                            scale_inliers[feature_idx];
        return scale_weights.empty() ? 1.0 : scale_weights[idx];
    };
    auto get_orientation_weight = [&orient_inliers, &orient_weights](
        const size_t& feature_idx
    )
    {
        const size_t& idx = orient_inliers.empty() ?
                            feature_idx :
                            orient_inliers[feature_idx];
        return orient_weights.empty() ? 1.0 : orient_weights[idx];
    };
    // the number of rows in the coefficient matrix is the total number of constraints.
    const auto n_rows = n_scale_constraints + n_orientation_constraints;
    // populate coeffs and rhs matrices
    Eigen::Matrix<double, Eigen::Dynamic, 3> coeffs(n_rows, 3);
    Eigen::VectorXd rhs(n_rows, 1);
    // populate first sample_size rows of coeffs and rhs matrices with the
    // constraints derived from positions and scales
    size_t curr_idx = 0;
    for (size_t i = 0; i < scale_inliers.size(); i++)
    {
        const auto* scale_inlier = get_scale_inlier(i);
        const auto& scale_weight = get_scale_weight(i);
        setScaleConstraint(scale_inlier, scale_weight, curr_idx++, coeffs, rhs);
    }
    // populate last "sample_size choose 2" rows of coeffs and rhs matrices
    // with the constraints derived from positions and orientations
    for (size_t i = 0; i < orient_inliers.size() - 1; i++)
    {
        const auto* orient_inlier_i = get_orientation_inlier(i);
        const auto& orient_weight_i = get_orientation_weight(i);
        for (size_t j = i + 1; j < orient_inliers.size(); j++)
        {
            const auto* orient_inlier_j = get_orientation_inlier(j);
            const auto& orient_weight_j = get_orientation_weight(j);
            setOrientationConstraint(
                orient_inlier_i, orient_inlier_j,
                orient_weight_i, orient_weight_j,
                curr_idx++, coeffs, rhs
            );
        }
    }
    // verify coefficient matrix was constructed as expected.
    if (curr_idx != n_rows)
    {
        fprintf(
            stderr,
            "Error while computing coefficient matrix in the non-minimal solver:\n"
            "The number of constraints added to the matrix (%ld) is different from "
            "the number of rows of the matrix (%ld).\n",
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
    std::vector<double> rectified_angles(orient_inliers.size(), 0.0);
    std::vector<double> angle_weights(orient_inliers.size(), 0.0);
    for (size_t i = 0; i < orient_inliers.size(); i++)
    {
        const auto* orient_inlier = get_orientation_inlier(i);
        rectified_angles.at(i) = rectifiedAngle(orient_inlier, model);
        angle_weights.at(i) = get_orientation_weight(i);
    }
    model.vanishing_point_dir1 = findWeightedMode(
        rectified_angles, angle_weights, kBinWidth
    );
    // the second vanishing point's direction is orthogonal to the first.
    model.vanishing_point_dir2 = fmod(model.vanishing_point_dir1 + M_PI_2, M_PI);
    models.emplace_back(model);
    return true;
}

bool RectifyingHomographyTwoSIFTSolver::estimateModel(
    const cv::Mat& data,
    const InlierContainerType& inliers,
    std::vector<SIFTRectifyingHomography>& models,
    const WeightType& weights
) const
{
    if (inliers.size() < 2)
    {
        fprintf(
            stderr,
            "Not enought inlier sets were given for model estimation. "
            "Received %ld inlier sets.\n", inliers.size() 
        );
        return false;
    }
    const auto& scale_inliers = inliers[scale_set_idx];
    const auto& orient_inliers = inliers[orient_set_idx];
    const auto n_scale_constraints = scale_inliers.size();
    const auto n_orientation_constraints = nChoose2(orient_inliers.size());
    if (n_scale_constraints < 2 || n_orientation_constraints < 1)
    {
        fprintf(
            stderr,
            "Insufficient combination of scale- and orientation-based "
            "constraints to estimate model.\n"
            "There are %ld scale-based constraints and %ld "
            "orientation-based constraints.\n",
            n_scale_constraints, n_orientation_constraints
        );
        return false;
    }
    if (n_scale_constraints == 2 && n_orientation_constraints == 1)
    {
        return estimateMinimalModel(
            data, scale_inliers, orient_inliers, models
        );
    }
    const auto& scale_weights = weights[scale_set_idx];
    const auto& orient_weights = weights[orient_set_idx];
    return estimateNonMinimalModel(
        data, scale_inliers, orient_inliers, models,
        scale_weights, orient_weights
    );
}

RectifyingHomographyTwoSIFTSolver::ResidualType RectifyingHomographyTwoSIFTSolver::residual(
    const cv::Mat& feature,
    const SIFTRectifyingHomography& model
) const
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

RectifyingHomographyTwoSIFTSolver::ResidualType RectifyingHomographyTwoSIFTSolver::squaredResidual(
    const cv::Mat& feature,
    const SIFTRectifyingHomography& model
) const
{
    const auto r = residual(feature, model);
    return r * r;
}

bool RectifyingHomographyTwoSIFTSolver::normalizePoints(
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
    const auto n_cols = static_cast<size_t>(normalized_features.cols);
    for (size_t i = 0; i < inliers.size(); i++)
    {
        const auto* feature = get_inlier(i);
        auto norm_x = feature[x_pos]; // x-coordinate
        auto norm_y = feature[y_pos]; // y-coordinate
        auto norm_scale = feature[s_pos]; // scale

        normalizing_transform.normalize(norm_x, norm_y);
        normalizing_transform.normalizeScale(norm_scale);
        
        norm_features_ptr[i * n_cols + x_pos] = norm_x;
        norm_features_ptr[i * n_cols + y_pos] = norm_y;
        // orientation is not affected by translation and isotropic scaling
        norm_features_ptr[i * n_cols + t_pos] = feature[t_pos];
        norm_features_ptr[i * n_cols + s_pos] = norm_scale;
        // ensures that if the dimension of the features is larger
        // than 4, then the normalization will still succeed.
        for (size_t j = feature_size; j < n_cols; j++)
        {
			norm_features_ptr[i * n_cols + j] = feature[j];
        }
    }

    // for (size_t i = 0; i < inliers.size(); i++)
    // {
    //     const auto* feature = get_inlier(i);
    //     norm_features_ptr[i * n_cols + x_pos] = feature[x_pos];
    //     norm_features_ptr[i * n_cols + y_pos] = feature[y_pos];
    //     norm_features_ptr[i * n_cols + t_pos] = feature[t_pos];
    //     norm_features_ptr[i * n_cols + s_pos] = feature[s_pos];

    //     for (size_t j = feature_size; j < n_cols; j++)
    //     {
	// 	    norm_features_ptr[i * n_cols + j] = feature[j];
    //     }
    // }
    // normalizing_transform.x0 = 0.0;
    // normalizing_transform.y0 = 0.0;
    // normalizing_transform.s = 1.0;

    return true;
}

}
