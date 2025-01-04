#pragma once

#include <vector>
#include <cmath>
#include <limits.h>
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
    using DataType = typename Base::DataType;
    using MutableDataType = typename Base::MutableDataType;
    using ResidualType = typename Base::ResidualType;
    using WeightType = typename Base::WeightType;

    RectifyingHomographyTwoSIFTSolver() {}
    ~RectifyingHomographyTwoSIFTSolver() {}

    inline std::array<size_t, 2> sampleSize() const
    {
        // 2 scale-samples amd 2 orientation-samples are required
        return {2, 2};
    }

    bool isValidSample(
		const DataType& data,
		const InlierContainerType& inliers
	) const override;

    bool isValidModelInternal(
		const Model& model,
        const cv::Mat& scale_features,
        const std::vector<size_t>& scale_inliers
	) const;

    inline bool isValidModel(
		const Model& model,
		[[maybe_unused]] const DataType& data,
		[[maybe_unused]] const InlierContainerType& inliers,
		[[maybe_unused]] const InlierContainerType& minimal_sample,
		[[maybe_unused]] const ResidualType& threshold,
		[[maybe_unused]] bool& model_updated
	) const override
    {
        constexpr double kValidHValue{1e-3};
        auto h_abs_max = std::fmax(std::fabs(model.h7), std::fabs(model.h8));
        if (h_abs_max >= kValidHValue)
        {
            return false;
        }
        return true;
    }

    bool estimateModel(
        const DataType& data,
        const InlierContainerType& inliers,
        std::vector<SIFTRectifyingHomography>& models,
        const WeightType& weights = WeightType{}
    ) const;

    static double scaleResidual(
        double x, double y, double s, const SIFTRectifyingHomography& model
    );

    static double orientationResidual(
        double x, double y, double t, const SIFTRectifyingHomography& model
    );

    double residual(
        size_t type, const cv::Mat& feature,
        const SIFTRectifyingHomography& model
    ) const;

    double squaredResidual(
        size_t type, const cv::Mat& feature,
        const SIFTRectifyingHomography& model
    ) const;

    bool normalizePoints(
        const DataType& data,
        const InlierContainerType& inliers,
        MutableDataType& normalized_features,
        NormalizingTransform& normalizing_transform
    ) const;

protected:
    static constexpr double kScalePower = 1.0 / 3.0;
    static constexpr double kEpsilon = 1e-9;
    static constexpr size_t x_pos = 0; // x-coordinate position
    static constexpr size_t y_pos = 1; // y-coordinate position
    static constexpr size_t t_pos = 2; // orientation position
    static constexpr size_t s_pos = 2; // scale position
    static constexpr size_t scale_set_idx = 0;
    static constexpr size_t orient_set_idx = 1;

    static void setScaleConstraint(
        double x, double y, double scale, size_t row_idx,
        Eigen::Matrix<double, 3, 4>& coeffs
    );

    static void setScaleConstraint(
        double x, double y, double scale, double weight, size_t row_idx, 
        Eigen::Matrix<double, Eigen::Dynamic, 3>& coeffs, Eigen::VectorXd& rhs
    );

    static void setOrientationConstraint(
        double x1, double y1, double theta1, double weight1,
        double x2, double y2, double theta2, double weight2,
        size_t row_idx, Eigen::Matrix<double, Eigen::Dynamic, 3>& coeffs,
        Eigen::VectorXd& rhs
    );

    bool estimateNonMinimalModel(
        const std::unique_ptr<const cv::Mat>& scale_features,
        const std::vector<size_t>& scale_inliers,
        const std::unique_ptr<const cv::Mat>& orientation_features,
        const std::vector<size_t>& orient_inliers,
        std::vector<SIFTRectifyingHomography>& models,
        const std::vector<double>& scale_weights,
        const std::vector<double>& orient_weights
    ) const;

    bool estimateMinimalModel(
        const std::unique_ptr<const cv::Mat>& scale_features,
        const std::vector<size_t>& scale_inliers,
        const std::unique_ptr<const cv::Mat>& orientation_features,
        const std::vector<size_t>& orient_inliers,
        std::vector<SIFTRectifyingHomography>& models
    ) const;
};

bool RectifyingHomographyTwoSIFTSolver::isValidSample(
    const DataType& data,
    const InlierContainerType& inliers
) const
{
    constexpr double kCollinearityThresh{1.0};
    const auto& scale_features = data[scale_set_idx];
    const auto& scale_inliers = inliers[scale_set_idx];
    const auto& orient_features = data[orient_set_idx];
    const auto& orient_inliers = inliers[orient_set_idx];
    if (scale_inliers.size() !=2 || orient_inliers.size() != 2)
    {
        fprintf(
            stderr,
            "RectifyingHomographyTwoSIFTSolver::isValidSample called with "
            "incorrect numbers of inliers (%ld scale inliers and %ld "
            "orientation inliers).\n",
            scale_inliers.size(), orient_inliers.size()
        );
        return false;
    }
    // compute line from first orientation feature
    auto idx = orient_inliers.at(0);
    auto x1 = orient_features->at<double>(idx, x_pos);
    auto y1 = orient_features->at<double>(idx, y_pos);
    auto t1 = orient_features->at<double>(idx, t_pos);
    auto l1 = utils::lineFromPointAndAngle(x1, y1, t1);
    // compute line from second orientation feature
    idx = orient_inliers.at(1);
    auto x2 = orient_features->at<double>(idx, x_pos);
    auto y2 = orient_features->at<double>(idx, y_pos);
    auto t2 = orient_features->at<double>(idx, t_pos);
    auto l2 = utils::lineFromPointAndAngle(x2, y2, t2);
    // compute vanishing point
    auto vp = l1.cross(l2); // intersection of lines is vanishing point
    if ((vp.array().cwiseAbs() < 1e-6).all())
    {
        // vanishing point is the degenerate zero-vector
        return false;
    }
    if (std::abs(vp[2]) < 1e-6)
    {
        // If vanishing point is at infinity, it is outside the convex-hull ,
        // and would have been zero if it were collinear with the scale features
        return true;
    }
    vp /= vp[2];
    utils::Point2D vp2d{vp[0], vp[1]};
    // create Point2D object from first scale feature coordinates
    idx = scale_inliers.at(0);
    auto x = scale_features->at<double>(idx, x_pos);
    auto y = scale_features->at<double>(idx, y_pos);
    utils::Point2D p1{x, y};
    // create Point2D object from second scale feature coordinates
    idx = scale_inliers.at(1);
    x = scale_features->at<double>(idx, x_pos);
    y = scale_features->at<double>(idx, y_pos);
    utils::Point2D p2{x, y};
    // compute collinearity tolerance
    if (utils::areCollinear(p1, p2, vp2d, kCollinearityThresh))
    {
        return false;
    }
    // compute convex-hull of all detected points
    std::vector<utils::Point2D> points{
        p1, p2, {x1, y1}, {x2, y2}
    };
    const auto convex_hull = utils::computeConvexHull(points);
    // check if vanishing point is inside the convex-hull
    if (utils::pointInConvexPolygon(vp2d, convex_hull))
    {
        return false;
    }
    return true;
}

void RectifyingHomographyTwoSIFTSolver::setScaleConstraint(
    double x, double y, double scale, size_t row_idx,
    Eigen::Matrix<double, 3, 4>& coeffs
)
{
    coeffs(row_idx, 0) = x;
    coeffs(row_idx, 1) = y;
    coeffs(row_idx, 2) = pow(scale, kScalePower);
    coeffs(row_idx, 3) = 1.0;
}

void RectifyingHomographyTwoSIFTSolver::setScaleConstraint(
    double x, double y, double scale, double weight, size_t row_idx, 
    Eigen::Matrix<double, Eigen::Dynamic, 3>& coeffs, Eigen::VectorXd& rhs
)
{
    coeffs(row_idx, 0) = weight * x;
    coeffs(row_idx, 1) = weight * y;
    coeffs(row_idx, 2) = weight * pow(scale, kScalePower);
    rhs(row_idx) = weight;
}

void RectifyingHomographyTwoSIFTSolver::setOrientationConstraint(
    double x1, double y1, double theta1, double weight1,
    double x2, double y2, double theta2, double weight2,
    size_t row_idx, Eigen::Matrix<double, Eigen::Dynamic, 3>& coeffs,
    Eigen::VectorXd& rhs
)
{
    const auto w = weight1 * weight2;
    const auto l1 = utils::lineFromPointAndAngle(x1, y1, theta1);
    const auto l2 = utils::lineFromPointAndAngle(x2, y2, theta2);
    auto vp = l1.cross(l2); // intersection of lines is vanishing point
    const auto max_abs_value = vp.cwiseAbs().maxCoeff();
    if (max_abs_value > 1.0)
    {
        vp /= max_abs_value;
    }
    coeffs(row_idx, 0) = w * vp(0);
    coeffs(row_idx, 1) = w * vp(1);
    coeffs(row_idx, 2) = 0.0;
    rhs(row_idx) = w * vp(2);
}

bool RectifyingHomographyTwoSIFTSolver::estimateMinimalModel(
    const std::unique_ptr<const cv::Mat>& scale_features,
    const std::vector<size_t>& scale_inliers,
    const std::unique_ptr<const cv::Mat>& orientation_features,
    const std::vector<size_t>& orient_inliers,
    std::vector<SIFTRectifyingHomography>& models
) const
{
    const auto n_scale_constraints = scale_inliers.size();
    const auto n_orientation_constraints = utils::nChoose2(orient_inliers.size());
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
       
    Eigen::Matrix<double, 3, 4> coeffs;

    auto inlier_idx = scale_inliers[0];
    double x = scale_features->at<double>(inlier_idx, x_pos);
    double y = scale_features->at<double>(inlier_idx, y_pos);
    double s = scale_features->at<double>(inlier_idx, s_pos);
    setScaleConstraint(x, y, s, 0, coeffs);

    inlier_idx = scale_inliers[1];
    x = scale_features->at<double>(inlier_idx, x_pos);
    y = scale_features->at<double>(inlier_idx, y_pos);
    s = scale_features->at<double>(inlier_idx, s_pos);
    setScaleConstraint(x, y, s, 1, coeffs);

    inlier_idx = orient_inliers[0];
    double x1 = orientation_features->at<double>(inlier_idx, x_pos);
    double y1 = orientation_features->at<double>(inlier_idx, y_pos);
    double theta1 = orientation_features->at<double>(inlier_idx, t_pos);

    inlier_idx = orient_inliers[1];
    double x2 = orientation_features->at<double>(inlier_idx, x_pos);
    double y2 = orientation_features->at<double>(inlier_idx, y_pos);
    double theta2 = orientation_features->at<double>(inlier_idx, t_pos);

    auto l1 = utils::lineFromPointAndAngle(x1, y1, theta1);
    auto l2 = utils::lineFromPointAndAngle(x2, y2, theta2);
    auto vp = l1.cross(l2); // intersection of lines is vanishing point
    coeffs(2, 0) = vp(0);
    coeffs(2, 1) = vp(1);
    coeffs(2, 2) = 0;
    coeffs(2, 3) = vp(2);

    Eigen::Matrix<double, 3, 1> solution;
    gcransac::utils::gaussElimination<3>(coeffs, solution);
    if (solution.hasNaN())
    {
        return false;
    }
    // construct model
    SIFTRectifyingHomography model;
    model.h7 = solution(0);
    model.h8 = solution(1);
    model.alpha = solution(2);
    if (model.alpha < kEpsilon)
    {
        return false;
    }
    // Compute the directions of the vanishing points.
    // Vanishing point should be mapped to infinity in rectified image.
    // Model is for warping homography, so we "unrectify" the vanishing point
    // to map it to the rectified image.
    model.rectifyPoint(vp);
    if (std::abs(vp[2]) > kEpsilon)
    {
        return false;
    }
    model.vanishing_point_dir = utils::clipAngle(std::atan2(vp[1], vp[0]));
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

size_t nonZeroWeightInliers(
    const std::vector<size_t>& inliers,
    const std::vector<double>& weights,
    std::vector<size_t>& result
)
{
    if (weights.empty())
    {
        result = inliers;
        return result.size();
    }
    if (inliers.size() != weights.size())
    {
        throw std::runtime_error(
            "nonZeroWeightInliers: number of weights and inliers is different."
        );
    }
    for (size_t i = 0; i < inliers.size(); i++)
    {
        if (weights.at(i) > 1e-9)
        {
            result.push_back(inliers.at(i));
        }
    }
    return result.size();
}

bool RectifyingHomographyTwoSIFTSolver::estimateNonMinimalModel(
    const std::unique_ptr<const cv::Mat>& scale_features,
    const std::vector<size_t>& scale_inliers,
    const std::unique_ptr<const cv::Mat>& orient_features,
    const std::vector<size_t>& orient_inliers,
    std::vector<SIFTRectifyingHomography>& models,
    const std::vector<double>& scale_weights,
    const std::vector<double>& orient_weights
) const
{
    using namespace std;
    const auto kBinWidth = utils::deg2rad(0.5); // half-degree in radians

    std::vector<size_t> actual_scale_inliers;
    std::vector<size_t> actual_orient_inliers;
    auto n_scale_inliers = nonZeroWeightInliers(scale_inliers, scale_weights,
                                                actual_scale_inliers);
    auto n_orient_inliers = nonZeroWeightInliers(orient_inliers, orient_weights,
                                                 actual_orient_inliers);
    const auto n_scale_constraints = n_scale_inliers;
    const auto n_orientation_constraints = utils::nChoose2(n_orient_inliers);
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
    // helper function to fetch correct weights
    auto get_scale_weight = [&actual_scale_inliers, &scale_weights](
        const size_t& feature_idx
    )
    {
        const size_t& idx = actual_scale_inliers[feature_idx];
        return scale_weights.empty() ? 1.0 : scale_weights[idx];
    };
    auto get_orientation_weight = [&actual_orient_inliers, &orient_weights](
        const size_t& feature_idx
    )
    {
        const size_t& idx = actual_orient_inliers[feature_idx];
        return orient_weights.empty() ? 1.0 : orient_weights[idx];
    };
    // the number of rows in the coefficient matrix is the total number of constraints.
    const auto n_rows = n_scale_constraints + n_orientation_constraints;
    // populate coeffs and rhs matrices
    Eigen::Matrix<double, Eigen::Dynamic, 3> coeffs(n_rows, 3);
    Eigen::VectorXd rhs(n_rows, 1);
    // populate first sample_size rows of coeffs and rhs matrices with the
    // constraints derived from positions and scales
    size_t curr_idx{0};
    size_t inlier_idx{0};
    for (size_t i = 0; i < actual_scale_inliers.size(); i++)
    {
        inlier_idx = actual_scale_inliers[i];
        auto x = scale_features->at<double>(inlier_idx, x_pos);
        auto y = scale_features->at<double>(inlier_idx, y_pos);
        auto scale = scale_features->at<double>(inlier_idx, s_pos);
        auto weight = get_scale_weight(i);
        setScaleConstraint(x, y, scale, weight, curr_idx++, coeffs, rhs);
    }
    // populate last "sample_size choose 2" rows of coeffs and rhs matrices
    // with the constraints derived from positions and orientations
    for (size_t i = 0; i < actual_orient_inliers.size() - 1; i++)
    {
        inlier_idx = actual_orient_inliers[i];
        auto xi = orient_features->at<double>(inlier_idx, x_pos); 
        auto yi = orient_features->at<double>(inlier_idx, y_pos); 
        auto ti = orient_features->at<double>(inlier_idx, t_pos);
        auto wi = get_orientation_weight(i);
        for (size_t j = i + 1; j < actual_orient_inliers.size(); j++)
        {
            inlier_idx = actual_orient_inliers[j];
            auto xj = orient_features->at<double>(inlier_idx, x_pos); 
            auto yj = orient_features->at<double>(inlier_idx, y_pos); 
            auto tj = orient_features->at<double>(inlier_idx, t_pos);
            auto wj = get_orientation_weight(j); 
            setOrientationConstraint(
                xi, yi, ti, wi, xj, yj, tj, wj, curr_idx++, coeffs, rhs
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
    Eigen::Matrix<double, 3, 1> solution = coeffs.colPivHouseholderQr().solve(rhs);
    // verify validity of solution
    if (solution.hasNaN())
    {
        return false;
    }
    // construct model
    SIFTRectifyingHomography model;
    model.h7 = solution(0);
    model.h8 = solution(1);
    model.alpha = solution(2);
    if (model.alpha < kEpsilon)
    {
        return false;
    }
    std::vector<double> rectified_angles(actual_orient_inliers.size(), 0.0);
    std::vector<double> angle_weights(actual_orient_inliers.size(), 0.0);
    double weights_sum{0};
    for (size_t i = 0; i < actual_orient_inliers.size(); i++)
    {
        auto inlier_idx = actual_orient_inliers[i];
        auto x = orient_features->at<double>(inlier_idx, x_pos);
        auto y = orient_features->at<double>(inlier_idx, y_pos);
        auto theta = orient_features->at<double>(inlier_idx, t_pos);
        rectified_angles.at(i) = model.rectifiedAngle(x, y, theta);
        double weight = get_orientation_weight(i);
        angle_weights.at(i) = weight;
        weights_sum += weight;
    }
    if (weights_sum < kEpsilon)
    {
        fprintf(
            stderr,
            "Error in non-minimal solver: sum of weights (%f) is considered "
            "non-positive.\n", weights_sum
        );
        return false;
    }
    // normalize angle and weights for mode computation
    for (size_t i = 0; i < actual_orient_inliers.size(); i++)
    {
        auto& angle = rectified_angles.at(i);
        if (angle > M_PI)
        {
            // a line with angle theta and a line with angle theta + PI,
            // both with the same intersection, are equivalent.
            angle -= M_PI;
        }
        angle_weights.at(i) /= weights_sum;
    }
    model.vanishing_point_dir = findWeightedMode( 
        rectified_angles, angle_weights, kBinWidth
    );
    models.emplace_back(model);
    return true;
}

bool RectifyingHomographyTwoSIFTSolver::estimateModel(
    const DataType& data,
    const InlierContainerType& inliers,
    std::vector<SIFTRectifyingHomography>& models,
    const WeightType& weights
) const
{
    const auto& scale_features = data[scale_set_idx];
    const auto& scale_inliers = inliers[scale_set_idx];
    const auto& orient_features = data[orient_set_idx];
    const auto& orient_inliers = inliers[orient_set_idx];
    const auto n_scale_constraints = scale_inliers.size();
    const auto n_orientation_constraints = utils::nChoose2(orient_inliers.size());
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
            scale_features, scale_inliers, orient_features, orient_inliers,
            models
        );
    }
    const auto& scale_weights = weights[scale_set_idx];
    const auto& orient_weights = weights[orient_set_idx];
    return estimateNonMinimalModel(
        scale_features, scale_inliers, orient_features, orient_inliers, models,
        scale_weights, orient_weights
    );
}

double RectifyingHomographyTwoSIFTSolver::scaleResidual(
    double x, double y, double s, const SIFTRectifyingHomography& model
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
    // We assume an inlier scale feature can't possibly be rectified into a
    // non-positive scale in the rectified image of a plane.
    if (rectified_scale < kEpsilon)
    {
        return DBL_MAX;
    }
    // the model's estimation of the feature's cubed-scale in the rectified image
    const auto alpha_cube = utils::cube(model.alpha);
    // scale-based residual: logarithmic scale difference between the feature's
    // rectified scale and the model's estimated rectified scale for all features.
    return std::fabs(std::log(alpha_cube * rectified_scale));
}

double RectifyingHomographyTwoSIFTSolver::orientationResidual(
    double x, double y, double t, const SIFTRectifyingHomography& model
)
{
    Eigen::Vector3d point(x, y, 1.0);
    // Normalize coordinates (orientation is unchanged by normalization)
    model.normalize(point);
    // Rectify orientation
    const auto rectified_orientation = model.rectifiedAngle(
        point(0), point(1), t
    );
    // orientation-based residual: the minimal angular distance between the
    // feature's rectified orientation and the model's principle orientation.  
    return utils::linesAnglesDiff(model.vanishing_point_dir, rectified_orientation);
}

double RectifyingHomographyTwoSIFTSolver::residual(
    size_t type, const cv::Mat& feature, const SIFTRectifyingHomography& model
) const
{
    double r{DBL_MAX};
    if (feature.rows != 1)
    {
        std::stringstream err_msg;
        err_msg << "Invalid feature argument in class method "
                << "RectifyingHomographyTwoSIFTSolver::residual. "
                << "Expected single row feature, but received " << feature.rows
                << " rows.\n";
        throw std::runtime_error(err_msg.str());
    }
    if (type == scale_set_idx)
    {
        auto x = feature.at<double>(0, x_pos);
        auto y = feature.at<double>(0, y_pos);
        auto scale = feature.at<double>(0, s_pos);
        r = scaleResidual(x, y, scale, model);
    }
    else if (type == orient_set_idx)
    {
        auto x = feature.at<double>(0, x_pos);
        auto y = feature.at<double>(0, y_pos);
        auto theta = feature.at<double>(0, t_pos);
        r = orientationResidual(x, y, theta, model);
    }
    else
    {
        std::stringstream err_msg;
        err_msg << "Invalid type argument in class method "
                << "RectifyingHomographyTwoSIFTSolver::residual. "
                << "Expected 0 or 1, but received " << type << std::endl;
        throw std::runtime_error(err_msg.str());
    }
    return r;
}

double RectifyingHomographyTwoSIFTSolver::squaredResidual(
    size_t type, const cv::Mat& feature,
    const SIFTRectifyingHomography& model
) const
{
    const auto r = residual(type, feature, model);
    return r * r;
}

bool RectifyingHomographyTwoSIFTSolver::normalizePoints(
    const DataType& data, // The data points
    const InlierContainerType& inliers,
    MutableDataType& normalized_features, // The normalized features
    NormalizingTransform& normalizing_transform // the normalization transformation model
) const
{
    const auto n_total_inliers = inliers[0].size() + inliers[1].size();
    if (n_total_inliers < 1)
    {
        fprintf(stderr,
            "Feature normalization failed because number of input features is zero.\n"
        );
        return false;
    }
    // compute mean position of features
    normalizing_transform.x0 = 0.0;
    normalizing_transform.y0 = 0.0;
    for (size_t i = 0; i < ResidualDimension::value; i++)
    {
        for (size_t j = 0; j < inliers[i].size(); j++)
        {
            auto inlier_idx = inliers[i][j];
            auto x = data[i]->at<double>(inlier_idx, x_pos);
            auto y = data[i]->at<double>(inlier_idx, y_pos);
            normalizing_transform.x0 += x; // x-coordinate
            normalizing_transform.y0 += y; // y-coordinate
        }
    }
    const auto inv_n = 1.0 / static_cast<double>(n_total_inliers);
    normalizing_transform.x0 *= inv_n;
    normalizing_transform.y0 *= inv_n;
    // compute average Euclidean distance to mean position
    double avg_dist = 0.0;
    for (size_t i = 0; i < ResidualDimension::value; i++)
    {
        for (size_t j = 0; j < inliers[i].size(); j++)
        {
            auto inlier_idx = inliers[i][j];
            auto x = data[i]->at<double>(inlier_idx, x_pos);
            auto y = data[i]->at<double>(inlier_idx, y_pos);
            const auto dx = x - normalizing_transform.x0;
            const auto dy = y - normalizing_transform.y0;
            avg_dist += sqrt(dx * dx + dy * dy);
        }
    }
    avg_dist *= inv_n;
    if (avg_dist < kEpsilon)
    {
        fprintf(
            stderr,
            "Feature normalization failed because all features are located in "
            "the same position (near-zero average distance from mean "
            "position), or because the average distance came out negative. "
            "Average distance: %f\n", avg_dist
        );
        return false;
    }
    // compute scaling factor to transform all feature positions to so that averge
    // distance is sqrt(2).
    normalizing_transform.s = M_SQRT2 / avg_dist;
    // compute normalized features - normalizing is relevant only for coordinates
    // and scale as the scaling of feature positions about the origin is isotropic
    // for (size_t i = 0; i < ResidualDimension::value; i++)
    // {
    //     for (size_t j = 0; j < inliers[i].size(); j++)
    //     {
    //         auto inlier_idx = inliers[i][j];
    //         auto x = data[i]->at<double>(inlier_idx, x_pos);
    //         auto y = data[i]->at<double>(inlier_idx, y_pos);
    //         normalizing_transform.normalize(x, y);
    //         normalized_features[i]->at<double>(j, x_pos) = x;
    //         normalized_features[i]->at<double>(j, y_pos) = y;

    //         if (i == scale_set_idx)
    //         {
    //             auto scale = data[i]->at<double>(inlier_idx, s_pos);
    //             normalizing_transform.normalizeScale(scale);
    //             normalized_features[i]->at<double>(j, s_pos) = scale;
    //         }
    //         else if (i == orient_set_idx)
    //         {
    //             // orientation is not affected by translation and isotropic
    //             // scaling
    //             auto theta = data[i]->at<double>(inlier_idx, t_pos);
    //             normalized_features[i]->at<double>(j, t_pos) = theta;
    //         }
    //         else
    //         {
    //             std::stringstream err_msg;
    //             err_msg << "ERROR in class method "
    //                     << "RectifyingHomographyTwoSIFTSolver::normalizePoints: "
    //                     << "unexpected inlier index " << i << " received.\n";
    //             throw std::runtime_error(err_msg.str());
    //         }
    //     }
    // }

    for (size_t i = 0; i < ResidualDimension::value; i++)
    {
        for (size_t j = 0; j < inliers[i].size(); j++)
        {
            auto inlier_idx = inliers[i][j];
            auto x = data[i]->at<double>(inlier_idx, x_pos);
            auto y = data[i]->at<double>(inlier_idx, y_pos);
            normalized_features[i]->at<double>(j, x_pos) = x;
            normalized_features[i]->at<double>(j, y_pos) = y;

            if (i == scale_set_idx)
            {
                auto scale = data[i]->at<double>(inlier_idx, s_pos);
                normalized_features[i]->at<double>(j, s_pos) = scale;
            }
            else if (i == orient_set_idx)
            {
                auto theta = data[i]->at<double>(inlier_idx, t_pos);
                normalized_features[i]->at<double>(j, t_pos) = theta;
            }
            else
            {
                std::stringstream err_msg;
                err_msg << "ERROR in class method "
                        << "RectifyingHomographyTwoSIFTSolver::normalizePoints: "
                        << "unexpected inlier index " << i << " received.\n";
                throw std::runtime_error(err_msg.str());
            }
        }
    }
    normalizing_transform.x0 = 0.0;
    normalizing_transform.y0 = 0.0;
    normalizing_transform.s = 1.0;

    return true;
}

}
