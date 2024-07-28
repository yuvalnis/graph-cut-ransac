#pragma once

#include <vector>
#include <optional>
#include <cmath>
#include "solver_engine.h"
#include "math_utils.h"

namespace gcransac::estimator::solver
{

// This is the estimator class for estimating a rectifying homography matrix of an image. A model estimation method and error calculation method are implemented
class RectifyingHomographyThreeSIFTSolver : public SolverEngine<ScaleBasedRectifyingHomography>
{
public:
    RectifyingHomographyThreeSIFTSolver() {}
    ~RectifyingHomographyThreeSIFTSolver() {}

    // Determines if there is a chance of returning multiple models
    // the function 'estimateModel' is applied.
    static OLGA_INLINE constexpr bool returnMultipleModels()
    {
        return maximumSolutions() > 1;
    }

    // The maximum number of solutions returned by the estimator
    static OLGA_INLINE constexpr size_t maximumSolutions()
    {
        return 1;
    }
    
    // The minimum number of points required for the estimation
    static OLGA_INLINE constexpr size_t sampleSize()
    {
        return 3;
    }

    // It returns true/false depending on if the solver needs the gravity direction
    // for the model estimation. 
    static OLGA_INLINE constexpr bool needsGravity()
    {
        return false;
    }

    // Estimate the model parameters from the given point sample
    // using weighted fitting if possible.
    bool estimateModel(
        const cv::Mat& data_, // The set of data points
        const size_t *sample_, // The sample used for the estimation
        size_t sample_number_, // The size of the sample
        std::vector<ScaleBasedRectifyingHomography> &models_, // The estimated model parameters
        const double *weights_ = nullptr // The weight for each point
    ) const;

    static double residual(
        const cv::Mat& feature, const ScaleBasedRectifyingHomography& model
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
    static constexpr size_t x_pos = 0; // x-coordinate position
    static constexpr size_t y_pos = 1; // y-coordinate position
    static constexpr size_t s_pos = 2; // scale position

    bool estimateNonMinimalModel(
        const cv::Mat &data_,
        const size_t *sample_,
        size_t sample_number_,
        std::vector<ScaleBasedRectifyingHomography> &models_,
        const double *weights_
    ) const;

    bool estimateMinimalModel(
        const cv::Mat &data_,
        const size_t *sample_,
        size_t sample_number_,
        std::vector<ScaleBasedRectifyingHomography> &models_
    ) const;

};

bool RectifyingHomographyThreeSIFTSolver::estimateMinimalModel(
    const cv::Mat &data_, // The set of data points
    const size_t *sample_, // The sample used for the estimation
    size_t sample_number_, // The size of the sample
   std::vector<ScaleBasedRectifyingHomography> &models_ // The estimated model parameters
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

    const auto *data_ptr = reinterpret_cast<double*>(data_.data);
    Eigen::Matrix<double, 3, 4> coeffs;
    for (size_t i = 0; i < sample_number_; ++i)
    {
        // if the sample indices are not given, they are the first items in the
        // set of data points.
        const size_t idx = sample_ == nullptr ? i : sample_[i];
        // compute position of sample in the set of data points
        const auto *point_ptr = data_ptr + idx * data_.cols;
        // unpack sample into position coordinates and scale
        const auto &x = point_ptr[0];
        const auto &y = point_ptr[1];
        const auto &s = point_ptr[2];

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
    models_.emplace_back(model);
    return true;
}

bool RectifyingHomographyThreeSIFTSolver::estimateNonMinimalModel(
    const cv::Mat &data_,
    const size_t *sample_,
    size_t sample_number_,
    std::vector<ScaleBasedRectifyingHomography> &models_,
    const double *weights_
) const
{
    Eigen::MatrixXd coeffs(sample_number_, 3);
    Eigen::MatrixXd rhs(sample_number_, 1);

    const auto *data_ptr = reinterpret_cast<double*>(data_.data);
    for (size_t i = 0; i < sample_number_; ++i)
    {
        // if the sample indices are not given, they are the first items in the
        // set of data points.
        const size_t idx = sample_ == nullptr ? i : sample_[i];
        // compute position of sample in the set of data points
        const auto *point_ptr = data_ptr + idx * data_.cols;
        // unpack sample into position coordinates and scale
        const auto &x = point_ptr[0];
        const auto &y = point_ptr[1];
        const auto &s = point_ptr[2];

        const auto weight = weights_ == nullptr ? 1.0 : weights_[idx];

        coeffs(i, 0) = weight * x;
        coeffs(i, 1) = weight * y;
        coeffs(i, 2) = -weight * pow(s, kScalePower);
        rhs(i) = -weight;
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
    models_.emplace_back(model);
    return true;
}

bool RectifyingHomographyThreeSIFTSolver::estimateModel(
    const cv::Mat& data_, // The set of data points
    const size_t* sample_, // The sample used for the estimation
    size_t sample_number_, // The size of the sample
    std::vector<ScaleBasedRectifyingHomography>& models_, // The estimated model parameters
    const double* weights_ // The weight for each point
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

double RectifyingHomographyThreeSIFTSolver::residual(
    const cv::Mat& feature,
    const ScaleBasedRectifyingHomography& model
)
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

    return r_scale;
}

bool RectifyingHomographyThreeSIFTSolver::normalizePoints(
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
        normalizing_transform.x0 += feature[0]; // x-coordinate
        normalizing_transform.y0 += feature[1]; // y-coordinate
    }
    const auto inv_n = 1.0 / static_cast<double>(sample_number);
    normalizing_transform.x0 *= inv_n;
    normalizing_transform.y0 *= inv_n;
    // compute average Euclidean distance to mean position
    double avg_dist = 0.0;
    for (size_t i = 0; i < sample_number; i++)
    {
        const auto* feature = get_sample_ptr(i);
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
    for (size_t i = 0; i < sample_number; i++)
    {
        const auto* feature = get_sample_ptr(i);
        auto norm_x = feature[0]; // x-coordinate
        auto norm_y = feature[1]; // y-coordinate
        auto norm_scale = feature[2]; // scale

        normalizing_transform.normalize(norm_x, norm_y);
        normalizing_transform.normalizeScale(norm_scale);

        *norm_features_ptr++ = norm_x;
        *norm_features_ptr++ = norm_y;
        *norm_features_ptr++ = norm_scale;
        // ensures that if the dimension of the features is larger
        // than 4, then the normalization will still succeed.
        for (size_t i = 3; i < normalized_features.cols; ++i)
        {
			*norm_features_ptr++ = feature[i];
        }
    }

    return true;
}

}
