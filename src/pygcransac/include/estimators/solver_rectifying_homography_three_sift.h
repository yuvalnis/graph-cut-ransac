#pragma once

#include "solver_engine.h"
#include "homography_estimator.h"
#include "math_utils.h"

namespace gcransac::estimator::solver
{
// This is the estimator class for estimating a rectifying homography matrix of an image. A model estimation method and error calculation method are implemented
class RectifyingHomographyThreeSIFTSolver : public SolverEngine
{
public:
    RectifyingHomographyThreeSIFTSolver()
    {
    }

    ~RectifyingHomographyThreeSIFTSolver()
    {
    }

    // Determines if there is a chance of returning multiple models
    // the function 'estimateModel' is applied.
    static constexpr bool returnMultipleModels()
    {
        return maximumSolutions() > 1;
    }

    // The maximum number of solutions returned by the estimator
    static constexpr size_t maximumSolutions()
    {
        return 1;
    }
    
    // The minimum number of points required for the estimation
    static constexpr size_t sampleSize()
    {
        return 3;
    }

    // It returns true/false depending on if the solver needs the gravity direction
    // for the model estimation. 
    static constexpr bool needsGravity()
    {
        return false;
    }

    // Estimate the model parameters from the given point sample
    // using weighted fitting if possible.
    OLGA_INLINE bool estimateModel(
        const cv::Mat& data_, // The set of data points
        const size_t *sample_, // The sample used for the estimation
        size_t sample_number_, // The size of the sample
        std::vector<Model> &models_, // The estimated model parameters
        const double *weights_ = nullptr // The weight for each point
    ) const;

protected:
    OLGA_INLINE bool estimateNonMinimalModel(
        const cv::Mat &data_,
        const size_t *sample_,
        size_t sample_number_,
        std::vector<Model> &models_,
        const double *weights_
    ) const;

    OLGA_INLINE bool estimateMinimalModel(
        const cv::Mat &data_,
        const size_t *sample_,
        size_t sample_number_,
        std::vector<Model> &models_,
        const double *weights_
    ) const;

}; 

OLGA_INLINE bool RectifyingHomographyThreeSIFTSolver::estimateMinimalModel(
    const cv::Mat &data_, // The set of data points
    const size_t *sample_, // The sample used for the estimation
    size_t sample_number_, // The size of the sample
    std::vector<Model> &models_, // The estimated model parameters
    const double *weights_ // The weight for each point
) const
{
    constexpr size_t kEquationsPerSample = 1; // how many equations does each sample give
    constexpr size_t kRowNumber = kEquationsPerSample * sampleSize();
    constexpr double kScalePower = -1.0 / 3.0;
    constexpr double Eps = 1e-9;
    const size_t kColNumber = data_.cols + 1;
    Eigen::MatrixXd coeffs(kRowNumber, kColNumber);

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
        coeffs(i, 3) = 1.0;
    }
    Eigen::Matrix<double, 3, 1> x;
    gcransac::utils::gaussElimination<3>(coeffs, x);
    if (x.hasNaN())
    {
        return false;
    }
    // construct homography from x
    double h7 = x(0);
    double h8 = x(1);
    const double alpha = x(2);
    if (std::abs(alpha) > Eps)
    {
        h7 /= alpha;
        h8 /= alpha;
    }
    Homography model;
    model.descriptor << 1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        h7,  h8,  1.0;
    models_.emplace_back(model);
    return true;
}

OLGA_INLINE bool RectifyingHomographyThreeSIFTSolver::estimateNonMinimalModel(
    const cv::Mat &data_,
    const size_t *sample_,
    size_t sample_number_,
    std::vector<Model> &models_,
    const double *weights_
) const
{

}

OLGA_INLINE bool RectifyingHomographyThreeSIFTSolver::estimateModel(
    const cv::Mat& data_, // The set of data points
    const size_t *sample_, // The sample used for the estimation
    size_t sample_number_, // The size of the sample
    std::vector<Model> &models_, // The estimated model parameters
    const double *weights_ // The weight for each point
) const
{
    if (sample_number_ < sampleSize())
    {
        fprintf(stderr, "There were not enough affine correspondences provided for the solver (%d < %d).\n", sample_number_, sampleSize());
        return false;
    }
    if (sample_number_ == sampleSize())
    {
        return estimateMinimalModel(data_, sample_, sample_number_, models_, weights_);
    }
    return estimateNonMinimalModel(data_, sample_, sample_number_, models_, weights_);
}

}
