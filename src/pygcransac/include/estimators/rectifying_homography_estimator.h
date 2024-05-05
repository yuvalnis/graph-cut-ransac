#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <cmath>
#include <random>
#include <vector>
#include <sstream>

#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

#include "estimator.h"
#include "model.h"


namespace gcransac::estimator
{

// This is the estimator class for estimating a rectifying homography matrix
// between a warped image and its rectified counterpart. A model estimation 
// method and error calculation method are implemented.
template<
    class _MinimalSolverEngine, // The solver used for estimating the model from a minimal sample
    class _NonMinimalSolverEngine // The solver used for estimating the model from a non-minimal sample
> class RectifyingHomographyEstimator : public Estimator<cv::Mat, Model>
{
protected:
    // Minimal solver engine used for estimating a model from a minimal sample
    const std::shared_ptr<_MinimalSolverEngine> minimal_solver;

    // Non-minimal solver engine used for estimating a model from a bigger than minimal sample
    const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

public:

    RectifyingHomographyEstimator() :
        minimal_solver(std::make_shared<_MinimalSolverEngine>()),
        non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>())
    {}
    
    ~RectifyingHomographyEstimator() {}

    OLGA_INLINE const _MinimalSolverEngine* getMinimalSolver() const
    {
        return minimal_solver.get();
    }

    OLGA_INLINE _MinimalSolverEngine* getMutableMinimalSolver()
    {
        return minimal_solver.get();
    }

    OLGA_INLINE const _NonMinimalSolverEngine* getNonMinimalSolver() const
    {
        return non_minimal_solver.get();
    }

    OLGA_INLINE _NonMinimalSolverEngine* getMutableNonMinimalSolver()
    {
        return non_minimal_solver.get();
    }

    OLGA_INLINE static constexpr size_t nonMinimalSampleSize()
    {
        return _NonMinimalSolverEngine::sampleSize();
    }

    OLGA_INLINE static constexpr size_t sampleSize()
    {
        return _MinimalSolverEngine::sampleSize();
    }

    // A flag deciding if the points can be weighted when the non-minimal fitting is applied 
    OLGA_INLINE static constexpr bool isWeightingApplicable()
    {
        return true;
    }

    // The size of a minimal sample_ required for the estimation
    OLGA_INLINE static constexpr size_t maximumMinimalSolutions()
    {
        return _MinimalSolverEngine::maximumSolutions();
    }

    // The size of a sample when doing inner RANSAC on a non-minimal sample
    OLGA_INLINE size_t inlierLimit() const
    {
        return 7 * sampleSize();
    }

    // static std::pair<double, double> _mean(
    //     const cv::Mat& data,
    //     const size_t* sample,
    //     const size_t& sample_number
    // ) const
    // {
    //     if (sample_number < 1)
    //     {
    //         return {0.0, 0.0};
    //     }

    //     const auto* data_ptr = reinterpret_cast<double*>(data.data);
    //     // initialize coordinas of sample center-of-mass
    //     double mean_x = 0.0;
    //     double mean_y = 0.0;
    //     for (size_t i = 0; i < sample_number; ++i)
    //     {
    //         // if the sample indices are not given, they are the first items in the
    //         // set of data points.
    //         const size_t idx = sample == nullptr ? i : sample[i];
    //         // compute position of sample in the set of data points
    //         const auto *feature_ptr = data_ptr + idx * data.cols;
    //         // unpack coordinates of feature sample
    //         const auto &x = feature_ptr[0];
    //         const auto &y = feature_ptr[1];

    //         mean_x += x;
    //         mean_y += y;
    //     }
        
    //     mean_x /= static_cast<double>(sample_number);
    //     mean_y /= static_cast<double>(sample_number);

    //     return {mean_x, mean_y}
    // }

    // double _meanDistFromPoint(
    //     const cv::Mat& data,
    //     const size_t* sample,
    //     const size_t& sample_number,
    //     const double point_x,
    //     const double point_y
    // ) const
    // {
    //     if (sample_number < 1)
    //     {
    //         return 0.0;
    //     }

    //     double mean_dist = 0.0;

    //     for (size_t i = 0; i < sample_number; ++i)
    //     {
    //         // if the sample indices are not given, they are the first items in the
    //         // set of data points.
    //         const size_t idx = sample == nullptr ? i : sample[i];
    //         // compute position of sample in the set of data points
    //         const auto *feature_ptr = data_ptr + idx * data.cols;
    //         // unpack coordinates of feature sample
    //         const auto &x = feature_ptr[0];
    //         const auto &y = feature_ptr[1];

    //         const auto dx = point_x - x;
    //         const auto dy = point_y - y;

    //         mean_dist += sqrt(dx * dx + dy * dy); 
    //     }

    //     mean_dist /= static_cast<double>(sample_number);

    //     return mean_dist;
    // }

    // void _normalizeData(
    //     const cv::Mat& data,
    //     const size_t* sample,
    //     const size_t& sample_number,
    //     cv::Mat& normalize_data,
    //     Eigen::Matrix3d& transform
    // ) const
    // {
    //     const auto [mean_x, mean_y] = _mean(data, sample, sample_number);
    //     const auto mean_dist = _meanDistFromPoint(
    //         data, sample, sample_number, mean_x, mean_y
    //     );

    //     const auto data_scaling_factor = (sample_number < 1) ? 1.0 : (M_SQRT2 / mean_dist);
    //     // compute normalized coordinates and scale
    //     double* normalized_data_ptr = reinterpret_cast<double*>(normalize_data.data);
    //     for (size_t i = 0; i < sample_number; ++i)
    //     {
    //         // if the sample indices are not given, they are the first items in the
    //         // set of data points.
    //         const size_t idx = sample == nullptr ? i : sample[i];
    //         // compute position of sample in the set of data points
    //         const auto *feature_ptr = data_ptr + idx * data.cols;
    //         // unpack coordinates and scale of feature sample
    //         const auto &x = feature_ptr[0];
    //         const auto &y = feature_ptr[1];
    //         const auto &s = feature_ptr[2];

    //         *normalized_data_ptr++ = (x - mean_x) * data_scaling_factor;
    //         *normalized_data_ptr++ = (y - mean_y) * data_scaling_factor;
    //         *normalized_data_ptr++ = s * data_scaling_factor;
    //         // make sure normalized data pointer is advanced to the start of
    //         // a new row. Unhandled parameters are left as is.
    //         for (size_t i = 3; i < normalized_points_.cols; ++i)
    //         {
	// 		    *normalized_points_ptr++ = *(feature_ptr + i);
    //         }
    //     }
    //     // create the normalizing transformation
    //     const auto s = data_scaling_factor;
    //     const auto tx = -mean_x;
    //     const auto ty = -mean_y;
    //     transform << s  , 0.0, s * tx,
    //                  0.0, s  , s * ty,
    //                  0.0, 0.0, 1.0;
    // }

    // Estimating the model from a minimal sample
    OLGA_INLINE bool estimateModel(
        const cv::Mat& data_, // The data points
        const size_t *sample_, // The sample usd for the estimation
        std::vector<Model>* models_ // The estimated model parameters
    ) const
    {
        return minimal_solver->estimateModel(
            data_,
            sample_,
            sampleSize(),
            *models_
        );
    }

    // Estimating the model from a non-minimal sample
    OLGA_INLINE bool estimateModelNonminimal(
        const cv::Mat& data_, // The data points
        const size_t *sample_, // The sample used for the estimation
        const size_t &sample_number_, // The size of a minimal sample
        std::vector<Model>* models_,
        const double *weights_ = nullptr // The estimated model parameters
    ) const
    {
        if (sample_number_ < nonMinimalSampleSize())
        {
            return false;
        }
        return non_minimal_solver->estimateModel(
            data_, nullptr, sample_number_, *models_, weights_
        );
    }

    double residual(const cv::Mat& feature_, const Eigen::MatrixXd& descriptor_) const
    {
        constexpr double kMinDeterminant = 1e-08;
        constexpr double alpha = 1.0; // for now we assume alpha can only take on this fixed value
        if (descriptor_.rows() != 3 || descriptor_.cols() != 3)
        {
            std::stringstream error_msg;
            error_msg << "A descriptor of a rectifying homography has to be a "
                      << "3x3 matrix.\n The input descriptor is a "
                      << descriptor_.rows() << "x" << descriptor_.cols()
                      << " matrix:\n" << descriptor_ << "\n";
            throw std::runtime_error(error_msg.str());
        }
        
        if (std::abs(descriptor_.determinant()) < kMinDeterminant)
        {
            throw std::runtime_error("The descriptor of a rectifying homography should be invertible");
        }

        const auto* feature_ptr = reinterpret_cast<double*>(feature_.data);
        const auto& x = feature_ptr[0];
        const auto& y = feature_ptr[1];
        const auto& s = feature_ptr[2];

        const Eigen::Vector3d orig_p(x, y, 1.0);
        const auto rectified_p = descriptor_.inverse() * orig_p;

        if (std::abs(rectified_p.z()) < std::numeric_limits<double>::epsilon())
        {
            throw std::runtime_error("The given descriptor maps the feature point to infinity in the rectified image");
        }

        const auto rectified_x = rectified_p.x() / rectified_p.z();
        const auto rectified_y = rectified_p.y() / rectified_p.z();
        const auto h7 = descriptor_(2, 0);
        const auto h8 = descriptor_(2, 1);

        const auto reprojected_s =
            std::pow(alpha / (h7 * rectified_x + h8 * rectified_y + 1.0), 3.0);

        return 1.0 - (s / reprojected_s);
    }

    OLGA_INLINE double residual(const cv::Mat& feature_, const Model& model_) const
    {
        return residual(feature_, model_.descriptor);
    }

    OLGA_INLINE double squaredResidual(const cv::Mat& feature_,
                                       const Eigen::MatrixXd& descriptor_) const
    {
        const auto r = residual(feature_, descriptor_);
        return r * r;
    }

    OLGA_INLINE double squaredResidual(const cv::Mat& feature_,
                                      const Model& model_) const
    {
        const auto r = residual(feature_, model_.descriptor);
        return r * r;
    }
};

}
