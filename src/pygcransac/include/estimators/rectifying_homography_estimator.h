#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <cmath>
#include <random>
#include <vector>
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
    class _NonMinimalSolverEngine, // The solver used for estimating the model from a non-minimal sample
    class _ModelType,
    typename _ResidualType = double
> class RectifyingHomographyEstimator : public Estimator<cv::Mat, _ModelType, _ResidualType>
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

    // Estimating the model from a minimal sample
    OLGA_INLINE bool estimateModel(
        const cv::Mat& data_, // The data points
        const size_t* sample_, // The sample usd for the estimation
        std::vector<_ModelType>* models_ // The estimated model parameters
    ) const
    {
        return minimal_solver->estimateModel(
            data_, sample_, sampleSize(), *models_, nullptr
        );
    }

    // Estimating the model from a non-minimal sample
    OLGA_INLINE bool estimateModelNonminimal(
        const cv::Mat& data_, // The data points
        const size_t* sample_, // The sample used for the estimation
        const size_t& sample_number_, // The size of a minimal sample
        std::vector<_ModelType>* models_,
        const double *weights_ = nullptr // The estimated model parameters
    ) const
    {
        if (sample_number_ < nonMinimalSampleSize())
        {
            return false;
        }
        // normalize features
        cv::Mat normalized_features(
            sample_number_, data_.cols, data_.type()
        );
        NormalizingTransform normalizing_transform;
        bool success = non_minimal_solver->normalizePoints(
            data_, sample_, sample_number_, normalized_features,
            normalizing_transform
        );
        if (!success)
        {
            return false;
        }
        // only inlier features are used to estimate model so the same must
        // be done for the weights.
        std::vector<double> inlier_weights;
        non_minimal_solver->getInlierWeights(
            sample_, sample_number_, weights_, inlier_weights
        );
        // sample_ = nullptr because normalized features and wieights are now
        // made up only of inlier features and weights.
        success = non_minimal_solver->estimateModel(
            normalized_features, nullptr, sample_number_, *models_,
            inlier_weights.data()
        );
        if (!success)
        {
            return false;
        }
        for (auto& model : *models_)
        {
            model.x0 = normalizing_transform.x0;
            model.y0 = normalizing_transform.y0; 
            model.s = normalizing_transform.s;
        }
        return true;
    }
 
    OLGA_INLINE _ResidualType residual(
        const cv::Mat& feature_,
        const _ModelType& model_
    ) const
    {
        return _MinimalSolverEngine::residual(feature_, model_);
    }

    OLGA_INLINE _ResidualType squaredResidual(
        const cv::Mat& feature_,
        const _ModelType& model_
    ) const
    {
        auto r = residual(feature_, model_);
        if constexpr (std::is_same_v<_ResidualType, double>)
		{
            r = r * r;
        }
        else
        {
            r = r.cwiseProduct(r);
        }
        return r;
    }
};

}
