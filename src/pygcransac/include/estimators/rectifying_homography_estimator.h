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
template<class Solver>
class RectifyingHomographyEstimator : public Estimator<Solver>
{
protected:
    const std::shared_ptr<Solver> solver;

public:
    using Base = Estimator<Solver>;
    using Model = typename Base::Model;
    using ResidualDimension = typename Base::ResidualDimension;
    using InlierContainerType = typename Base::InlierContainerType;
    using SampleSizesType = typename Base::SampleSizesType;
    using WeightType = typename Base::WeightType;

    RectifyingHomographyEstimator() : solver(std::make_shared<Solver>()) {}
    
    ~RectifyingHomographyEstimator() {}

    OLGA_INLINE static constexpr size_t nonMinimalSampleSize()
    {
        return Solver::sampleSize();
    }

    OLGA_INLINE static constexpr size_t sampleSize()
    {
        return Solver::sampleSize();
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
        const cv::Mat& data,
		const InlierContainerType& inliers,
		std::vector<Model>& model
    ) const
    {
        return solver->estimateModel(
            data, inliers, sampleSize(), *models_, nullptr
        );
    }

    // Estimating the model from a non-minimal sample
    OLGA_INLINE bool estimateModelNonminimal(
        const cv::Mat& data,
		const InlierContainerType& inliers,
		std::vector<Model>& model,
		const WeightType& weights
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
        bool success = solver->normalizePoints(
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
        solver->getInlierWeights(
            sample_, sample_number_, weights_, inlier_weights
        );
        // sample_ = nullptr because normalized features and wieights are now
        // made up only of inlier features and weights.
        success = solver->estimateModel(
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
};

}
