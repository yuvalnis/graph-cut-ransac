#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <cmath>
#include <random>
#include <vector>
#include <set>
#include <numeric>
#include <unordered_map>
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
public:

    using Base = Estimator<Solver>;
    using Model = typename Base::Model;
    using ResidualDimension = typename Base::ResidualDimension;
    using InlierContainerType = typename Base::InlierContainerType;
    using ResidualType = typename Base::ResidualType;
    using WeightType = typename Base::WeightType;
    using SampleSizeType = typename Base::SampleSizeType;
    using DataType = typename Base::DataType;
    using MutableDataType = typename Base::MutableDataType;

protected:

    const std::shared_ptr<Solver> solver;

private:

    const SampleSizeType m_inliers_limit;

    static inline SampleSizeType computeInlierLimit(
        const SampleSizeType& sample_sizes
    )
    {
        SampleSizeType result{};
        for (size_t i = 0; i < ResidualDimension::value; i++)
        {
            result[i] = 7 * sample_sizes[i]; 
        }
        return result;
    }

    static void getInlierWeights(
        const WeightType& weights,
        const InlierContainerType& inliers,
        WeightType& inlier_weights
    )
    {
        for (size_t i = 0; i < ResidualDimension::value; i++)
        {
            const auto& weights_i = weights[i];
            if (weights_i.empty())
            {
                inlier_weights[i] = {};
                continue;
            }
            const auto& inliers_i = inliers[i];
            auto& inlier_weights_i = inlier_weights[i];
            inlier_weights_i.clear();
            inlier_weights_i.reserve(inliers_i.size());
            for (size_t j = 0; j < inliers_i.size(); j++)
            {
                const auto idx = inliers_i.empty() ? j : inliers_i[j];
                inlier_weights_i.push_back(weights_i[idx]);
            }
        }
    }

public:

    RectifyingHomographyEstimator() :
        solver(std::make_shared<Solver>()),
        m_inliers_limit(computeInlierLimit(sampleSize()))
    {}
    
    ~RectifyingHomographyEstimator() {}

    inline SampleSizeType sampleSize() const
    {
        return solver->sampleSize();
    }

    // The size of a minimal sample_ required for the estimation
    inline static constexpr size_t maximumMinimalSolutions()
    {
        return Solver::maximumSolutions();
    }

    // The size of a sample when doing inner RANSAC on a non-minimal sample
    inline const SampleSizeType& inlierLimit() const 
    {
        return m_inliers_limit;
    }

    // Given a model and a data point, calculate the error. Users should implement
	// this function appropriately for the task being solved.
	inline double residual(
        size_t type, const cv::Mat& feature, const Model& model
    ) const
	{
		return solver->residual(type, feature, model);
	}

	inline double squaredResidual(
        size_t type, const cv::Mat& feature, const Model& model
    ) const
	{
		return solver->squaredResidual(type, feature, model);
	}

    inline bool isValidSample(
		const DataType& data,
		const InlierContainerType& inliers
	) const override
	{
		return solver->isValidSample(data, inliers);
	}

    inline bool isValidModel([[maybe_unused]] const Model& model) const override
	{
		return true;
	}

	// Enable a quick check to see if the model is valid. This can be a geometric
	// check or some other verification of the model structure.
	inline bool isValidModel(
		[[maybe_unused]] Model& model,
		[[maybe_unused]] const DataType& data,
		[[maybe_unused]] const InlierContainerType& inliers,
		[[maybe_unused]] const InlierContainerType& minimal_sample,
		[[maybe_unused]] const ResidualType& threshold,
		[[maybe_unused]] bool& model_updated
	) const override
	{
		return solver->isValidModel(model, data, inliers, minimal_sample,
                                    threshold, model_updated);
	}

    // Estimating the model from a minimal sample
    inline bool estimateModel(
        const DataType& data,
		const InlierContainerType& inliers,
		std::vector<Model>& models
    ) const
    {
        return solver->estimateModel(data, inliers, models);
    }

    // Estimating the model from a non-minimal sample
    bool estimateModelNonminimal(
        const DataType& data,
		const InlierContainerType& inliers,
		std::vector<Model>& models,
		const WeightType& weights = WeightType{}
    ) const
    {
        MutableDataType normalized_data{};
        InlierContainerType inliers_of_normed_data{};
        for (size_t i = 0; i < ResidualDimension::value; i++)
        {
            if (inliers[i].size() < sampleSize()[i])
            {
                return false;
            }
            // initialize container for normalized data
            normalized_data[i] = std::make_unique<cv::Mat>(
                inliers[i].size(), data[i]->cols, data[i]->type()
            );
            // initialize indices of normalized data
            for (size_t j = 0; j < inliers[i].size(); j++)
            {
                inliers_of_normed_data[i].push_back(j);
            }
        }
        NormalizingTransform normalizing_transform;
        bool success = solver->normalizePoints(
            data, inliers, normalized_data,
            normalizing_transform
        );
        if (!success)
        {
            return false;
        }
        // only inlier features are used to estimate model so the same must
        // be done for the weights.
        WeightType inlier_weights;
        getInlierWeights(
            weights, inliers, inlier_weights
        );
        // sample_ = nullptr because normalized features and weights are now
        // made up only of inlier features and weights.
        DataType const_norm_data{};
        for (size_t i = 0; i < ResidualDimension::value; i++)
        {
            const_norm_data[i] = std::unique_ptr<const cv::Mat>(
                std::move(normalized_data[i])
            );
        }
        success = solver->estimateModel(
            const_norm_data, inliers_of_normed_data, models, inlier_weights
        );
        if (!success)
        {
            return false;
        }
        for (auto& model : models)
        {
            model.x0 = normalizing_transform.x0;
            model.y0 = normalizing_transform.y0; 
            model.s = normalizing_transform.s;
        }
        return true;
    }
};

}
