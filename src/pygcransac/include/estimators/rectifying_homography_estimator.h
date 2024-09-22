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
    using ResidualType = typename Base::ResidualType;
    using InlierContainerType = typename Base::InlierContainerType;
    using WeightType = typename Base::WeightType;
    using SampleSizeType = typename Base::SampleSizeType;

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
	inline ResidualType residual(const cv::Mat& data, const Model& model) const
	{
		return solver->residual(data, model);
	}

	inline ResidualType squaredResidual(const cv::Mat& data, const Model& model) const
	{
		return solver->squaredResidual(data, model);
	}

    inline bool isValidSample(
		const cv::Mat& data,
		const InlierContainerType& inliers
	) const override
	{
		return solver->isValidSample(data, inliers);
	}

    // Estimating the model from a minimal sample
    inline bool estimateModel(
        const cv::Mat& data,
		const InlierContainerType& inliers,
		std::vector<Model>& models
    ) const
    {
        return solver->estimateModel(data, inliers, models);
    }

    static void computeNormalizedDataInliers(
        const InlierContainerType& inliers,
        InlierContainerType& inliers_of_normed_data,
        std::vector<size_t>& unique_indices_vector
    )
    {
        std::set<int> unique_indices;
        std::unordered_map<int, int> index_map;  // Maps old indices to new indices
        // Collect all unique indices from inliers
        for (const auto& inlier_list : inliers)
        {
            unique_indices.insert(inlier_list.begin(), inlier_list.end());
        }
        // Convert the set to a vector to maintain sorted order
        unique_indices_vector.assign(unique_indices.begin(), unique_indices.end());
        // Populate normed_data and create the index map
        size_t new_index = 0;
        for (const auto& idx : unique_indices) {
            index_map[idx] = new_index;
            new_index++;
        }
        // Create the new inliers_of_normed_data
        for (size_t i = 0; i < ResidualDimension::value; i++)
        {
            for (const auto& idx : inliers[i])
            {
                inliers_of_normed_data[i].push_back(index_map[idx]);
            }
        }
    }

    // Estimating the model from a non-minimal sample
    bool estimateModelNonminimal(
        const cv::Mat& data,
		const InlierContainerType& inliers,
		std::vector<Model>& models,
		const WeightType& weights = WeightType{}
    ) const
    {
        for (size_t i = 0; i < ResidualDimension::value; i++)
        {
            if (inliers[i].size() < sampleSize()[i])
            {
                return false;
            }
        }
        // since all data types are joined in the same data set, we compute
        // the indices which are inliers in any of the sub-types.
        std::vector<size_t> inlier_union{};
        InlierContainerType inliers_of_normed_data{};
        computeNormalizedDataInliers(
            inliers, inliers_of_normed_data, inlier_union
        );
        // normalize features
        cv::Mat normalized_features(
            inlier_union.size(), data.cols, data.type()
        );
        NormalizingTransform normalizing_transform;
        bool success = solver->normalizePoints(
            data, inlier_union, normalized_features,
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
        success = solver->estimateModel(
            normalized_features, inliers_of_normed_data, models, inlier_weights
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
