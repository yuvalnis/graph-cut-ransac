#pragma once

#include "scoring_function.hpp"
#include <opencv2/core/mat.hpp>

namespace gcransac
{

template<class ModelEstimator>
class MSACScoringFunction : public ScoringFunction<ModelEstimator>
{
public:

	using Base = ScoringFunction<ModelEstimator>;
    using NumInlierTypes = typename Base::NumInlierTypes;
    using ScoreType = Score<NumInlierTypes::value>;
	using Model = typename Base::Model;
	using InlierContainerType = typename Base::InlierContainerType;
	using ThresholdType = typename Base::ThresholdType;
	using DataType = typename Base::DataType;

	void updateScoreWithFeature(
		size_t type,
		const std::unique_ptr<const cv::Mat>& data,
		size_t point_idx,
		const Model& model,
		const ModelEstimator& estimator,
		double sqr_truncated_threshold,
		std::vector<size_t>& inliers,
		ScoreType& score
	) const
	{
		const auto sqr_residual = estimator.squaredResidual(
			type, data->row(point_idx), model
		);
		if (sqr_residual <= sqr_truncated_threshold)
		{
			inliers.emplace_back(point_idx);
			score.increment_inlier_num(type);
			// Increase the score.
			// The original truncated quadratic loss is as follows: 
			// score = 1 - residual^2 / threshold^2.
			// For RANSAC, -residual^2 is enough:
			// It can been re-arranged as
			// score = 1 - residual^2 / threshold^2				->
			// score * threshold^2 = threshold^2 - residual^2	->
			// score * threshold^2 - threshold^2 = - residual^2
			// This is faster to calculate and it is normalized back afterwards.
			score.increment_value(type, -sqr_residual);
		}
	}

	ScoreType getScore(
		const DataType &data, // The input data points
		const Model &model, // The current model parameters
		const ModelEstimator &estimator, // The model estimator
		const ThresholdType& thresholds,
		InlierContainerType& inliers, // The selected inliers
		const std::vector<const std::vector<size_t>*>* index_sets = nullptr // Index sets to be verified
	) const
	{
		ScoreType score{};

		const auto sqr_truncated_thresholds = 2.25 * thresholds * thresholds;
		for (auto& inlier_set : inliers)
		{
			inlier_set.clear();
		}

		// If the points are not prefiltered into index sets, iterate through all of them.
		if (index_sets == nullptr)
		{
			for (size_t i = 0; i < NumInlierTypes::value; i++)
			{
				auto point_number = static_cast<size_t>(data[i]->rows);
				// Iterate through all points, calculate their squared residuals
				// and store the points as inliers if needed.
				for (size_t point_idx = 0; point_idx < point_number; point_idx++)
				{
					updateScoreWithFeature(
						i, data[i], point_idx, model, estimator,
						sqr_truncated_thresholds(i), inliers[i], score
					);
				}
			}
		}
		else
		{
			throw std::runtime_error(
				"Reached unimplemented code section in "
				"MSACScoringFunction::getScore method with non-null "
				"index-sets.\n"
			);
			// Iterating through the index sets
			// for (const auto &current_set : *index_sets)
			// {
			// 	// Iterating through the point indices in the current set
			// 	for (const auto point_idx : *current_set)
			// 	{
			// 		updateScoreWithFeature(
			// 			data, point_idx, model, estimator,
			// 			sqr_truncated_thresholds, inliers, score
			// 		);
			// 	}
			// }
		}

		// return score values to MSAC ones and sum them up to get final score
		const auto sample_sizes = estimator.sampleSize();
		for (size_t i = 0; i < NumInlierTypes::value; i++)
		{
			const auto& n_inliers = score.num_inliers_by_type(i);
			if (n_inliers < sample_sizes[i])
			{
				// if score involves less inliers than minimal model requires,
				// reset it to worst score (zero).
				score = ScoreType{};
				break;
			}
			else
			{
				const auto normed_score = score.value_by_type(i) /
										  sqr_truncated_thresholds(i);
				const auto msac_score = normed_score + static_cast<double>(n_inliers);
				score.reset_value(i, msac_score);
			}
		}
		
		return score;
	}

};

}
