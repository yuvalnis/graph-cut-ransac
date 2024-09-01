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

	void updateScoreWithFeature(
		const cv::Mat& data,
		const size_t& point_idx,
		const Model& model,
		const ModelEstimator& estimator,
		const ThresholdType& sqr_truncated_thresholds,
		InlierContainerType& inliers,
		ScoreType& score
	) const
	{
		const auto sqr_residuals = estimator.squaredResidual(
			data.row(point_idx), model
		);
		const Eigen::Array<bool, Eigen::Dynamic, 1> comparison =
			sqr_residuals.array() <= sqr_truncated_thresholds.array();

		for (size_t i = 0; i < comparison.size(); i++)
		{
			if (comparison(i))
			{
				inliers[i].emplace_back(point_idx);
				auto& inlier_num = score.num_inliers_by_type(i);
				inlier_num++;
				auto& score_val = score.value_by_type(i);
				// Increase the score.
				// The original truncated quadratic loss is as follows: 
				// score = 1 - residual^2 / threshold^2.
				// For RANSAC, -residual^2 is enough:
				// It can been re-arranged as
				// score = 1 - residual^2 / threshold^2				->
				// score * threshold^2 = threshold^2 - residual^2		->
				// score * threshold^2 - threshold^2 = - residual^2
				// This is faster to calculate and it is normalized back afterwards.
				score_val -= sqr_residuals(i);
			}
		}
	}

	ScoreType getScore(
		const cv::Mat &data, // The input data points
		const Model &model, // The current model parameters
		const ModelEstimator &estimator, // The model estimator
		const ThresholdType& thresholds,
		InlierContainerType& inliers, // The selected inliers
		const ScoreType& best_score = ScoreType(), // The score of the current so-far-the-best model
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
			const auto point_number = static_cast<size_t>(data.rows);
			// Iterate through all points, calculate the sqr_residuals and store the points as inliers if needed.
			for (size_t point_idx = 0; point_idx < point_number; point_idx ++)
			{
				updateScoreWithFeature(
					data, point_idx, model, estimator,
					sqr_truncated_thresholds, inliers, score
				);
			}
		}
		else
		{
			// Iterating through the index sets
			for (const auto &current_set : *index_sets)
			{
				// Iterating through the point indices in the current set
				for (const auto point_idx : *current_set)
				{
					updateScoreWithFeature(
						data, point_idx, model, estimator,
						sqr_truncated_thresholds, inliers, score
					);
				}
			}
		}

		// return score values to MSAC ones and sum them up to get final score
		for (size_t i = 0; i < NumInlierTypes::value; i++)
		{
			const auto& n_inliers = score.num_inliers_by_type(i);
			if (n_inliers > 0)
			{
				auto& score_i = score.value_by_type(i);
				score_i = (score_i / sqr_truncated_thresholds(i)) + n_inliers;
			}
		}

		score.finalize();

		return score;
	}

};

}
