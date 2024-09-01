#pragma once

#include "score.hpp"

// forward declaration
namespace cv
{
	class Mat;
}

namespace gcransac
{

template<class ModelEstimator>
class ScoringFunction
{
public:
	using Estimator = ModelEstimator;
    using NumInlierTypes = typename ModelEstimator::ResidualDimension;
    using ScoreType = Score<NumInlierTypes::value>;
	using Model = typename Estimator::Model;
	using InlierContainerType = typename Estimator::InlierContainerType;
	using ThresholdType = typename Estimator::ResidualType;

	virtual ~ScoringFunction() = 0;

	virtual ScoreType getScore(
		const cv::Mat &data, // The input data points
		const Model &model, // The current model parameters
		const Estimator &estimator, // The model estimator
		const ThresholdType& thresholds,
		InlierContainerType& inliers, // The selected inliers
		const ScoreType& best_score = ScoreType(), // The score of the current so-far-the-best model
		const std::vector<const std::vector<size_t>*>* index_sets = nullptr // Index sets to be verified
	) const = 0;
};

// Definition of the pure virtual destructor
template <typename Estimator>
ScoringFunction<Estimator>::~ScoringFunction() {
    // Usually, this is empty, but it's required to have a definition
}

}
