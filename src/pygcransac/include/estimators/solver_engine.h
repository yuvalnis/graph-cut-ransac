// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <array>
#include <vector>
#include <opencv2/core/core.hpp>

namespace gcransac::estimator::solver
{

template <class Model_t, size_t ResidualDimension_t>
class SolverEngine
{
public:
	using Model = Model_t;
	using ResidualDimension = std::integral_constant<size_t, ResidualDimension_t>;
	using InlierContainerType = std::array<std::vector<size_t>, ResidualDimension_t>;
	using ResidualType = Eigen::Array<double, ResidualDimension_t, 1>;
	using WeightType = std::array<std::vector<double>, ResidualDimension_t>;
	using SampleSizeType = std::array<size_t, ResidualDimension_t>;

	SolverEngine() {}
	virtual ~SolverEngine() {}

	// Determines if there is a chance of returning multiple models
	// the function 'estimateModel' is applied.
	inline static constexpr bool returnMultipleModels()
	{ 
		return maximumSolutions() > 1;
	}

	// The maximum number of solutions returned by the estimator
	inline static constexpr size_t maximumSolutions() { return 1; }

	// The minimum number of points required for the estimation
	inline virtual SampleSizeType sampleSize() const = 0;

	inline virtual bool isValidSample(
		[[maybe_unused]] const cv::Mat& data,
		[[maybe_unused]] const InlierContainerType& inliers
	) const
	{
		return true;
	}

	// Estimate the model parameters from the given point sample
	// using weighted fitting if possible.
	virtual bool estimateModel(
		const cv::Mat& data,
		const InlierContainerType& inliers,
		std::vector<Model>& models,
		const WeightType& weights = WeightType{}
	) const = 0;

	virtual ResidualType residual(const cv::Mat& feature, const Model& model) const = 0;
	virtual ResidualType squaredResidual(const cv::Mat& feature, const Model& model) const = 0;
};

}
