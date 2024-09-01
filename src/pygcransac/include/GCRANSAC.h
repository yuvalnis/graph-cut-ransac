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

#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <random>
#include "GCoptimization.h"
#include "model.h"
#include "settings.h"
#include "statistics.h"
#include "scoring_functions/MSAC_scoring_function.hpp"
#include "preemption/preemption_empty.h"
#include "inlier_selectors/empty_inlier_selector.h"

namespace gcransac
{

bool get_random_subset(
	const std::vector<size_t>& indices,
	const size_t& N,
	std::vector<size_t>& output
)
{
    // Ensure that N is not greater than the size of the container
    if (N > indices.size())
	{
        fprintf(stderr, "N cannot be greater than the size of the container.\n");
		return false;
    }

    // Create a random engine
    std::random_device rd;
    std::mt19937 gen(rd());

    // Convert the container to a vector if it's not already
    std::vector<size_t> indices_vec(indices.begin(), indices.end());

    // Shuffle the indices
    std::shuffle(indices_vec.begin(), indices_vec.end(), gen);

    // Clear the output vector and add the first N elements
    output.clear();
    output.insert(output.end(), indices_vec.begin(), indices_vec.begin() + N);
	return true;
}

template <
	class _ModelEstimator,
	class _NeighborhoodGraph,
	class _ScoringFunction = MSACScoringFunction<_ModelEstimator>,
	class _PreemptiveModelVerification = preemption::EmptyPreemptiveVerfication<_ModelEstimator>,
	class _FastInlierSelector = inlier_selector::EmptyInlierSelector<_ModelEstimator, _NeighborhoodGraph>
> class GCRANSAC
{
public:
	using Model = typename _ModelEstimator::Model;
	using ResidualType = typename _ModelEstimator::ResidualType;
	using InlierContainerType = typename _ModelEstimator::InlierContainerType;
	using WeightType = typename _ModelEstimator::WeightType;
	using SampleSizeType = typename _ModelEstimator::SampleSizeType;
	using ScoreType = Score<_ModelEstimator::ResidualDimension::value>;
	static constexpr size_t ResidualDimension = _ModelEstimator::ResidualDimension::value;

	utils::Settings settings;

	GCRANSAC() :
		settings(ResidualDimension),
		time_limit(std::numeric_limits<double>::max()),
		truncated_thresholds(1.5 * settings.threshold),
		squared_truncated_thresholds(truncated_thresholds * truncated_thresholds),
		scoring_function(std::make_unique<_ScoringFunction>())
		
	{}

	~GCRANSAC() 
	{ 
		_ScoringFunction* scoring_function_ptr = scoring_function.release();
		delete scoring_function_ptr;
	}

	static bool isSampleSubMinimal(
		const _ModelEstimator& model_estimator,
		const SampleSizeType& inliers_sizes
	)
	{
		const SampleSizeType min_sample_sizes = model_estimator.sampleSize();
		for (size_t i = 0; i < ResidualDimension; i++)
		{
			if (inliers_sizes[i] < min_sample_sizes[i])
			{
				return false;
			}
		}
		return true;
	}

	static bool isSampleMinimal(
		const _ModelEstimator& model_estimator,
		const SampleSizeType& inliers_sizes
	)
	{
		const SampleSizeType min_sample_sizes = model_estimator.sampleSize();
		for (size_t i = 0; i < ResidualDimension; i++)
		{
			if (inliers_sizes[i] > min_sample_sizes[i])
			{
				return false;
			}
		}
		return true;
	}

	static bool isSampleMinimal(
		const _ModelEstimator& model_estimator,
		const InlierContainerType& inliers
	)
	{
		SampleSizeType inliers_sizes{};
		for (size_t i = 0; i < ResidualDimension; i++)
		{
			inliers_sizes[i] = inliers[i].size();
		}
		return isSampleMinimal(model_estimator, inliers_sizes);
	}

	static bool sample(
		const std::vector<size_t>& inliers,
		const size_t& sample_size,
		std::vector<size_t>& samples
	)
	{
		if (!get_random_subset(inliers, sample_size, samples))
		{
			fprintf(stderr, "Failed to sample inlier set.\n");
			return false;
		}
		return true;
	}

	static bool sample(
		const InlierContainerType& inliers,
		const SampleSizeType& sample_sizes,
		InlierContainerType& samples
	)
	{
		for (size_t i = 0; i < ResidualDimension; i++)
		{
			if (!sample(inliers[i], sample_sizes[i], samples[i]))
			{
				fprintf(stderr, "Error when sampling inlier set at index %ld.\n", i);
				return false;
			}
		}
		return true;
	}

	// The main method applying Graph-Cut RANSAC to the input data points
	void run(
		const cv::Mat &points_,  // Data points
		const _ModelEstimator &model_estimator, // The model estimator
		const _NeighborhoodGraph *neighborhood_graph_, // The initialized neighborhood graph
		Model &obtained_model_, // The output model
		_PreemptiveModelVerification &preemptive_verification_, // The preemptive verification strategy used
		_FastInlierSelector &fast_inlier_selector_ // The fast inlier selector used
	)
	{
		/* Initialization */
		// Variables for measuring the processing time
		std::chrono::time_point<std::chrono::system_clock> start, end;
		std::chrono::duration<double> elapsed_seconds;

		statistics.main_sampler_name = "Uniform Sampler";
		statistics.local_optimizer_sampler_name = "Uniform Sampler";
		statistics.iteration_number = 0;
		statistics.graph_cut_number = 0;
		statistics.local_optimization_number = 0;
		statistics.neighbor_number = 0;
		statistics.processing_time = 0.0;

		// The size of a minimal sample used for the estimation
		const SampleSizeType sample_number = model_estimator.sampleSize();
		point_number = points_.rows; // Number of points in the dataset
		const auto min_sample_num = *(std::min_element(
			sample_number.begin(), sample_number.end()
		));
		if (point_number < min_sample_num)
		{
			std::stringstream error_msg;
			error_msg << "ERROR: insufficient samples to proceed!\n"
						"There are only " << point_number << " samples and the "
						"minimal number of samples needed for the model "
						"estimation is " << min_sample_num << ".\n";
			throw std::runtime_error(error_msg.str());
		}

		// log(1 - confidence) used for determining the required number of iterations
		log_probability = log(1.0 - settings.confidence);
		// Maximum number of iterations
		SampleSizeType init_inlier_nums{};
		init_inlier_nums.fill(1);
		auto max_iteration = getIterationNumber(init_inlier_nums, sample_number);

		InlierContainerType current_sample{}; // Minimal sample for model fitting
		bool do_local_optimization = false; // Flag to show if local optimization should be applied

		size_t inlier_container_offset = 0; // Index to show which inlier vector is currently in use
		size_t inlier_container_idx; // The index of the currently tested inlier container
		Model so_far_the_best_model; // The current so-far-the-best model parameters
		ScoreType current_score; // The score of the current model 
		ScoreType so_far_the_best_score; // The score of the current so-far-the-best model
		std::array<InlierContainerType, 2> temp_inner_inliers{}; // The inliers of the current and previous best models
		for (size_t i = 0; i < ResidualDimension; i++)
		{
			temp_inner_inliers[0][i].reserve(point_number);
			temp_inner_inliers[1][i].reserve(point_number);
		}

		neighborhood_graph = neighborhood_graph_; // The neighborhood graph used for the graph-cut local optimization

		// Initialize the pool for sampling
		InlierContainerType pool{};
		for (size_t i = 0; i < ResidualDimension; i++)
		{
			for (size_t j = 0; j < point_number; j++)
			{
				pool[i][j] = j;
			}
		}

		// Initialize the starting time if there is a desired FPS set
		start = std::chrono::system_clock::now();

		// The model estimated from a minimal subset
		std::vector<Model> models;
		models.reserve(model_estimator.maximumMinimalSolutions());
		
		// Variables used for the fast inlier selection if needed
		std::vector<const std::vector<size_t>*> preselected_index_sets;  // The indices of the points selected by the proposed approach
		size_t selected_point_number;
		// Initializing the variables if the inlier selection is used
		if constexpr (_FastInlierSelector::doesSomething())
		{
			const auto &cell_number = neighborhood_graph->filledCellNumber();
			preselected_index_sets.reserve(cell_number * cell_number); // Occupying the required memory
		}

		/*
			The main RANSAC iteration
		*/
		while (settings.min_iteration_number > statistics.iteration_number ||
			statistics.iteration_number < MIN(max_iteration, settings.max_iteration_number))
		{
			// Do not apply local optimization if not needed
			do_local_optimization = false;

			// Increase the iteration number counter
			++statistics.iteration_number;

			// Resize the vector containing the models
			models.resize(0);

			size_t unsuccessful_model_generations = 0;
			// Select a minimal sample and estimate the implied model parameters if possible.
			// If, after a certain number of sample selections, there is no success, terminate.
			while (unsuccessful_model_generations++ <= settings.max_unsuccessful_model_generations)
			{
				// If the sampling is not successful, try again.
				if (!sample(pool, sample_number, current_sample))
				{
					// main_sampler->update(
					// 	current_sample, sample_number,
					// 	statistics.iteration_number, 0.0
					// );
					continue;
				}

				// Check if the selected sample is valid before estimating the model
				// parameters which usually takes more time. 
				if (!model_estimator.isValidSample(points_, current_sample))
				{
					// main_sampler->update(
					// 	current_sample,
					// 	sample_number,
					// 	statistics.iteration_number,
					// 	0.0);
					continue;
				}

				// Estimate the model parameters using the current sample
				if (model_estimator.estimateModel(points_, current_sample, models))
				{
					break;
				}

				// main_sampler->update(
				// 	current_sample,
				// 	sample_number,
				// 	statistics.iteration_number,
				// 	0.0);
			}

			// Increase the iteration number by the number of unsuccessful model generations as well.
			statistics.iteration_number += (unsuccessful_model_generations - 1);

			// Select the so-far-the-best from the estimated models
			for (auto &model : models)
			{
				// Do point pre-filtering if needed.
				if constexpr (_FastInlierSelector::doesSomething())
				{
					// Remove the previous selection
					preselected_index_sets.clear();
					selected_point_number = 0;

					// Get the indices of the points using the proposed grid-based selection
					fast_inlier_selector_.run(
						points_, // The point correspondences
						model, // The model to be verified
						*neighborhood_graph_, // The neighborhood structure
						truncated_thresholds, // The inlier-outlier threshold
						preselected_index_sets, // The 4D cells selected by the algorithm
						selected_point_number); // The total number of points in the cells

					// Check if the inlier upper bound is lower than the inlier number of 
					// the so-far-the-best model. 
					if (selected_point_number < so_far_the_best_score.num_inliers())
					{
						++statistics.rejected_models;
						continue;
					}
					++statistics.accepted_models;
				} 

				// Check if the model should be rejected by the used preemptive verification strategy.
				// If there is no preemption, i.e. EmptyPreemptiveVerfication is used, this should be skipped.
				if constexpr (!std::is_same<preemption::EmptyPreemptiveVerfication<_ModelEstimator>, _PreemptiveModelVerification>())
				{
					bool should_reject = false;
					// Use the pre-selected inlier indices if the pre-selection is applied
					if constexpr (_FastInlierSelector::doesSomething())
					{
						if (!preemptive_verification_.verifyModel(model, // The current model
							model_estimator, // The model estimation object
							truncated_thresholds, // The truncated threshold
							statistics.iteration_number, // The current iteration number
							so_far_the_best_score, // The current best score
							points_, // The data points
							current_sample, // The current minimal sample
							sample_number, // The number of samples used
							temp_inner_inliers[inlier_container_offset], // The current inlier set
							current_score,
							&preselected_index_sets))
							should_reject = true;
					}
					else
					{
						if (!preemptive_verification_.verifyModel(model, // The current model
							model_estimator, // The model estimation object
							truncated_thresholds, // The truncated threshold
							statistics.iteration_number, // The current iteration number
							so_far_the_best_score, // The current best score
							points_, // The data points
							current_sample, // The current minimal sample
							sample_number, // The number of samples used
							temp_inner_inliers[inlier_container_offset], // The current inlier set
							current_score))
							should_reject = true;
					}

					if (should_reject)
					{
						++statistics.rejected_models;
						continue;
					}
					++statistics.accepted_models;
				}

				// Get the inliers and the score of the non-optimized model
				if constexpr (!_PreemptiveModelVerification::providesScore())
				{
					// Use the pre-selected inlier indices if the pre-selection is applied
					if constexpr (_FastInlierSelector::doesSomething())
						current_score = scoring_function->getScore(
							points_, // All points
							model, // The current model parameters
							model_estimator, // The estimator 
							settings.threshold, // The current threshold
							temp_inner_inliers[inlier_container_offset], // The current inlier set
							so_far_the_best_score, // The score of the current so-far-the-best model
							&preselected_index_sets // The point index set consisting of the pre-selected points' indices
						);
					else // Otherwise, run on all points
						current_score = scoring_function->getScore(
							points_, // All points
							model, // The current model parameters
							model_estimator, // The estimator 
							settings.threshold, // The current threshold
							temp_inner_inliers[inlier_container_offset], // The current inlier set
							so_far_the_best_score // The score of the current so-far-the-best model
						);
				}

				bool is_model_updated = false;

				// Store the model of its score is higher than that of the previous best
				if (so_far_the_best_score < current_score && // Comparing the so-far-the-best model's score and current model's score
					model_estimator.isValidModel(model, // The current model parameters
						points_, // All input points
						temp_inner_inliers[inlier_container_offset], // The inliers of the current model
						current_sample, // The minimal sample initializing the model
						truncated_thresholds, // The truncated inlier-outlier threshold
						is_model_updated))
				{
					// Get the inliers and the score of the non-optimized model
					if (is_model_updated)
					{
						current_score = scoring_function->getScore(
							points_, // All points
							model, // The current model parameters
							model_estimator, // The estimator 
							settings.threshold, // The current threshold
							temp_inner_inliers[inlier_container_offset], // The current inlier set
							so_far_the_best_score // The score of the current so-far-the-best model
						);
					}

					inlier_container_offset = 1 - inlier_container_offset; // flip offset
					so_far_the_best_model = model; // The new so-far-the-best model
					so_far_the_best_score = current_score; // The new so-far-the-best model's score
					// Decide if local optimization is needed. The current criterion requires a minimum number of iterations
					// and number of inliers before applying GC.
					// TODO is this the right criterion when there are multiple
					// inliers types?
					bool non_minimal_inlier_set = false;
					for (size_t i = 0; i < ResidualDimension; i++)
					{
						if (so_far_the_best_score.num_inliers_by_type(i) > sample_number[i])
						{
							non_minimal_inlier_set = true;
							break;
						}
					}
					const bool sufficient_num_iters = statistics.iteration_number > settings.min_iteration_number_before_lo;
					do_local_optimization = sufficient_num_iters && non_minimal_inlier_set;

					// Update the number of maximum iterations
					max_iteration = getIterationNumber(
						so_far_the_best_score.inlier_num_array(),
						sample_number
					);
				}
			}

			// main_sampler->update(
			// 	current_sample,
			// 	sample_number,
			// 	statistics.iteration_number,
			// 	0.0);

			// Apply local optimziation
			if (settings.do_local_optimization && // Input flag to decide if local optimization is needed
				do_local_optimization) // A flag to decide if all the criteria meet to apply local optimization
			{
				// Increase the number of local optimizations applied
				++statistics.local_optimization_number;

				// Graph-cut-based local optimization 
				graphCutLocalOptimization(points_, // All points
					temp_inner_inliers[inlier_container_offset], // Inlier set of the current so-far-the-best model
					so_far_the_best_model, // Best model parameters
					so_far_the_best_score, // Best model score
					model_estimator, // Estimator
					settings.max_local_optimization_number); // Maximum local optimization steps

				// Update the maximum number of iterations variable
				max_iteration = getIterationNumber(
					so_far_the_best_score.inlier_num_array(), // The current inlier number
					sample_number // The sample size
				);
			}

			// Apply time limit if there is a required FPS set
			if (settings.desired_fps > -1)
			{
				end = std::chrono::system_clock::now(); // The current time
				elapsed_seconds = end - start; // Time elapsed since the algorithm started

				// Interrupt the algorithm if the time limit is exceeded
				if (elapsed_seconds.count() > time_limit)
				{
					settings.min_iteration_number = 0;
					max_iteration = 0;
					break;
				}
			}
		}

		// If the best model has only minimal number of points, the model
		// is not considered to be found.
		if (isSampleMinimal(model_estimator, so_far_the_best_score.inlier_num_array()))
		{
			end = std::chrono::system_clock::now(); // The current time
			elapsed_seconds = end - start; // Time elapsed since the algorithm started
			statistics.processing_time = elapsed_seconds.count();
			return;
		}

		// Apply a final local optimization if it hasn't been applied yet
		if (settings.do_local_optimization &&
			statistics.local_optimization_number == 0)
		{
			// Increase the number of local optimizations applied
			++statistics.local_optimization_number;

			// Graph-cut-based local optimization 
			graphCutLocalOptimization(points_, // All points
				temp_inner_inliers[inlier_container_offset], // Inlier set of the current so-far-the-best model
				so_far_the_best_model, // Best model parameters
				so_far_the_best_score, // Best model score
				model_estimator, // Estimator
				settings.max_local_optimization_number); // Maximum local optimization steps
		}

		// Recalculate the score if needed (i.e. there is some inconstistency in
		// in the number of inliers stored and calculated).
		if (temp_inner_inliers[inlier_container_offset].size() != so_far_the_best_score.num_inliers())
			inlier_container_offset = 1 - inlier_container_offset;

		if (temp_inner_inliers[inlier_container_offset].size() != so_far_the_best_score.num_inliers())
			so_far_the_best_score = scoring_function->getScore(
				points_, // All points
				so_far_the_best_model, // Best model parameters
				model_estimator, // The estimator
				settings.threshold, // The inlier-outlier threshold
				temp_inner_inliers[inlier_container_offset] // The current inliers
			);

		// Apply iteration least-squares fitting to get the final model parameters if needed
		bool iterative_refitting_applied = false;
		if (settings.do_final_iterated_least_squares)
		{
			Model model = so_far_the_best_model; // The model which is re-estimated by iteratively re-weighted least-squares
			bool success = iteratedLeastSquaresFitting(
				points_, // The input data points
				model_estimator, // The model estimator
				settings.threshold, // The inlier-outlier threshold
				temp_inner_inliers[inlier_container_offset], // The resulting inlier set
				model); // The estimated model

			if (success)
			{
				inlier_container_idx = 1 - inlier_container_offset;
				for (auto& inlier_set : temp_inner_inliers[inlier_container_idx])
				{
					inlier_set.clear();
				}
				current_score = scoring_function->getScore(
					points_, // All points
					model, // Best model parameters
					model_estimator, // The estimator
					settings.threshold, // The inlier-outlier threshold
					temp_inner_inliers[inlier_container_idx] // The current inliers
				);

				if (so_far_the_best_score < current_score)
				{
					iterative_refitting_applied = true;
					so_far_the_best_model = model;
					inlier_container_offset = inlier_container_idx;
				}
			}
		}
		
		if (!iterative_refitting_applied) // Otherwise, do only one least-squares fitting on all of the inliers
		{
			// Estimate the final model using the full inlier set
			models.clear();
			model_estimator.estimateModelNonminimal(
				points_,
				temp_inner_inliers[inlier_container_offset],
				models
			);

			// Selecting the best model by their scores
			for (auto &model : models)
			{
				inlier_container_idx = 1 - inlier_container_offset;
				for (auto& inlier_set : temp_inner_inliers[inlier_container_idx])
				{
					inlier_set.clear();
				}
				current_score = scoring_function->getScore(
					points_, // All points
					model, // Best model parameters
					model_estimator, // The estimator
					settings.threshold, // The inlier-outlier threshold
					temp_inner_inliers[inlier_container_idx]
				);

				if (so_far_the_best_score < current_score)
				{
					so_far_the_best_model = model;
					inlier_container_offset = inlier_container_idx;
				}
			}

			if (models.size() == 0)
			{
				so_far_the_best_model = models[0]; // TODO won't this cause a segfault?
			} 
		}

		// Return the inlier set and the estimated model parameters
		statistics.inliers.swap(temp_inner_inliers[inlier_container_offset]);
		statistics.score = so_far_the_best_score.value();
		obtained_model_ = so_far_the_best_model;

		end = std::chrono::system_clock::now(); // The current time
		elapsed_seconds = end - start; // Time elapsed since the algorithm started
		statistics.processing_time = elapsed_seconds.count();
	}

	// The main method applying Graph-Cut RANSAC to the input data points
	void run(
		const cv::Mat &points_,  // Data points
		const _ModelEstimator &model_estimator, // The model estimator
		const _NeighborhoodGraph *neighborhood_graph_, // The initialized neighborhood graph
		Model &obtained_model_ // The output model
	)
	{
		// Instantiate the preemptive model verification strategy
		_PreemptiveModelVerification preemptive_verification;
		
		// Instantiate the fast inlier selector object
		_FastInlierSelector fast_inlier_selector(neighborhood_graph_);

		// Running GC-RANSAC by using the specified preemptive verification
		run(points_, model_estimator, neighborhood_graph_, obtained_model_, 
			preemptive_verification, fast_inlier_selector);
	}

	OLGA_INLINE void setFPS(double fps_)
	{
		settings.desired_fps = fps_;
		time_limit = 1.0 / fps_;
	}

	// Return the constant reference of the scoring function
	const utils::RANSACStatistics<ResidualDimension> &getRansacStatistics() const { return statistics; }

	// Return the reference of the scoring function
	utils::RANSACStatistics<ResidualDimension> &getMutableRansacStatistics() { return statistics; }

	// Return the constant reference of the scoring function
	const _ScoringFunction &getScoringFunction() const { return *scoring_function; }

	// Return the reference of the scoring function
	_ScoringFunction &getMutableScoringFunction() { return *scoring_function; }

protected:
	double time_limit; // The desired time limit
	std::vector<std::vector<cv::DMatch>> neighbours; // The neighborhood structure
	utils::RANSACStatistics<ResidualDimension> statistics; // RANSAC statistics
	size_t point_number; // The point number
	Eigen::ArrayXd truncated_thresholds; // 3 / 2 * threshold_
	Eigen::ArrayXd squared_truncated_thresholds; // 9 / 4 * threshold_^2
	int step_size; // Step size per processes
	double log_probability; // The logarithm of 1 - confidence
	const _NeighborhoodGraph *neighborhood_graph;
	std::unique_ptr<_ScoringFunction> scoring_function; // The scoring function used to measure the quality of a model

	Graph<double, double, double> *graph; // The graph for graph-cut

	// Computes the desired iteration number for RANSAC w.r.t. to the current inlier number
	size_t getIterationNumber(
		const SampleSizeType& inlier_numbers, // The inlier number
		const SampleSizeType& samples_sizes  // The current_sample size
	) const
	{
		double q = 1.0;
		const auto inv_point_num = 1.0 / static_cast<double>(point_number);
		for (size_t i = 0; i < ResidualDimension; i++)
		{
			const auto inlier_ratio = inv_point_num * static_cast<double>(inlier_numbers[i]);
			q *= std::pow(inlier_ratio, static_cast<double>(samples_sizes[i]));
		}
		const auto log2 = std::log(1 - q);
		if (abs(log2) < std::numeric_limits<double>::epsilon())
		{
			return std::numeric_limits<size_t>::max();
		}
		const auto iter = std::ceil(log_probability / log2);
		return static_cast<size_t>(iter);
	}

	// Returns a labeling w.r.t. the current model and point set
	void labeling(const cv::Mat &points_, // The input data points
		size_t neighbor_number_, // The neighbor number in the graph
		const std::vector<std::vector<cv::DMatch>> &neighbors_, // The neighborhood
		const Model &model_, // The current model_
		const _ModelEstimator& model_estimator, // The model estimator
		double lambda_, // The weight for the spatial coherence term
		const double& threshold_, // The threshold for the inlier-outlier decision
		std::vector<size_t> &inliers, // The resulting inlier set
		double &energy_ // The resulting energy
	)
	{
		static_assert(ResidualDimension == 1, "Labeling function should not be called when multiple residual types exist!");
		inliers.reserve(points_.rows);
		const int &point_number = points_.rows;

		// Initializing the problem graph for the graph-cut algorithm.
		Energy<double, double, double> *problem_graph =
			new Energy<double, double, double>(point_number, // The number of vertices
				neighbor_number_, // The number of edges
				NULL);

		// Add a vertex for each point
		for (auto i = 0; i < point_number; ++i)
			problem_graph->add_node();

		// The distance and energy for each point
		std::vector<double> distance_per_threshold;
		distance_per_threshold.reserve(point_number);
		double tmp_energy;
		// TODO Gaussian kernel should be multivariate now. What is the 9/4 factor for?
		const double squared_truncated_threshold = (9.0 / 4.0) * threshold_ * threshold_;
		const double one_minus_lambda = 1.0 - lambda_;

		// Estimate the vertex capacities
		for (size_t i = 0; i < point_number; ++i)
		{
			// Calculating the point-to-model squared residual
			double tmp_squared_distance = model_estimator.squaredResidual(
				points_.row(i), model_
			)[0];
			// Storing the residual divided by the squared threshold
			distance_per_threshold.emplace_back(
				std::clamp(tmp_squared_distance / squared_truncated_threshold, 0.0, 1.0));
			// Calculating the implied unary energy
			tmp_energy = 1 - distance_per_threshold.back();

			// Adding the unary energy to the graph
			if (tmp_squared_distance <= squared_truncated_threshold)
				problem_graph->add_term1(i, one_minus_lambda * tmp_energy, 0);
			else 
				problem_graph->add_term1(i, 0, one_minus_lambda * (1 - tmp_energy));
		}

		std::vector<std::vector<int>> used_edges(point_number, std::vector<int>(point_number, 0));

		if (lambda_ > 0)
		{
			double energy1, energy2, energy_sum;
			double e00, e11 = 0; // Unused: e01 = 1.0, e10 = 1.0,

			// Iterate through all points and set their edges
			for (auto point_idx = 0; point_idx < point_number; ++point_idx)
			{
				energy1 = distance_per_threshold[point_idx]; // Truncated quadratic cost

				// Iterate through  all neighbors
				for (const size_t &actual_neighbor_idx : neighborhood_graph->getNeighbors(point_idx))
				{
					if (actual_neighbor_idx == point_idx)
						continue;

					if (actual_neighbor_idx == point_idx || actual_neighbor_idx < 0)
						continue;

					if (used_edges[actual_neighbor_idx][point_idx] == 1 ||
						used_edges[point_idx][actual_neighbor_idx] == 1)
						continue;

					used_edges[actual_neighbor_idx][point_idx] = 1;
					used_edges[point_idx][actual_neighbor_idx] = 1;

					energy2 = distance_per_threshold[actual_neighbor_idx]; // Truncated quadratic cost
					energy_sum = energy1 + energy2;

					e00 = 0.5 * energy_sum;

					constexpr double e01_plus_e10 = 2.0; // e01 + e10 = 2
					if (e00 + e11 > e01_plus_e10)
						printf("Non-submodular expansion term detected; smooth costs must be a metric for expansion\n");

					problem_graph->add_term2(point_idx, // The current point's index
						actual_neighbor_idx, // The current neighbor's index
						e00 * lambda_,
						lambda_, // = e01 * lambda
						lambda_, // = e10 * lambda
						e11 * lambda_);
				}
			}
		}

		// Run the standard st-graph-cut algorithm
		problem_graph->minimize();

		// Select the inliers, i.e., the points labeled as SINK.
		for (auto point_idx = 0; point_idx < points_.rows; ++point_idx)
			if (problem_graph->what_segment(point_idx) == Graph<double, double, double>::SINK)
				inliers.emplace_back(point_idx);

		// Clean the memory
		delete problem_graph;
	}

	// Apply the graph-cut optimization for GC-RANSAC
	bool graphCutLocalOptimization(
		const cv::Mat &points_, // The input data points
		InlierContainerType& so_far_the_best_inliers, // The input, than the resulting inlier set
		Model &so_far_the_best_model_, // The current model
		ScoreType &so_far_the_best_score, // The current score
		const _ModelEstimator &model_estimator, // The model estimator
		const size_t trial_number_ // The max trial number
	)
	{
		const auto &inlier_limit = model_estimator.inlierLimit(); // Number of points used in the inner RANSAC
		ScoreType max_score = so_far_the_best_score; // The current best score
		Model best_model = so_far_the_best_model_; // The current best model
		std::vector<Model> models; // The estimated models' parameters
		InlierContainerType best_inliers{}; // Inliers of the best model
		InlierContainerType	inliers{};
		InlierContainerType	tmp_inliers{}; // The inliers of the current model
		InlierContainerType current_sample{}; // The current sample used in the inner RANSAC
		SampleSizeType sample_size{};
		
		const auto n_points = static_cast<size_t>(points_.rows);
		for (size_t i = 0; i < ResidualDimension; i++)
		{
			inliers[i].reserve(n_points);
			best_inliers[i].reserve(n_points);
			tmp_inliers[i].reserve(n_points);
			current_sample[i].reserve(inlier_limit[i]);
		}
		models.reserve(model_estimator.maximumMinimalSolutions());

		// Increase the number of the local optimizations applied
		++statistics.local_optimization_number;

		// Apply the graph-cut-based local optimization
		bool updated = false; // A flag to see if the model is updated
		while (++statistics.graph_cut_number < settings.max_graph_cut_number)
		{
			// In the beginning, the best model is not updated
			updated = false;

			// Clear the inliers
			for (size_t i = 0; i < ResidualDimension; i++)
			{
				inliers[i].clear();
			}

			// Apply the graph-cut-based inlier/outlier labeling.
			// The inlier set will contain the points closer than the threshold and
			// their neighbors depending on the weight of the spatial coherence term.
			if constexpr (ResidualDimension > 1)
			{
				// TODO this is for non-scalar residuals and thresholds.
				// Right now this works only when lambda_ = 0.
				using namespace Eigen;
				for (auto point_idx = 0; point_idx < n_points; point_idx++)
				{
					const auto sqr_residuals = model_estimator.squaredResidual(points_.row(point_idx), best_model);
					Array<bool, ResidualDimension, 1> comparison = sqr_residuals <= squared_truncated_thresholds;
					// construct weights matrix such that inliers only contribute
					// constraints for the residuals below the corresponding
					// thresholds.
					for (size_t i = 0; i < ResidualDimension; i++)
					{
						if (comparison(i))
						{
							inliers[i].emplace_back(point_idx);
						}
					}
				}
				for (size_t i = 0; i < ResidualDimension; i++)
				{
					inliers[i].shrink_to_fit();
					sample_size[i] = inliers[i].size();
				}
			}
			else
			{
				double energy{0};
				labeling(
					points_, // The input points
					statistics.neighbor_number, // The number of neighbors, i.e. the edge number of the graph 
					neighbours, // The neighborhood graph
					best_model, // The best model parameters
					model_estimator, // The model estimator
					settings.spatial_coherence_weight, // The weight of the spatial coherence term
					settings.threshold(0), // The inlier-outlier threshold
					inliers[0], // The selected inliers
					energy // The energy after the procedure
				);
				sample_size[0] = inliers[0].size();
			}

			// Number of points (i.e. the sample size) used in the inner RANSAC

			// TODO deal with multple sample types:
			// 1. Each inlier type has its own sample size.
			// 2. A sample needs to be drawn from each inlier type.
			// 3. Estimating the model needs to be done with a combination of
			// 	  all sample types.
			for (size_t i = 0; i < ResidualDimension; i++)
			{
				sample_size[i] = std::min(inlier_limit[i], sample_size[i]);
			}

			// Run an inner RANSAC on the inliers coming from the graph-cut algorithm
			for (auto trial = 0; trial < trial_number_; ++trial)
			{
				// Reset the model vector
				models.clear();
				bool sampling_success = true;
				for (size_t i = 0; i < ResidualDimension; i++)
				{
					if (sample_size[i] < inliers[i].size())
					{
						// If there are more inliers available than the minimum number, sample randomly.
						if (!sample(inliers[i], sample_size[i], current_sample[i]))
						{
							sampling_success = false;
							break;
						}
					}
					else if (model_estimator.sampleSize()[i] < inliers[i].size())
					{
						// If there are enough inliers to estimate the model, use all of them.
						current_sample[i] = inliers[i];
					}
					else
					{
						// Otherwise, break the for-loop.
						sampling_success = false;
						break;
					}
				}
				// if sampling failed, break the for-loop.				
				if (!sampling_success)
				{
					break;
				}
				// Apply least-squares model fitting to the selected points.
				// If it fails, continue the for-loop and, thus, the sampling.
				if (!model_estimator.estimateModelNonminimal(points_, current_sample, models))
				{
					continue;
				}
				// Select the best model from the estimated set
				for (auto &model : models)
				{
					for (auto& inlier_subset : tmp_inliers)
					{
						inlier_subset.clear();
					}

					// Calculate the score of the current model
					ScoreType score = scoring_function->getScore(
						points_, model, model_estimator, settings.threshold,
						tmp_inliers, max_score
					);

					// If this model is better than the previous best, update.
					if (max_score < score) // Comparing the so-far-the-best model's score and current model's score
					{
						updated = true; // Flag saying that we have updated the model parameters
						max_score = score; // Store the new best score
						best_model = model; // Store the new best model parameters
						best_inliers.swap(tmp_inliers);
					}
				}
			}

			// If the model is not updated, interrupt the procedure
			if (!updated)
			{
				break;
			}
		}

		// If the new best score is better than the original one, update the model parameters.
		if (so_far_the_best_score < max_score) // Comparing the original best score and best score of the local optimization
		{
			so_far_the_best_score = max_score; // Store the new best score
			so_far_the_best_model_ = best_model;
			so_far_the_best_inliers.swap(best_inliers);
			return true;
		}
		return false;
	}

	void iteratedLeastSquaresComputeWeights(
		const cv::Mat& points,
		const _ModelEstimator& model_estimator,
		const InlierContainerType& inliers,
		const Model& model,
		WeightType& weights
	) const 
	{
		const auto n_weights = static_cast<size_t>(points.rows);
		for (size_t i = 0; i < ResidualDimension; i++)
		{
			auto& weights_i = weights[i];
			weights_i = std::vector<double>(n_weights, 0.0);
			const auto& inliers_i = inliers[i];
			const auto inv_thresh_i = 1.0 / squared_truncated_thresholds(i);

			for (const auto& point_idx : inliers_i)
			{
				// The squares residual of the current inlier
				const auto squared_residual = model_estimator.squaredResidual(
					points.row(point_idx), model
				);
				// Calculate the Tukey bisquare weights
				auto weight = 1.0 - (squared_residual(i) * inv_thresh_i);
				weight = std::max(0.0, weight);
				weights_i[point_idx] = weight * weight;
			}
		}
	}

	void iteratedLeastSquaresModelFitting(
		const cv::Mat& points,
		const _ModelEstimator& model_estimator,
		const InlierContainerType& inliers,
		const Model& current_model,
		std::vector<Model> models
	) const
	{
		WeightType weights{};
		// Calculate the weights if iteratively re-weighted least-squares is used			
		if (model_estimator.isWeightingApplicable())
		{
			iteratedLeastSquaresComputeWeights(
				points, model_estimator, inliers, current_model, weights
			);
		}
		// Estimate the model from the current inlier set
		model_estimator.estimateModelNonminimal(
			points, inliers, models, weights
		);
	}

	bool iteratedLeastSquaresFitting(
		const cv::Mat& points,
		const _ModelEstimator& model_estimator,
		const Eigen::ArrayXd& thresholds,
		InlierContainerType& inliers,
		Model& current_model
	)
	{
		using namespace Eigen;
		
		if (isSampleMinimal(model_estimator, inliers)) // Return if there are not enough points
		{
			return false;
		}

		size_t iterations = 0; // Number of least-squares iterations
		// Iterated least-squares model fitting
		ScoreType best_score; // The score of the best estimated model
		while (++iterations < settings.max_least_squares_iterations)
		{
			std::vector<Model> models;
			iteratedLeastSquaresModelFitting(
				points, model_estimator, inliers, current_model, models
			);
			
			if (models.size() == 0) // If there is no model estimated, interrupt the procedure
			{
				break;
			}
			if (models.size() == 1) // If a single model is estimated we do not have to care about selecting the best
			{
				// Calculate the score of the current model
				InlierContainerType tmp_inliers; // Inliers of the current model
				ScoreType score = scoring_function->getScore(
					points, models[0], model_estimator, thresholds, tmp_inliers
				);
				
				// Break if the are not enough inliers
				if (isSampleSubMinimal(model_estimator, score.inlier_num_array()))
				{
					break;
				}
				// check if inlier numbers have changed
				bool equal_inliers_sets = true;
				for (size_t i = 0; i < ResidualDimension; i++)
				{
					if (score.num_inliers_by_type(i) != inliers[i].size())
					{
						equal_inliers_sets = false;
						break;
					}
				}
				// Interrupt the procedure if the inlier number has not changed.
				// Therefore, the previous and current model parameters are likely the same.
				if (equal_inliers_sets)
				{
					break;
				}
				// Update the output model
				current_model = models[0];
				// Store the inliers of the new model
				inliers.swap(tmp_inliers);
			}
			else // If multiple models are estimated select the best (i.e. the one having the highest score) one
			{
				bool updated = false; // A flag determining if the model is updated

				// Evaluate all the estimated models to find the best
				for (const auto &cand_model : models)
				{
					// Calculate the score of the current model
					InlierContainerType tmp_inliers;
					ScoreType score = scoring_function->getScore(
						points, cand_model, model_estimator, thresholds,
						tmp_inliers
					);

					if (isSampleSubMinimal(model_estimator, score.inlier_num_array()))
					{
						// Continue if the are not enough inliers
						continue;
					}

					// check if inlier numbers have changed
					bool equal_inliers_sets = true;
					for (size_t i = 0; i < ResidualDimension; i++)
					{
						if (score.num_inliers_by_type(i) != tmp_inliers[i].size())
						{
							equal_inliers_sets = false;
							break;
						}
					}
					// Interrupt the procedure if the inlier number has not changed.
					// Therefore, the previous and current model parameters are likely the same.
					if (equal_inliers_sets)
					{
						break;
					}

					// Update the model if its score is higher than that of the current best
					if (score > best_score)
					{
						updated = true; // Set a flag saying that the model is updated, so the process should continue
						best_score = score; // Store the new score
						current_model = cand_model; // Store the new model
						inliers.swap(tmp_inliers); // Store the inliers of the new model
					}
				}

				// If the model has not been updated, interrupt the procedure
				if (!updated)
				{
					break;
				}
			}
		}

		// If there were more than one iterations, the procedure is considered successfull
		return iterations > 1;
	}
};

}
