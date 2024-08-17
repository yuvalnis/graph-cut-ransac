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

#include <numeric>
#include <Eigen/Dense>

namespace gcransac::utils
{

struct Settings
{
	bool do_final_iterated_least_squares{true}; // Flag to decide a final iterated least-squares fitting is needed to polish the output model parameters.
	bool do_local_optimization{true}; // Flag to decide if local optimization is needed
	bool do_graph_cut{true}; // Flag to decide of graph-cut is used in the local optimization
	bool use_inlier_limit{false}; // Flag to decide if an inlier limit is used in the local optimization to speed up the procedure

	double desired_fps{-1.0}; // The desired FPS

	size_t max_local_optimization_number{10}; // Maximum number of local optimizations
	size_t min_iteration_number_before_lo{20}; // Minimum number of RANSAC iterations before applying local optimization
	size_t min_iteration_number{20}; // Minimum number of RANSAC iterations
	size_t max_iteration_number{std::numeric_limits<size_t>::max()}; // Maximum number of RANSAC iterations
	size_t max_unsuccessful_model_generations{100}; // Maximum number of unsuccessful model generations
	size_t max_least_squares_iterations{10}; // Maximum number of iterated least-squares iterations
	size_t max_graph_cut_number{10}; // Maximum number of graph-cuts applied in each iteration
	size_t core_number{1}; // Number of parallel threads

	double confidence{0.95}; // Required confidence in the result
	double neighborhood_sphere_radius{20.0}; // The radius of the ball used for creating the neighborhood graph
	double spatial_coherence_weight{0.14}; // The weight of the spatial coherence term

	Eigen::ArrayXd threshold; // the inlier-outlier threshold

	Settings(const size_t& thresh_dim)
	{
		// Ensure the dimension is greater than 0
        if (thresh_dim == 0) {
            throw std::invalid_argument("thresh_dim must be greater than 0");
        }
		threshold = Eigen::ArrayXd::Constant(thresh_dim, 2.0);
	}
};
	
}
