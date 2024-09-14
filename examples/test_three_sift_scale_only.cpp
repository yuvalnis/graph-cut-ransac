#include <vector>
#include <iostream>
#include <cstdio>
#include <random>
#include <stdlib.h>
#include <string>

#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>

#include "GCRANSAC.h"
#include "types.h"
#include "neighborhood/grid_neighborhood_graph.h"
#include "estimators/rectifying_homography_estimator.h"
#include "inlier_selectors/empty_inlier_selector.h"
#include "preemption/preemption_empty.h"
#include "estimators/solver_rectifying_homography_three_sift.hpp"

using namespace gcransac;

constexpr size_t kFeatureSize = 3;
constexpr double kSquareSize = 40.0;

double gaussianNoise(double mean, double stddev)
{
	static std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, stddev);
    return distribution(generator);
}

int main(int argc, char* argv[])
{
	using namespace neighborhood;
	using ModelEstimator = utils::ScaleBasedRectifyingHomographyEstimator;

	int arg_count = 1;
	if (argc < 2 || argc > 4)
	{	
		std::cout << "Usage: " << argv[0]
				  << " scale_thresh"
				  << " [num_of_squares]"
				  << " [coord_noise_stddev]"
				  << "\n";
		return -1;
	}
	
	double scale_residual_thresh = std::stod(argv[arg_count++]);
	if (std::signbit(scale_residual_thresh))
	{
		std::cout << "Error: scale threshold must be non-negative\n";
		return -1;
	}

	size_t num_squares = 5;
	if (argc > arg_count)
	{
		if (atoi(argv[arg_count]) < 1)
		{
			std::cout << "Error: number of squares must be a positive number\n";
			return -1;
		}
		num_squares = static_cast<size_t>(atoi(argv[arg_count++]));
	}

	double coord_noise_stddev = 0.0;
	if (argc > arg_count)
	{
		coord_noise_stddev = std::stod(argv[arg_count++]);
		if (std::signbit(coord_noise_stddev))
		{
			std::cout << "Error: standard deviation of noise in coordinates must be non-negative\n";
			return -1;
		}
	}

	const double spatial_coherence_weight = 0;
	const size_t min_iteration_number = 10000;
	const size_t max_iteration_number = 10000;
	const size_t max_local_optimization_number = 50;

    // prepare inputs
    std::vector<double> features;
    for (size_t i = 0; i < num_squares; i++)
    {
        for (size_t j = 0; j < num_squares; j++)
        {
			double x = 0.5 * kSquareSize * (2 * i + 1) + gaussianNoise(0.0, coord_noise_stddev);
			double y = 0.5 * kSquareSize * (2 * j + 1) + gaussianNoise(0.0, coord_noise_stddev);
			double s = kSquareSize + gaussianNoise(0.0, coord_noise_stddev);
            features.push_back(x); // x-coordinate
            features.push_back(y); // y-coordinate
            features.push_back(s); // scale
        }
    }

    std::vector<bool> inliers(num_squares * num_squares);
	std::vector<double> weights(num_squares * num_squares, 1.0);
    std::vector<double> homography(9);

    // start of block of code which goes in gcransac_python.cpp
    if (features.size() % kFeatureSize != 0)
	{
		fprintf(stderr,
			"The container of SIFT features should have a size which is a \
			multiple of %lu. Its size is %lu.\n",
			kFeatureSize,
			features.size()
		);
		return 0;
	}

	const auto num_features = features.size() / kFeatureSize; 
	if (num_features != weights.size())
	{
		fprintf(
			stderr,
			"The number of weights (%lu) is different than the number of features (%lu).\n",
			weights.size(),
			num_features
		);
		return 0;
	}

	cv::Mat sample_set(num_features, kFeatureSize, CV_64F, &features[0]);

	// initialize neighborhood graph
	// Using only the point coordinates and not the affine elements when constructing the neighborhood.
	cv::Mat empty_point_matrix(0, kFeatureSize, CV_64F);
	std::vector<double> cell_sizes(kFeatureSize, 0.0);
	std::unique_ptr<NeighborhoodGraph> neighborhood_graph =
		std::make_unique<GridNeighborhoodGraph<kFeatureSize>>(
			&empty_point_matrix, cell_sizes, 1
	);

	// initialize SIFT-based rectifying homography estimator
	ModelEstimator estimator;

	preemption::EmptyPreemptiveVerfication<ModelEstimator> preemptive_verification;
	inlier_selector::EmptyInlierSelector<
		ModelEstimator, NeighborhoodGraph
	> inlier_selector(neighborhood_graph.get());

	GCRANSAC<ModelEstimator, NeighborhoodGraph> gcransac;
	gcransac.settings.threshold(0) = scale_residual_thresh;
	gcransac.settings.do_local_optimization = true; 
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight;
	gcransac.settings.min_iteration_number = min_iteration_number;
	gcransac.settings.max_iteration_number = max_iteration_number;
	gcransac.settings.max_local_optimization_number = max_local_optimization_number;
	gcransac.settings.do_final_iterated_least_squares = true;

	ScaleBasedRectifyingHomography model;
	gcransac.run(
		sample_set, estimator, neighborhood_graph.get(), model,
		preemptive_verification, inlier_selector
	);
	const auto& statistics = gcransac.getRansacStatistics();
    const auto H = model.getHomography();
	// store final homography in vector in row-major order
	homography.resize(9);
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			homography[i * 3 + j] = H(i, j);
		}	
	}

	inliers.resize(num_features, false);
	const auto num_inliers = statistics.inliers.size();
	for (size_t pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[0][pt_idx]] = true;
	}

    // end of block of code which goes in gcransac_python.cpp

    std::cout << "\nFinal model:\n"
			  << "\nh7: " << model.h7 << ", h8: " << model.h8 << ", alpha: " << model.alpha << "\n"
			  << "\nx0: " << model.x0 << ", y0: " << model.y0 << ", s: " << model.s << "\n"
			  << "\nHomography:\n"
              << H << "\n\n"
			  << "Number of inliers: " << num_inliers << "\n";
    return 0;
}
