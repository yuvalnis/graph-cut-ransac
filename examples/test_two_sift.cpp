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
#include "estimators/solver_rectifying_homography_two_sift.hpp"

using namespace gcransac;

constexpr size_t kFeatureSize = 4;
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
	using ModelEstimator = utils::SIFTBasedRectifyingHomographyEstimator;

	int arg_count = 1;
	if (argc < 3 || argc > 6)
	{	
		std::cout << "Usage: " << argv[0]
				  << " scale_thresh"
				  << " orientation_thresh"
				  << " [num_of_squares]"
				  << " [coord_noise_stddev]"
				  << " [angle_noise_stddev]"
				  << "\n";
		return -1;
	}
	
	double scale_residual_thresh = std::stod(argv[arg_count++]);
	if (std::signbit(scale_residual_thresh))
	{
		std::cout << "Error: scale threshold must be non-negative\n";
		return -1;
	}

	double orientation_residual_thresh = std::stod(argv[arg_count++]);
	if (std::signbit(orientation_residual_thresh))
	{
		std::cout << "Error: orientation threshold must be non-negative\n";
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

	double angle_noise_stddev = 0.0;
	if (argc > arg_count)
	{
		angle_noise_stddev = std::stod(argv[arg_count++]);
		if (std::signbit(angle_noise_stddev))
		{
			std::cout << "Error: standard deviation of noise in angles must be non-negative\n";
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
			double t = 0.0 + gaussianNoise(0.0, angle_noise_stddev);
			double s = kSquareSize + gaussianNoise(0.0, coord_noise_stddev);
            features.push_back(x); // x-coordinate
            features.push_back(y); // y-coordinate
            features.push_back(t); // orientation
            features.push_back(s); // scale
        }
    }

    std::vector<bool> scale_inliers(num_squares * num_squares);
	std::vector<bool> orientation_inliers(num_squares * num_squares);
	std::vector<double> weights(num_squares * num_squares, 1.0);
    std::vector<double> homography(9);
	std::vector<double> vanishing_points(6);

    // start of block of code which goes in gcransac_python.cpp
    if (features.empty() || features.size() % kFeatureSize != 0)
	{
		fprintf(stderr,
			"The container of SIFT features should have a non-zero size which "
            "is a multiple of %lu. Its size is %lu.\n",
			kFeatureSize, features.size()
		);
		return 0;
	}

	const auto num_features = features.size() / kFeatureSize;
	if (num_features != weights.size())
	{
		fprintf(stderr,
			"The number of weights (%lu) is different than the number of "
			"features (%lu).\n",
			weights.size(), num_features
		);
		return 0;
	}

    cv::Mat sample_set(num_features, kFeatureSize, CV_64F);
	std::memcpy(sample_set.data, features.data(), features.size() * sizeof(double));

    // initialize neighborhood graph
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

	GCRANSAC<
		ModelEstimator,
		NeighborhoodGraph,
		MSACScoringFunction<ModelEstimator>,
		preemption::EmptyPreemptiveVerfication<ModelEstimator>,
		inlier_selector::EmptyInlierSelector<ModelEstimator, NeighborhoodGraph>
	> gcransac;
	gcransac.settings.threshold(0) = scale_residual_thresh;
	gcransac.settings.threshold(1) = orientation_residual_thresh;
	gcransac.settings.do_local_optimization = true; 
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight;
	gcransac.settings.min_iteration_number = min_iteration_number;
	gcransac.settings.max_iteration_number = max_iteration_number;
	gcransac.settings.max_local_optimization_number = max_local_optimization_number;
	gcransac.settings.do_final_iterated_least_squares = true;

	SIFTRectifyingHomography model;
	gcransac.run(
		sample_set, estimator, neighborhood_graph.get(), model,
		preemptive_verification, inlier_selector
	);
	const auto& statistics = gcransac.getRansacStatistics();
    const Eigen::Matrix3d H = model.getHomography();
	// store final homography in vector in row-major order
	homography.resize(9);
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			homography[i * 3 + j] = H(i, j);
		}	
	}
	// Compute the two orthogonal vanishing points in the rectified image.
	const auto c1 = std::cos(model.vanishing_point_dir1);
	const auto s1 = std::sin(model.vanishing_point_dir1);
	const auto c2 = std::cos(model.vanishing_point_dir2);
	const auto s2 = std::sin(model.vanishing_point_dir2);
	Eigen::Vector3d vp1(c1, s1, 0.0);
	Eigen::Vector3d vp2(c2, s2, 0.0);
	if (std::fabs(vp1.dot(vp2)) > 1e-9)
	{
		fprintf(stderr, "ERROR: rectified vanishing points should be orthogonal!\n");
	}
	// Warp vanishing points as the output is the vanishing points in the warped image.
	vp1 = H * vp1;
	vp2 = H * vp2;
	vanishing_points.resize(6);
	for (size_t i = 0; i < 3; i++)
	{
		vanishing_points[2 * i] = vp1(i);
		vanishing_points[2 * i + 1] = vp2(i);
	}
	// create a boolean mask of the scale inliers
    scale_inliers.resize(num_features, false);
	const auto num_scale_inliers = statistics.inliers[0].size();
	for (size_t pt_idx = 0; pt_idx < num_scale_inliers; ++pt_idx) {
		scale_inliers[statistics.inliers[0][pt_idx]] = true;
	}
	// create a boolean mask of the orientation inliers
	orientation_inliers.resize(num_features, false);
	const auto num_orientation_inliers = statistics.inliers[1].size();
	for (size_t pt_idx = 0; pt_idx < num_orientation_inliers; ++pt_idx) {
		orientation_inliers[statistics.inliers[1][pt_idx]] = true;
	}

    // end of block of code which goes in gcransac_python.cpp

    std::cout << "\nFinal model:\n"
			  << "\nh7: " << model.h7 << ", h8: " << model.h8 << ", alpha: " << model.alpha << "\n"
			  << "\nx0: " << model.x0 << ", y0: " << model.y0 << ", s: " << model.s << "\n"
			  << "\nHomography:\n"
              << H << "\n\n"
			  << "first vanishing point (unnormalized): " << vp1.transpose() << "\n"
			  << "second vanishing point (unnormalized): " << vp2.transpose() << "\n"
			  << "Number of scale-inliers: " << scale_inliers.size() << "\n"
			  << "Number of orientation-inliers: " << orientation_inliers.size() << "\n";
    return 0;
}
