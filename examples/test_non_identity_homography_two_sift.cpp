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
	if (argc < 3 || argc > 7)
	{	
		std::cout << "Usage: " << argv[0]
				  << " scale_thresh"
				  << " orientation_thresh"
				  << " [num_of_squares]"
				  << " [h7]"
				  << " [h8]"
				  << " [angle_noise]"
				  << "\n";
		return -1;
	}
	
	double scale_residual_thresh = std::stod(argv[arg_count++]);
	if (scale_residual_thresh < 1.0)
	{
		std::cout << "Error: scale threshold must be non-negative\n";
		return -1;
	}
	scale_residual_thresh = std::log(scale_residual_thresh);

	double orientation_residual_thresh = std::stod(argv[arg_count++]);
	if (std::signbit(orientation_residual_thresh))
	{
		std::cout << "Error: orientation threshold must be non-negative\n";
		return -1;
	}
	orientation_residual_thresh = M_PI * orientation_residual_thresh / 180.0;

	size_t num_squares = 3;
	if (argc > arg_count)
	{
		if (atoi(argv[arg_count]) < 1)
		{
			std::cout << "Error: number of squares must be a positive number\n";
			return -1;
		}
		num_squares = static_cast<size_t>(atoi(argv[arg_count++]));
	}

	double h7 = 0.0;
	if (argc > arg_count)
	{
		h7 = std::stod(argv[arg_count++]);
	}

	double h8 = 0.0;
	if (argc > arg_count)
	{
		h8 = std::stod(argv[arg_count++]);
	}

	double angle_noise = 0.0;
	if (argc > arg_count)
	{
		angle_noise = std::stod(argv[arg_count++]);
	}
	angle_noise = M_PI * angle_noise / 180.0;

	const double spatial_coherence_weight = 0;
	const size_t min_iteration_number = 10000;
	const size_t max_iteration_number = 10000;
	const size_t max_local_optimization_number = 50;

    SIFTRectifyingHomography gt_model{};
    gt_model.h7 = h7;
    gt_model.h8 = h8;

    // prepare inputs
	const auto input_size = num_squares * num_squares * kFeatureSize;
    std::vector<double> gt_rectified_features;
	gt_rectified_features.reserve(input_size);
    std::vector<double> features;
	features.reserve(input_size);
	std::cout << "h7 = " << h7 << "\n";
	std::cout << "h8 = " << h8 << "\n";
	std::cout << "Input features:\n";
    for (size_t i = 0; i < num_squares; i++)
    {
        for (size_t j = 0; j < num_squares; j++)
        {
			// generate rectified features
            double x = 0.5 * kSquareSize * (2 * i + 1);
			double y = 0.5 * kSquareSize * (2 * j + 1);
			double t = 0.0 + gaussianNoise(0.0, angle_noise);
			double s = kSquareSize;
            // push rectified features to ground-truth feature vector
            gt_rectified_features.push_back(x); // x-coordinate
            gt_rectified_features.push_back(y); // y-coordinate
            gt_rectified_features.push_back(t); // orientation
            gt_rectified_features.push_back(s); // scale
            // compute unrectified features
            gt_model.unrectifyFeature(x, y, t, s);
            // push unrectified features to vector
            features.push_back(x); // x-coordinate
            features.push_back(y); // y-coordinate
            features.push_back(t); // orientation
            features.push_back(s); // scale
			// print unrectified features
			std::cout << "\t# " << (i * num_squares + j) << ": (" << x << ", " << y << ", " << (180.0 * M_1_PI * t) << ", " << s << ")\n";
        }
    }
	std::cout << std::endl;

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

    cv::Mat sample_set(num_features, kFeatureSize, CV_64F, &features[0]);

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
    Eigen::Matrix3d H = model.getHomography();
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
	model.unrectifyPoint(vp1);
	model.unrectifyPoint(vp2);
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
	H /= H(2,2);
    std::cout << "\nFinal model:\n"
			  << "\nh7: " << model.h7 << ", h8: " << model.h8 << ", alpha: " << model.alpha << "\n"
			  << "\nx0: " << model.x0 << ", y0: " << model.y0 << ", s: " << model.s << "\n"
			  << "\nHomography:\n"
              << H << "\n\n"
			  << "first vanishing point (unnormalized): " << vp1.transpose() << "\n"
			  << "second vanishing point (unnormalized): " << vp2.transpose() << "\n"
			  << "Number of scale-inliers: " << num_scale_inliers << "\n"
			  << "Number of orientation-inliers: " << num_orientation_inliers << "\n"
              << "\n"
              << "Estimated vs Ground-Truth Rectified Feature:\n";

    for (size_t i = 0; i < num_features; i++)
    {
        double x = features[i * kFeatureSize + 0];
        double y = features[i * kFeatureSize + 1];
        double t = features[i * kFeatureSize + 2];
        double s = features[i * kFeatureSize + 3];
		model.normalizeFeature(x, y, s);
        model.rectifyFeature(x, y, t, s);
		model.denormalizeFeature(x, y, s);
        std::cout << "#" << i << ":\n"
				  << "\tScale-inlier: " << (scale_inliers[i] ? "true" : "false") << ", "
				  << "Orientation-inlier: " << (orientation_inliers[i] ? "true" : "false") << "\n"
                  << "\tEST: (" << x << ", " << y << ", " << (180.0 * M_1_PI * t) << ", " << s << ")\n";
        x = gt_rectified_features[i * kFeatureSize + 0];
        y = gt_rectified_features[i * kFeatureSize + 1];
        t = gt_rectified_features[i * kFeatureSize + 2];
        s = gt_rectified_features[i * kFeatureSize + 3];
        std::cout << "\tGT:  (" << x << ", " << y << ", " << t << ", " << s << ")\n"; 
    }

    return 0;
}
