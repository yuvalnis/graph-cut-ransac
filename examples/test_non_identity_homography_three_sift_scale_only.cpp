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
	if (argc < 3 || argc > 6)
	{	
		std::cout << "Usage: " << argv[0]
				  << " scale_thresh"
				  << " [num_of_squares]"
				  << " [h7]"
				  << " [h8]"
				  << " [scale_noise]"
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

	double scale_noise = 0.0;
	if (argc > arg_count)
	{
		scale_noise = std::stod(argv[arg_count++]);
	}

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
			const double s_inflation = 1.0 + gaussianNoise(0.0, scale_noise);
			double s = kSquareSize * s_inflation;
            // push rectified features to ground-truth feature vector
            gt_rectified_features.push_back(x); // x-coordinate
            gt_rectified_features.push_back(y); // y-coordinate
            gt_rectified_features.push_back(s); // scale
            // compute unrectified features
			double dummy = 0.0;
            gt_model.unrectifyFeature(x, y, dummy, s);
            // push unrectified features to vector
            features.push_back(x); // x-coordinate
            features.push_back(y); // y-coordinate
            features.push_back(s); // scale
			// print unrectified features
			std::cout << "\t# " << (i * num_squares + j)
					  << ": (" << x << ", " << y << ", " << s << "), "
					  << "scale inflated by " << s_inflation << " due to noise.\n";
        }
    }
	std::cout << std::endl;

    std::vector<bool> inliers(num_squares * num_squares);
	std::vector<double> weights(num_squares * num_squares, 1.0);
    std::vector<double> homography(9);

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
	// create a boolean mask of the scale inliers
    inliers.resize(num_features, false);
	const auto num_inliers = statistics.inliers[0].size();
	for (size_t pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[0][pt_idx]] = true;
	}
    // end of block of code which goes in gcransac_python.cpp
	H /= H(2,2);
    std::cout << "\nFinal model:\n"
			  << "\nh7: " << model.h7 << ", h8: " << model.h8 << ", alpha: " << model.alpha << "\n"
			  << "\nx0: " << model.x0 << ", y0: " << model.y0 << ", s: " << model.s << "\n"
			  << "\nHomography:\n"
              << H << "\n\n"
			  << "Number of scale-inliers: " << num_inliers << "\n"
              << "\n"
              << "Estimated vs Ground-Truth Rectified Feature:\n";

    for (size_t i = 0; i < num_features; i++)
    {
        double x = features[i * kFeatureSize + 0];
        double y = features[i * kFeatureSize + 1];
        double s = features[i * kFeatureSize + 2];
		model.normalizeFeature(x, y, s);
		double dummy = 0.0;
        model.rectifyFeature(x, y, dummy, s);
		model.denormalizeFeature(x, y, s);
        std::cout << "#" << i << ":\n"
				  << "\tScale-inlier: " << (inliers[i] ? "true" : "false") << "\n"
                  << "\tEST: (" << x << ", " << y << ", " << s << ")\n";
        x = gt_rectified_features[i * kFeatureSize + 0];
        y = gt_rectified_features[i * kFeatureSize + 1];
        s = gt_rectified_features[i * kFeatureSize + 2];
        std::cout << "\tGT:  (" << x << ", " << y << ", " << s << ")\n"; 
    }

    return 0;
}
