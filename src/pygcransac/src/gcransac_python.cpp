#include "gcransac_python.h"
#include "GCRANSAC.h"
#include "types.h"
#include "neighborhood/grid_neighborhood_graph.h"
#include "preemption/preemption_sprt.h"
#include "inlier_selectors/empty_inlier_selector.h"

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>
#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

using namespace gcransac;

bool validVectorSize(const std::vector<double>& vec, size_t col_size)
{
	if (vec.empty() || vec.size() % col_size != 0)
	{
		fprintf(stderr,
			"The container of features should have a non-zero size which "
			"is a multiple of %lu. Its size is %lu.\n",
			col_size, vec.size()
		);
		return false;
	}
	return true;
}

int findRectifyingHomographyScaleOnly_(
	const std::vector<double>& features,
	double scale_residual_thresh,
	double spatial_coherence_weight,
	size_t min_iteration_number,
	size_t max_iteration_number,
	size_t max_local_optimization_number,
	std::vector<bool>& inliers,
	std::vector<double>& homography,
	unsigned int verbose_level
)
{
	using namespace neighborhood;
	using ModelEstimator = utils::ScaleBasedRectifyingHomographyEstimator;

	constexpr size_t kFeatureSize = 3;	// each SIFT feature contains a 2D-coordinate and scale

	if (!validVectorSize(features, kFeatureSize))
	{
		return 0;
	}
	const auto num_features = features.size() / kFeatureSize; 

	std::unique_ptr<cv::Mat> features_ptr = std::make_unique<cv::Mat>(
		num_features, kFeatureSize, CV_64F
	);
	std::memcpy(features_ptr->data, features.data(), features.size() * sizeof(double));

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
	GCRANSAC<ModelEstimator, NeighborhoodGraph>::DataType data{
		std::move(features_ptr)
	};
	gcransac.run(
		data, estimator, neighborhood_graph.get(), model,
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
	const auto num_inliers = statistics.inliers[0].size();
	for (size_t pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[0][pt_idx]] = true;
	}

	if (verbose_level > 0)
	{
		std::cout << "\nFinal model:\n"
				  << "\nh7: " << model.h7 << ", h8: " << model.h8 << ", alpha: " << model.alpha << "\n"
				  << "\nx0: " << model.x0 << ", y0: " << model.y0 << ", s: " << model.s << "\n"
				  << "\nHomography:\n"
				  << H << "\n\n"
				  << "Number of scale-inliers: " << num_inliers << "\n"
				  << "\n"
				  << "Estimated rectified features:\n";

		if (verbose_level > 1)
		{
			for (size_t i = 0; i < num_features; i++)
			{
				double x = data[0]->at<double>(i, 0);
				double y = data[0]->at<double>(i, 1);
				double s = data[0]->at<double>(i, 2);
				model.normalizeFeature(x, y, s);
				model.rectifiedScale(x, y, s);
				model.rectifyPoint(x, y);
				model.denormalizeFeature(x, y, s);
				std::cout << "#" << i << ":\n"
						<< "\tInlier: " << (inliers[i] ? "true" : "false") << "\n"
						<< "\tRectified feature: (" << x << ", " << y << ", " << s << ")\n";
			}
		}
	}

	return num_inliers;
}

int findRectifyingHomographySIFT_(
	const std::vector<double>& scale_features,
	const std::vector<double>& orientation_features,
	double scale_residual_thresh,
	double orientation_residual_thresh,
	double spatial_coherence_weight,
	size_t min_iteration_number,
	size_t max_iteration_number,
	size_t max_local_optimization_number,
	std::vector<bool>& scale_inliers, // output scale inlier boolean mask
	std::vector<bool>& orientation_inliers,	// output orientation inlier boolean mask
	std::vector<double>& homography, // output estimated homography
	std::vector<double>& vanishing_points, // output vanishing points corresponsing to the estimated homography 
	unsigned int verbose_level
)
{
	using namespace neighborhood;
	using ModelEstimator = utils::SIFTBasedRectifyingHomographyEstimator;

	constexpr size_t kFeatureSize = 3;
	
	if (!validVectorSize(scale_features, kFeatureSize) ||
	    !validVectorSize(orientation_features, kFeatureSize)
	)
	{
		return 0;
	}

	const auto num_scale_features = scale_features.size() / kFeatureSize;
	std::unique_ptr<cv::Mat> scale_features_ptr = std::make_unique<cv::Mat>(
		num_scale_features, kFeatureSize, CV_64F
	);
	std::memcpy(scale_features_ptr->data, scale_features.data(),
				scale_features.size() * sizeof(double));

	const auto num_orientation_features = orientation_features.size() / kFeatureSize;
	std::unique_ptr<cv::Mat> orientation_features_ptr = std::make_unique<cv::Mat>(
		num_orientation_features, kFeatureSize, CV_64F
	);
	std::memcpy(orientation_features_ptr->data, orientation_features.data(),
				orientation_features.size() * sizeof(double));

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
	gcransac.settings.threshold(1) = orientation_residual_thresh;
	gcransac.settings.do_local_optimization = true; 
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight;
	gcransac.settings.min_iteration_number = min_iteration_number;
	gcransac.settings.max_iteration_number = max_iteration_number;
	gcransac.settings.max_local_optimization_number = max_local_optimization_number;
	gcransac.settings.do_final_iterated_least_squares = true;

	SIFTRectifyingHomography model;
	GCRANSAC<ModelEstimator, NeighborhoodGraph>::DataType data{
		std::move(scale_features_ptr), std::move(orientation_features_ptr)
	};
	gcransac.run(
		data, estimator, neighborhood_graph.get(), model,
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
    scale_inliers.resize(num_scale_features, false);
	const auto num_scale_inliers = statistics.inliers[0].size();
	for (size_t pt_idx = 0; pt_idx < num_scale_inliers; ++pt_idx) {
		scale_inliers[statistics.inliers[0][pt_idx]] = true;
	}
	// create a boolean mask of the orientation inliers
	orientation_inliers.resize(num_orientation_features, false);
	const auto num_orientation_inliers = statistics.inliers[1].size();
	for (size_t pt_idx = 0; pt_idx < num_orientation_inliers; ++pt_idx) {
		orientation_inliers[statistics.inliers[1][pt_idx]] = true;
	}

	if (verbose_level > 0)
	{
		if (std::abs(vp1[2]) > 1e-6)
		{
			vp1 /= vp1[2];
		}
		if (std::abs(vp2[2]) > 1e-6)
		{
			vp2 /= vp2[2];
		}
		std::cout << "\nFinal model:\n"
				  << "\nh7: " << model.h7 << ", h8: " << model.h8 << ", alpha: " << model.alpha << "\n"
				  << "\nx0: " << model.x0 << ", y0: " << model.y0 << ", s: " << model.s << "\n"
				  << "\nHomography:\n"
				  << H << "\n\n"
				  << "Rectified image vanishing point #1 direction: "
			  	  << (model.vanishing_point_dir1 * 180.0 * M_1_PI) << std::endl
				  << "Rectified image vanishing point #2 direction: "
			      << (model.vanishing_point_dir2 * 180.0 * M_1_PI) << std::endl
				  << "first vanishing point (normalized): " << vp1.transpose() << "\n"
				  << "second vanishing point (normalized): " << vp2.transpose() << "\n"
				  << "Number of scale-inliers: " << num_scale_inliers << "\n"
				  << "Number of orientation-inliers: " << num_orientation_inliers
				  << "\n\n";

		if (verbose_level > 1)
		{
			// print rectified scale features
			std::cout << "Estimated rectified scale features:\n";
			for (size_t i = 0; i < num_scale_features; i++)
			{
				double x = data[0]->at<double>(i, 0);
				double y = data[0]->at<double>(i, 1);
				double s = data[0]->at<double>(i, 2);
				model.normalizeFeature(x, y, s);
				model.rectifiedScale(x, y, s);
				model.rectifyPoint(x, y);
				model.denormalizeFeature(x, y, s);
				std::cout << "#" << i << ":\n"
						  << "\tInlier: " << (scale_inliers[i] ? "true" : "false") << "\n"
						  << "\tRectified feature: (" << x << ", " << y << ", " << s << ")\n";
			}
			std::cout << std::endl;
			// print rectified orientation features
			std::cout << "Estimated rectified orientation features:\n";
			for (size_t i = 0; i < num_scale_features; i++)
			{
				double x = data[1]->at<double>(i, 0);
				double y = data[1]->at<double>(i, 1);
				double t = data[1]->at<double>(i, 2);
				model.normalize(x, y);
				model.rectifiedAngle(x, y, t);
				model.rectifyPoint(x, y);
				model.denormalize(x, y);
				std::cout << "#" << i << ":\n"
						  << "\tInlier: " << (orientation_inliers[i] ? "true" : "false") << "\n"
						  << "\tRectified feature: (" << x << ", " << y << ", " << (180.0 * M_1_PI * t) << ")\n";
			}
			std::cout << std::endl;
		}
	}

	// TODO
	return static_cast<int>(num_scale_inliers + num_orientation_inliers);
}
