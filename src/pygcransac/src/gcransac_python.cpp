#include "gcransac_python.h"
#include "GCRANSAC.h"
#include "types.h"
#include "neighborhood/grid_neighborhood_graph.h"
#include "preemption/preemption_sprt.h"
#include "inlier_selectors/empty_inlier_selector.h"

#include <vector>
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>
#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

using namespace gcransac;

int findRectifyingHomographyScaleOnly_(
	std::vector<double>& features,	// input SIFT features
	std::vector<double>& weights,	// input SIFT feature weights
	double scale_residual_thresh, // threshold for inlier selection
	double spatial_coherence_weight,
	size_t min_iteration_number,
	size_t max_iteration_number,
	size_t max_local_optimization_number,
	sampler::SamplerType sampler_type,
	std::vector<bool>& inliers,	// output inlier boolean mask
	std::vector<double>& homography	// output homography
)
{
	using namespace neighborhood;
	using ModelEstimator = utils::ScaleBasedRectifyingHomographyEstimator;

	constexpr size_t kFeatureSize = 3;	// each SIFT feature contains a 2D-coordinate and scale
	constexpr size_t kResidualDim = 1;

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
	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[0][pt_idx]] = true;
	}

	return num_inliers;
}

int findRectifyingHomographySIFT_(
	std::vector<double>& features, // input SIFT features
	std::vector<double>& weights, // input SIFT feature weights
	double scale_residual_thresh, // threshold for inlier selection
	double orientation_residual_thresh,
	double spatial_coherence_weight,
	size_t min_iteration_number,
	size_t max_iteration_number,
	size_t max_local_optimization_number,
	sampler::SamplerType sampler_type,
	std::vector<bool>& scale_inliers, // output scale inlier boolean mask
	std::vector<bool>& orientation_inliers,	// output orientation inlier boolean mask
	std::vector<double>& homography, // output estimated homography
	std::vector<double>& vanishing_points // output vanishing points corresponsing to the estimated homography 
)
{
	using namespace neighborhood;
	using ModelEstimator = utils::SIFTBasedRectifyingHomographyEstimator;

	constexpr size_t kFeatureSize = 4;
	constexpr size_t kResidualDim = 2;
	
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

	// TODO
	return static_cast<int>(num_scale_inliers + num_orientation_inliers);
}
