#pragma once

#include <vector>
#include <cstddef>

int findRectifyingHomographyScaleOnly_(
	const std::vector<double>& features,
	double scale_residual_thresh,
	double spatial_coherence_weight,
	size_t min_iteration_number,
	size_t max_iteration_number,
	size_t max_local_optimization_number,
	std::vector<bool>& inliers,	// output inlier boolean mask
	std::vector<double>& homography,	// output homography
	unsigned int verbose_level = 0
);

int findRectifyingHomographySIFT_(
	const std::vector<double>& scale_features,
	const std::vector<double>& orientation_features,
	double scale_residual_thresh,
	double orientation_residual_thresh,
	double spatial_coherence_weight, // the spatial coherence weight used in the local optimization
	size_t min_iteration_number, // minimum iteration number. value equal or higher to 50 is recommended
	size_t max_iteration_number, // maximum iteration number. value equal or higher to 1000 is recommended
	size_t max_local_optimization_number, // the number of RANSAC iterations done in the local optimization
	std::vector<bool>& scale_inliers, // output scale inlier boolean mask
	std::vector<bool>& orientation_inliers,	// output orientation inlier boolean mask
	std::vector<double>& homography,	// output homography
	std::vector<double>& vanishing_points, // output vanishing points corresponsing to the estimated homography
	unsigned int verbose_level = 0
);
