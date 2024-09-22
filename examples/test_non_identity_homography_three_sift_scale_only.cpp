#include "gcransac_python.h"
#include "model.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <random>
#include <stdlib.h>

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

	std::vector<double> weights(num_squares * num_squares, 1.0);
    std::vector<bool> inliers(num_squares * num_squares);
    std::vector<double> homography(9);

	findRectifyingHomographyScaleOnly_(
		features, weights, scale_residual_thresh,
		spatial_coherence_weight,
		min_iteration_number, max_iteration_number,
		max_local_optimization_number, inliers, homography, /*verbose_level=*/2
	);

    return 0;
}
