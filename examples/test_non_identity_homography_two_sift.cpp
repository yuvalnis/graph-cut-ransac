#include "gcransac_python.h"
#include "model.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <random>
#include <stdlib.h>

using namespace gcransac;

constexpr size_t kFeatureSize = 4;
constexpr double kSquareSize = 40.0;

double gaussianNoise(double mean, double stddev)
{
	static std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, stddev);
    return distribution(generator);
}

bool coinFlip()
{
	std::random_device rd;  // Seed source
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_int_distribution<> dist(0, 1); // Distribution for {0, 1}
	return dist(gen);
}

int main(int argc, char* argv[])
{
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
    std::vector<double> scale_features;
    std::vector<double> orientation_features;
	scale_features.reserve(input_size);
	orientation_features.reserve(input_size);
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
			double t = (coinFlip() ? 0.0 : M_PI_2) + gaussianNoise(0.0, angle_noise);
			double s = kSquareSize;
            // compute unrectified features
			s = gt_model.unrectifiedScale(x, y, s);
			t = gt_model.unrectifiedAngle(x, y, t);
            gt_model.unrectifyPoint(x, y);
            // push unrectified features to vector
            scale_features.push_back(x); // x-coordinate
            scale_features.push_back(y); // y-coordinate
            scale_features.push_back(s); // scale
            orientation_features.push_back(x); // x-coordinate
            orientation_features.push_back(y); // y-coordinate
            orientation_features.push_back(t); // orientation
			// print unrectified features
			std::cout << "\t# " << (i * num_squares + j) << ": (" << x << ", " << y << ", " << (180.0 * M_1_PI * t) << ", " << s << ")\n";
        }
    }
	std::cout << std::endl;

    std::vector<bool> scale_inliers(num_squares * num_squares);
	std::vector<bool> orientation_inliers(num_squares * num_squares);
    std::vector<double> homography(9);
	std::vector<double> vanishing_points(6);
	gcransac::SIFTRectifyingHomography model;

    findRectifyingHomographySIFT_(
		scale_features, orientation_features,
		scale_residual_thresh, orientation_residual_thresh,
		spatial_coherence_weight,
		min_iteration_number, max_iteration_number,
		max_local_optimization_number,
		scale_inliers, orientation_inliers,
		homography, vanishing_points, model,
		/*verbose_level=*/2
	);

    return 0;
}
