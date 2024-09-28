#include "gcransac_python.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <random>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <algorithm>

constexpr size_t kFeatureSize{4};
constexpr size_t kMaxEntriesToPrint{10};

std::vector<double> parse_input(const std::string& input) {
    // Remove curly braces from the input string
    std::string cleaned_input = input;
    cleaned_input.erase(std::remove(cleaned_input.begin(), cleaned_input.end(), '{'), cleaned_input.end());
    cleaned_input.erase(std::remove(cleaned_input.begin(), cleaned_input.end(), '}'), cleaned_input.end());

    // Create a string stream to process the input
    std::stringstream ss(cleaned_input);
    std::string temp;
    std::vector<double> result;

    // Split the input by commas and convert to double
    while (std::getline(ss, temp, ',')) {
        result.push_back(std::stod(temp));  // Convert each part to double
    }

    return result;
}

int main(int argc, char* argv[])
{
	if (argc < 4 || argc > 5)
	{	
		std::cout << "Usage: " << argv[0]
				  << " scale_thresh"
				  << " orientation_thresh"
				  << " features"
				  << " [weights]"
				  << "\n";
		return -1;
	}
	
	double scale_residual_thresh = std::stod(argv[1]);
	if (scale_residual_thresh < 1.0)
	{
		std::cout << "Error: scale threshold must be larger than or equal to one\n";
		return -1;
	}
	scale_residual_thresh = std::log(scale_residual_thresh);
	std::cout << "Scale-residual threshold: " << scale_residual_thresh << "\n";

	double orientation_residual_thresh = std::stod(argv[2]);
	if (std::signbit(orientation_residual_thresh))
	{
		std::cout << "Error: orientation threshold must be non-negative\n";
		return -1;
	}
	std::cout << "Orientation-residual threshold: " << orientation_residual_thresh << "\n";
	orientation_residual_thresh = M_PI * orientation_residual_thresh / 180.0;

	// Parse the input features into a vector of doubles
	std::cout << "Parsing features...\n";
	std::string input = argv[3];
    std::vector<double> features = parse_input(input);
	if (features.size() % kFeatureSize != 0)
	{
		std::cout << "Error: the size of the input feature vector ("
				  << features.size() << ") is not a multiple of "
				  << kFeatureSize << ".\n";
		return -1;
	}
	const size_t n_features = features.size() / kFeatureSize;
    // Output the parsed vector
    std::cout << n_features << "-by-" << kFeatureSize << " feature matrix:\n";
	size_t i = 0;
	for (; i < std::min(n_features, kMaxEntriesToPrint); i++)
	{
		for (size_t j = 0; j < kFeatureSize; j++)
		{
			std::cout << features.at(i * kFeatureSize + j) << " ";
		}
		std::cout << "\n";
	}
	if (i < n_features)
	{
		std::cout << "...\n";
	}
    std::cout << std::endl;

	// Parse the input weights into a vector of doubles
	std::cout << "Parsing weights...\n";
	std::vector<double> weights{};
	weights.reserve(n_features);
	if (argc == 5)
	{
		input = argv[4];
    	weights = parse_input(input);
	}
	else
	{
		weights = std::vector<double>(n_features, 1.0);
	}
	if (weights.size() != n_features)
	{
		std::cout << "Error: the size of the input weights vector ("
				  << weights.size() << ") does not equal the number of features "
				  << kFeatureSize << ".\n";
		return -1;
	}
	// Output the parsed vector
    std::cout << n_features << "-by-" << kFeatureSize << " feature matrix:\n";
	i = 0;
	for (; i < std::min(n_features, kMaxEntriesToPrint); i++)
	{
		std::cout << weights.at(i) << "\n";
	}
	if (i < n_features)
	{
		std::cout << "...\n";
	}
    std::cout << std::endl;

	const double spatial_coherence_weight = 0;
	const size_t min_iteration_number = 10000;
	const size_t max_iteration_number = 10000;
	const size_t max_local_optimization_number = 50;

    std::vector<bool> scale_inliers(n_features);
    std::vector<bool> orientation_inliers(n_features);
    std::vector<double> homography(9);
	std::vector<double> vanishing_points(6);

	findRectifyingHomographySIFT_(
		features, weights, scale_residual_thresh, orientation_residual_thresh,
		spatial_coherence_weight, min_iteration_number, max_iteration_number,
		max_local_optimization_number, scale_inliers, orientation_inliers,
		homography, vanishing_points, /*verbose_level=*/1
	);

    return 0;
}
