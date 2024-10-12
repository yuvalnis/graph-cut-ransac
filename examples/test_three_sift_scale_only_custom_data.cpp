#include "gcransac_python.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <random>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <algorithm>

constexpr size_t kFeatureSize{3};
constexpr size_t kMaxEntriesToPrint{10};

size_t parse_input(const std::string& input, size_t block_size,
				   std::vector<double>& result
)
{
    // Remove curly braces from the input string
    std::string cleaned_input = input;
    cleaned_input.erase(std::remove(cleaned_input.begin(), cleaned_input.end(), '{'), cleaned_input.end());
    cleaned_input.erase(std::remove(cleaned_input.begin(), cleaned_input.end(), '}'), cleaned_input.end());

    // Create a string stream to process the input
    std::stringstream ss(cleaned_input);
    std::string temp;

    // Split the input by commas and convert to double
	result.clear();
    while (std::getline(ss, temp, ',')) {
        result.push_back(std::stod(temp));  // Convert each part to double
    }

	// Check the result is the correct size
	if (result.size() % block_size != 0)
	{
		std::cout << "Error: the size of the input vector ("
				  << result.size() << ") is not a multiple of "
				  << block_size << ".\n";
		return 0;
	}
	const size_t n_blocks = result.size() / block_size;

    // Output the parsed vector
    std::cout << n_blocks << "-by-" << block_size << " matrix:\n";
	size_t i = 0;
	for (; i < std::min(n_blocks, kMaxEntriesToPrint); i++)
	{
		for (size_t j = 0; j < block_size; j++)
		{
			std::cout << result.at(i * block_size + j) << " ";
		}
		std::cout << "\n";
	}
	if (i < n_blocks)
	{
		std::cout << "...\n";
	}
    std::cout << std::endl;

	return n_blocks;
}

int main(int argc, char* argv[])
{
	if (argc != 3)
	{	
		std::cout << "Usage: " << argv[0]
				  << " scale_thresh"
				  << " features"
				  << "\n";
		return -1;
	}
	
	double scale_residual_thresh = std::stod(argv[1]);
	if (scale_residual_thresh < 1.0)
	{
		std::cout << "Error: scale threshold must be non-negative\n";
		return -1;
	}
	scale_residual_thresh = std::log(scale_residual_thresh);
	std::cout << "Scale-residual threshold: " << scale_residual_thresh << "\n";

	// Parse the input features into a vector of doubles
	std::cout << "Parsing features...\n";
	std::string input = argv[2];
    std::vector<double> features;
	const auto n_features = parse_input(input, kFeatureSize, features);
	if (n_features == 0)
	{
		return -1;
	}

	const double spatial_coherence_weight = 0;
	const size_t min_iteration_number = 10000;
	const size_t max_iteration_number = 10000;
	const size_t max_local_optimization_number = 50;

    std::vector<bool> inliers(n_features);
    std::vector<double> homography(9);
	gcransac::SIFTRectifyingHomography model;

	findRectifyingHomographyScaleOnly_(
		features, scale_residual_thresh,
		spatial_coherence_weight,
		min_iteration_number, max_iteration_number,
		max_local_optimization_number, inliers, homography, model,
		/*verbose_level=*/1
	);

    return 0;
}
