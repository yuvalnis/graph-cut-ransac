#include <stdexcept>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "gcransac_python.h"

namespace py = pybind11;

void py2cpp_vector(py::array_t<double>& py_vec, std::vector<double>& cpp_vec)
{
	py::buffer_info vec_buff = py_vec.request();
	const auto* vec_ptr = static_cast<double*>(vec_buff.ptr);
	cpp_vec.assign(vec_ptr, vec_ptr + vec_buff.size);
}

py::tuple findRectifyingHomographyScaleOnly(
	py::array_t<double> features,
	// py::array_t<double> weights,
	double scale_residual_thresh,
	double spatial_coherence_weight,
	size_t min_iteration_number,
	size_t max_iteration_number,
	size_t max_local_optimization_number
)
{
	constexpr size_t kFeatureSize = 3;
	constexpr size_t kNumMinFeatures = 3;

	py::buffer_info features_buff = features.request();
	// py::buffer_info weights_buff = weights.request();

	if (features_buff.ndim != 2)
	{
		throw std::invalid_argument("Number of dimensions must be 2.");
	}

	const size_t num_features = features_buff.shape[0];
	const size_t feature_size = features_buff.shape[1];
	// validate dimenstions of input
	if (num_features < kNumMinFeatures || feature_size != kFeatureSize)
	{
		std::stringstream error_msg;
		error_msg << "Features should be an array with " << kFeatureSize
				  << " columns and at least " << kNumMinFeatures << " rows."
				  << " It has " << feature_size << " columns and "
				  << num_features << " rows.";
		throw std::invalid_argument(error_msg.str());
	}
	// construct C++ vector for features and weights from python array
	std::vector<double> cpp_features;
	py2cpp_vector(features, cpp_features);
	// const auto* weights_ptr = static_cast<double*>(weights_buff.ptr);
	// std::vector<double> cpp_weights;
	// cpp_weights.assign(weights_ptr, weights_ptr + weights_buff.size);

	std::vector<double> cpp_homography(9);
    std::vector<bool> cpp_inliers(num_features);

	const auto num_inliers = findRectifyingHomographyScaleOnly_(
		cpp_features,
		// cpp_weights,
		scale_residual_thresh,
		spatial_coherence_weight,
		min_iteration_number,
		max_iteration_number,
		max_local_optimization_number,
		cpp_inliers,
		cpp_homography
	);
	// construct python array for inliers from C++ vector
	py::array_t<bool> inliers = py::array_t<bool>(num_features);
    py::buffer_info inliers_buff = inliers.request();
    auto *inliers_ptr = static_cast<bool*>(inliers_buff.ptr);
    for (size_t i = 0; i < num_features; i++)
	{
    	inliers_ptr[i] = cpp_inliers[i];
	}

    if (num_inliers == 0)
	{
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers);
    }
	// construct python array for homography for C++ vector
    py::array_t<double> homography = py::array_t<double>({3, 3});
    py::buffer_info homography_buff = homography.request();
    auto *homography_ptr = static_cast<double*>(homography_buff.ptr);
    for (size_t i = 0; i < 9; i++)
	{
		homography_ptr[i] = cpp_homography[i];
	}

    return py::make_tuple(homography, inliers);
}

py::tuple findRectifyingHomographySIFT(
	py::array_t<double> scale_features,
	py::array_t<double> orientation_features,
	// py::array_t<double> weights,
	double scale_residual_thresh,
	double orientation_residual_thresh,
	double spatial_coherence_weight,
	size_t min_iteration_number,
	size_t max_iteration_number,
	size_t max_local_optimization_number
)
{
	constexpr size_t kFeatureSize = 3;
	constexpr size_t kNumMinFeatures = 2;

	py::buffer_info scale_features_buff = scale_features.request();
	py::buffer_info orientation_features_buff = orientation_features.request();
	// py::buffer_info weights_buff = weights.request();

	if (scale_features_buff.ndim != 2 || orientation_features_buff.ndim != 2)
	{
		throw std::invalid_argument("Number of dimensions must be 2.");
	}

	const size_t num_scale_features = scale_features_buff.shape[0];
	const size_t num_orientation_features = orientation_features_buff.shape[0];
	// validate dimensions of input
	if (num_scale_features < kNumMinFeatures || scale_features_buff.shape[1] != kFeatureSize)
	{
		std::stringstream error_msg;
		error_msg << "Scale features should be an array with " << kFeatureSize
				  << " columns and at least " << kNumMinFeatures << " rows."
				  << " It has " << scale_features_buff.shape[1] << " columns and "
				  << num_scale_features << " rows.";
		throw std::invalid_argument(error_msg.str());
	}
	if (num_orientation_features < kNumMinFeatures || orientation_features_buff.shape[1] != kFeatureSize)
	{
		std::stringstream error_msg;
		error_msg << "Orientation features should be an array with " << kFeatureSize
				  << " columns and at least " << kNumMinFeatures << " rows."
				  << " It has " << orientation_features_buff.shape[1] << " columns and "
				  << num_orientation_features << " rows.";
		throw std::invalid_argument(error_msg.str());
	}
	// if (weights_buff.shape[0] != num_features)
	// {
	// 	std::stringstream error_msg;
	// 	error_msg << "Weights should have the same length as the number of features.\n"
	// 			  << "There are " << weights_buff.shape[0] << " weights and "
	// 			  << num_features << " features.\n";
	// 	throw std::invalid_argument(error_msg.str());
	// }
	// construct C++ vector for features and weights from python array
	const auto* scale_features_ptr = static_cast<double*>(
		scale_features_buff.ptr
	);
	std::vector<double> cpp_scale_features;
	cpp_scale_features.assign(
		scale_features_ptr,
		scale_features_ptr + scale_features_buff.size
	);

	const auto* orientation_features_ptr = static_cast<double*>(
		orientation_features_buff.ptr
	);
	std::vector<double> cpp_orientation_features;
	cpp_orientation_features.assign(
		orientation_features_ptr,
		orientation_features_ptr + orientation_features_buff.size
	);

	// const auto* weights_ptr = static_cast<double*>(weights_buff.ptr);
	// std::vector<double> cpp_weights;
	// cpp_weights.assign(weights_ptr, weights_ptr + weights_buff.size);

	std::vector<double> cpp_homography(9);
	std::vector<double> cpp_vanishing_points(6);
    std::vector<bool> cpp_scale_inliers(num_scale_features);
	std::vector<bool> cpp_orientation_inliers(num_orientation_features);

	const auto num_inliers = findRectifyingHomographySIFT_(
		cpp_scale_features,
		cpp_orientation_features,
		// cpp_weights,
		scale_residual_thresh,
		orientation_residual_thresh,
		spatial_coherence_weight,
		min_iteration_number,
		max_iteration_number,
		max_local_optimization_number,
		cpp_scale_inliers,
		cpp_orientation_inliers,
		cpp_homography,
		cpp_vanishing_points
	);
	// construct python array for scale inliers from C++ vector
	py::array_t<bool> scale_inliers = py::array_t<bool>(num_scale_features);
    py::buffer_info scale_inliers_buff = scale_inliers.request();
    auto *scale_inliers_ptr = static_cast<bool*>(scale_inliers_buff.ptr);
    for (size_t i = 0; i < num_scale_features; i++)
	{
    	scale_inliers_ptr[i] = cpp_scale_inliers[i];
	}
	// construct python array for orientation inliers from C++ vector
	py::array_t<bool> orientation_inliers = py::array_t<bool>(num_orientation_features);
    py::buffer_info orientation_inliers_buff = orientation_inliers.request();
    auto *orientation_inliers_ptr = static_cast<bool*>(orientation_inliers_buff.ptr);
    for (size_t i = 0; i < num_orientation_features; i++)
	{
    	orientation_inliers_ptr[i] = cpp_orientation_inliers[i];
	}

    if (num_inliers == 0)
	{
		return py::make_tuple(
			pybind11::cast<pybind11::none>(Py_None),
			scale_inliers,
			orientation_inliers,
			pybind11::cast<pybind11::none>(Py_None)
		);
    }
	// construct python array for homography for C++ vector
    py::array_t<double> homography = py::array_t<double>({3, 3});
    py::buffer_info homography_buff = homography.request();
    auto *homography_ptr = static_cast<double*>(homography_buff.ptr);
    for (size_t i = 0; i < 9; i++)
	{
		homography_ptr[i] = cpp_homography[i];
	}
	// construct python array for vanishing points for C++ vector
	py::array_t<double> vanishing_points = py::array_t<double>({3, 2});
	py::buffer_info vps_buff = vanishing_points.request();
	auto *vps_ptr = static_cast<double*>(vps_buff.ptr);
	for (size_t i = 0; i < 6; i++)
	{
		vps_ptr[i] = cpp_vanishing_points[i];
	}

    return py::make_tuple(homography, scale_inliers, orientation_inliers, vanishing_points);
}

PYBIND11_PLUGIN(pygcransac) {

    py::module m("pygcransac", R"doc(
        Python module
        -----------------------
        .. currentmodule:: pygcransac
        .. autosummary::
           :toctree: _generate

			findRectifyingHomographyScaleOnly,
			findRectifyingHomographySIFT

    )doc");

	// py::enum_<gcransac::sampler::SamplerType>(m, "SamplerType")
	// 	.value("UNIFORM", gcransac::sampler::SamplerType::Uniform)
	// 	.value("PROSAC", gcransac::sampler::SamplerType::ProSaC)
	// 	.value("PROGRESSIVE_NAPSAC", gcransac::sampler::SamplerType::ProgressiveNapsac)
	// 	.value("IMPORTANCE", gcransac::sampler::SamplerType::Importance)
	// 	.value("ADAPTIVE_REORDERING", gcransac::sampler::SamplerType::AdaptiveReordering);

	m.def("findRectifyingHomographyScaleOnly", &findRectifyingHomographyScaleOnly, R"doc(some doc)doc",
		py::arg("features"),
		// py::arg("weights"),
		py::arg("scale_residual_thresh"),
		py::arg("spatial_coherence_weight") = 0.0,
		py::arg("min_iteration_number") = 10000,
		py::arg("max_iteration_number") = 10000,
		py::arg("max_local_optimization_number") = 50
	);

	m.def("findRectifyingHomographySIFT", &findRectifyingHomographySIFT, R"doc(some doc)doc",
		py::arg("scale_features"),
		py::arg("orientation_features"),
		// py::arg("weights"),
		py::arg("scale_residual_thresh"),
		py::arg("orientation_residual_thresh"),
		py::arg("spatial_coherence_weight") = 0.0,
		py::arg("min_iteration_number") = 10000,
		py::arg("max_iteration_number") = 10000,
		py::arg("max_local_optimization_number") = 50
	);

  	return m.ptr();
}
