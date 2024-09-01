// #pragma once

// #include "scoring_function.hpp"

// namespace gcransac
// {

// template<class _Estimator>
// class MAGSACScoringFunction : public ScoringFunction<_Estimator>
// {
// public:

// // Return the score of a model w.r.t. the data points and the threshold
// OLGA_INLINE Score getScore(const cv::Mat& points_, // The input data points
//     Model& model_, // The current model parameters
//     const _Estimator& estimator_, // The model estimator
//     const double threshold_, // The inlier-outlier threshold
//     std::vector<size_t>& inliers_, // The selected inliers
//     const Score& best_score_ = Score(), // The score of the current so-far-the-best model
//     const bool store_inliers_ = true, // A flag to decide if the inliers should be stored
//     const std::vector<const std::vector<size_t>*> *index_sets = nullptr) const // Index sets to be verified
// {
//     constexpr size_t _DimensionNumber = 4;

//     double increasedThreshold = threshold_;

//     // The degrees of freedom of the data from which the model is estimated.
//     // E.g., for models coming from point correspondences (x1,y1,x2,y2), it is 4.
//     // A 0.99 quantile of the Chi^2-distribution to convert sigma values to residuals
//     constexpr double k =
//         _DimensionNumber == 2 ?
//         3.03 : 3.64;
//     // A multiplier to convert residual values to sigmas
//     constexpr double threshold_to_sigma_multiplier = 1.0 / k;
//     // Calculating k^2 / 2 which will be used for the estimation and, 
//     // due to being constant, it is better to calculate it a priori.
//     constexpr double squared_k_per_2 = k * k / 2.0;
//     // Calculating (DoF - 1) / 2 which will be used for the estimation and, 
//     // due to being constant, it is better to calculate it a priori.
//     constexpr double dof_minus_one_per_two = (degrees_of_freedom - 1.0) / 2.0;
//     // TODO: check
//     constexpr double C = 0.25;
//     // The size of a minimal sample used for the estimation
//     constexpr size_t sample_size = _Estimator::sampleSize();
//     // Calculating 2^(DoF - 1) which will be used for the estimation and, 
//     // due to being constant, it is better to calculate it a priori.
//     static const double two_ad_dof = std::pow(2.0, dof_minus_one_per_two);
//     // Calculating C * 2^(DoF - 1) which will be used for the estimation and, 
//     // due to being constant, it is better to calculate it a priori.
//     static const double C_times_two_ad_dof = C * two_ad_dof;
//     // Calculating the gamma value of (DoF - 1) / 2 which will be used for the estimation and, 
//     // due to being constant, it is better to calculate it a priori.
//     static const double gamma_value = tgamma(dof_minus_one_per_two);
//     // Calculating the upper incomplete gamma value of (DoF - 1) / 2 with k^2 / 2.
//     constexpr double gamma_k = 0.0036572608340910764;
//     // Calculating the lower incomplete gamma value of (DoF - 1) / 2 which will be used for the estimation and, 
//     // due to being constant, it is better to calculate it a priori.
//     static const double gamma_difference = gamma_value - gamma_k;
//     // Divide C * 2^(DoF - 1) by \sigma_{max} a priori
//     const double one_over_sigma = C_times_two_ad_dof / increasedThreshold;
//     // Calculate the weight of a point with 0 residual (i.e., fitting perfectly) a priori
//     const double weight_zero = one_over_sigma * gamma_difference;

//     // Iterate through all points, calculate the squared_residualsand store the points as inliers if needed.
//     Score score;
//     double residual = 0;
//     const size_t& point_number = points_.rows;
//     if (store_inliers_)
//     {
//         inliers_.reserve(point_number);
//         inliers_.clear();
//     }
//     for (int point_idx = 0; point_idx < point_number; point_idx += 1)
//     {
//         // Calculate the point-to-model residual
//         residual =
//             estimator_.residual(points_.row(point_idx),
//                 model_.descriptor);

//         if (residual > increasedThreshold)
//             continue;

//         // If the residual is ~0, the point fits perfectly and it is handled differently
//         double weight = 0.0;
//         if (residual < std::numeric_limits<double>::epsilon())
//             weight = weight_zero;
//         else
//         {
//             // Calculate the squared residual
//             const double squared_residual = residual * residual;
//             // Get the position of the gamma value in the lookup table
//             size_t x = round(precision_of_stored_gammas * squared_residual / squared_sigma_max_2);

//             // If the sought gamma value is not stored in the lookup, return the closest element
//             if (stored_gamma_number < x)
//                 x = stored_gamma_number;

//             // Calculate the weight of the point
//             weight = one_over_sigma * (stored_gamma_values[x] - gamma_k);
//         }
//         score.value += weight / weight_zero;

//         if (residual > threshold_)
//             continue;

//         ++score.inlier_number;

//         if (store_inliers_)
//             inliers_.emplace_back(point_idx);
//     }

//     return score;
// }

// };

// }
