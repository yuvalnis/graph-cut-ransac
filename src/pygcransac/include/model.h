// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include <Eigen/Eigen>
#include "math_utils.hpp"

namespace gcransac
{

struct NormalizingTransform
{
	double x0 = 0.0; // x-coordinate of the unnormalized sample set mean
	double y0 = 0.0; // y-coordinate of the unnormalized sample set mean
	double s = 1.0; // normalizing rescaling factor of sample set

	// (de-)normalizing methods for homogeneous coordinates

	void normalize(double& x, double& y, double w) const
	{
		// generally, the homogeneous coordinate can be different than 1 (w != 1)
		x = s * (x - x0 * w);
		y = s * (y - y0 * w);
	}

	void denormalize(double& x, double& y, double w) const
	{
		const auto inv_s = 1.0 / s;
		// generally, the homogeneous coordinate can be different than 1 (w != 1)
		x = inv_s * x + x0 * w;
		y = inv_s * y + y0 * w;
	}

	// (de-)normalzing methods for non-homogeneous coordinates

	void normalize(double& x, double& y) const
	{
		normalize(x, y, 1.0);
	}

	void denormalize(double& x, double& y) const
	{
		denormalize(x, y, 1.0);
	}

	// (de-)normalizing methods for homogeneous vectors

	void normalize(Eigen::Vector3d& p) const
	{
		normalize(p(0), p(1), p(2));
	}

	void denormalize(Eigen::Vector3d& p) const
	{
		denormalize(p(0), p(1), p(2));
	}

	// (de-)normalizing methods for scales

	void normalizeScale(double& scale) const
	{
		scale *= s;
	}

	void denormalizeScale(double& scale) const
	{
		scale /= s;
	}

	// (de-)normalizing methods for SIFT features

	void normalizeFeature(double& x, double& y, double& scale) const
	{
		normalizeScale(scale);
		Eigen::Vector3d point(x, y, 1.0);
		normalize(point);
		x = point(0) / point(2);
		y = point(1) / point(2);
	}

	void denormalizeFeature(double& x, double& y, double& scale) const
	{
		denormalizeScale(scale);
		Eigen::Vector3d point(x, y, 1.0);
		denormalize(point);
		x = point(0) / point(2);
		y = point(1) / point(2);
	}
};

struct RectifyingHomography : public NormalizingTransform
{
	// model parameters
	double h7 = 0.0;
	double h8 = 0.0;
	
	inline void rectifyPoint(Eigen::Vector3d& p) const
	{
		p(2) = -h7 * p(0) - h8 * p(1) + p(2);
	} 

	inline void unrectifyPoint(Eigen::Vector3d& p) const
	{
		// negating h7 and h8 is equivalent to inverting the warping
		// homography matrix in this case
		p(2) = h7 * p(0) + h8 * p(1) + p(2);
	}

	inline void rectifyPoint(double& x, double& y) const
	{
		Eigen::Vector3d point(x, y, 1.0);
		rectifyPoint(point);
		x = point(0) / point(2);
		y = point(1) / point(2);
	}

	inline void unrectifyPoint(double& x, double& y) const
	{
		Eigen::Vector3d point(x, y, 1.0);
		unrectifyPoint(point);
		x = point(0) / point(2);
		y = point(1) / point(2);
	}

	double rectifiedAngle(double x, double y, double angle) const
	{
		const auto ct = std::cos(angle);
   		const auto st = std::sin(angle);
		// negating h7 and h8 is equivalent to inverting the warping
		// homography matrix in this case
		const auto numer = (-x * st + y * ct) * h7 + st;
		const auto denom = (x * st - y * ct) * h8 + ct;
		return utils::clipAngle(std::atan2(numer, denom));
	}

	double unrectifiedAngle(double x, double y, double angle) const
	{
		const auto ct = std::cos(angle);
   		const auto st = std::sin(angle);
		const auto numer = (x * st - y * ct) * h7 + st;
		const auto denom = (-x * st + y * ct) * h8 + ct;
		return utils::clipAngle(std::atan2(numer, denom));
	}

	/// @brief Computes the local scale changed applied by the perspective
	/// warping homography
	/// @param udx x-coordinate in the unwarped image (rectified coordinate)
	/// @param udy y-coordinate in the unwarped image (rectified coordinate)
	/// @return A scalar representing the local scale change applied to a scale
	/// feature in the coordinates (udx, udy) in the unwarped image
	inline double localScalePerspectiveWarp(double udx, double udy) const
	{
		// The determinant of the perspective component of the homography at (x, y)
		return std::pow(h7 * udx + h8 * udy + 1.0, -3.0);
	}

	/// @brief Computes the local scale changed applied by the inverse of the
	/// perspective warping homography
	/// @param udx x-coordinate in the warped image
	/// @param udy y-coordinate in the warped image
	/// @return A scalar representing the local scale change applied to a scale
	/// feature in the coordinates (dx, dy) in the warped image
	inline double localScaleAffineRectification(double dx, double dy) const
	{
		// negating h7 and h8 is equivalent to inverting the warping
		// homography matrix in this case
		return std::pow(-h7 * dx - h8 * dy + 1.0, -3.0);
	}

	inline double rectifiedScale(double dx, double dy, double ds) const
	{
		return ds * localScaleAffineRectification(dx, dy);
	}

	inline double unrectifiedScale(double udx, double udy, double uds) const
	{
		return uds * localScalePerspectiveWarp(udx, udy);
	}

	Eigen::Matrix3d getHomography() const
	{
		// normalizing transform
		Eigen::Matrix3d N;
		N << s, 0, -s * x0,
			 0, s, -s * y0,
			 0, 0, 1;
		// homography in normalized coordinate space
		Eigen::Matrix3d H;
		H << 1,  0,  0,
			 0,  1,  0,
			 h7, h8, 1;
		// homography in unnormalized coordinate space
		Eigen::Matrix3d result = N.inverse() * H * N;
		return result / result(2, 2);
	}
};

struct ScaleBasedRectifyingHomography : virtual public RectifyingHomography
{
	// feature-class scale (varies between different classes of features)
	// this is the relative scale of the recitified features.
	double alpha{1.0};
};

struct OrientationBasedRectifyingHomography : virtual public RectifyingHomography
{
	// The direction (in radians) of the vanishing point used to estimate the 
	// model in the rectified image.
	double phi{0.0};
};

struct SIFTRectifyingHomography : public ScaleBasedRectifyingHomography, OrientationBasedRectifyingHomography
{
	// a model joining the scale- and orientation-based models
};

}
