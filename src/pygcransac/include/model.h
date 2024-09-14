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

// #include <opencv2/core/core.hpp>
// #include <vector>
#include <Eigen/Eigen>
// #include "estimators/estimator.h"

namespace gcransac
{

// class Model
// {
// public:
// 	Eigen::MatrixXd descriptor; // The descriptor of the current model
// 	Model(const Eigen::MatrixXd &descriptor_) : descriptor(descriptor_) {}
// 	Model() {}
// };

// class RigidTransformation : public Model
// {
// public:
// 	RigidTransformation() : Model(Eigen::MatrixXd(4, 4)) {}
// 	RigidTransformation(const RigidTransformation& other)
// 	{
// 		descriptor = other.descriptor;
// 	}
// };

// class Line2D : public Model
// {
// public:
// 	Line2D() :
// 		Model(Eigen::MatrixXd(3, 1))
// 	{}
// 	Line2D(const Line2D& other)
// 	{
// 		descriptor = other.descriptor;
// 	}
// };

// class FundamentalMatrix : public Model
// {
// public:
// 	FundamentalMatrix() :
// 		Model(Eigen::MatrixXd(3, 3))
// 	{}
// 	FundamentalMatrix(const FundamentalMatrix& other)
// 	{
// 		descriptor = other.descriptor;
// 	}
// };

// class EssentialMatrix : public Model
// {
// public:
// 	EssentialMatrix() :
// 		Model(Eigen::MatrixXd(3, 3))
// 	{}
// 	EssentialMatrix(const EssentialMatrix& other)
// 	{
// 		descriptor = other.descriptor;
// 	}
// };

// class Pose6D : public Model
// {
// public:
// 	Pose6D() :
// 		Model(Eigen::MatrixXd(3, 4))
// 	{}
// 	Pose6D(const Pose6D& other_)
// 	{
// 		descriptor = other_.descriptor;
// 	}
// };

// class Homography : public Model
// {
// public:
// 	Homography() : Model(Eigen::MatrixXd(3, 3)) {}
// 	Homography(const Homography& other)
// 	{
// 		descriptor = other.descriptor;
// 	}
// };

// class RadialHomography : public Model
// {
// public:
// 	RadialHomography() :
// 		Model(Eigen::MatrixXd(3, 7))
// 	{}

// 	RadialHomography(const RadialHomography& other)
// 	{
// 		descriptor = other.descriptor;
// 	}
// };

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
};

struct ScaleBasedRectifyingHomography : public NormalizingTransform
{
	// model parameters
	double h7 = 0.0;
	double h8 = 0.0;
	// feature-class scale (varies between different classes of features)
	// this is the relative scale of the recitified features.
	double alpha = 1.0;

	inline void rectifyPoint(Eigen::Vector3d& p) const
	{
		p(2) = h7 * p(0) + h8 * p(1) + p(2);
	} 

	inline void unrectifyPoint(Eigen::Vector3d& p) const
	{
		// negating h7 and h8 is equivalent to inverting the warping
		// homography matrix in this case
		p(2) = -h7 * p(0) - h8 * p(1) + p(2);
	}

	double rectifiedAngle(const double& x, const double& y, const double& angle) const
	{
		constexpr double kTwoPI = 2.0 * M_PI;
		const auto ct = std::cos(angle);
   		const auto st = std::sin(angle);
		const auto numer = (x * st - y * ct) * h7 + st;
		const auto denom = (-x * st + y * ct) * h8 + ct;
		return fmod(std::atan2(numer, denom), kTwoPI);
	}

	double unrectifiedAngle(const double& x, const double& y, const double& angle) const
	{
		constexpr double kTwoPI = 2.0 * M_PI;
		const auto ct = std::cos(angle);
   		const auto st = std::sin(angle);
		// negating h7 and h8 is equivalent to inverting the warping
		// homography matrix in this case
		const auto numer = (x * st - y * ct) * (-h7) + st;
		const auto denom = (-x * st + y * ct) * (-h8) + ct;
		return fmod(std::atan2(numer, denom), kTwoPI);
	}

	inline double rectifiedScale(const double& x, const double& y, const double& scale) const
	{
		return scale * std::pow(h7 * x + h8 * y + 1.0, 3.0);
	}

	inline double unrectifiedScale(const double& x, const double& y, const double& scale) const
	{
		// negating h7 and h8 is equivalent to inverting the warping
		// homography matrix in this case
		return scale * std::pow(-h7 * x - h8 * y + 1.0, 3.0);
	}

	void rectifyFeature(double& x, double& y, double& angle, double& scale) const
	{
		angle = rectifiedAngle(x, y, angle);
		scale = rectifiedScale(x, y, scale);
		Eigen::Vector3d point(x, y, 1.0);
		rectifyPoint(point);
		x = point(0) / point(2);
		y = point(1) / point(2);
	}

	void unrectifyFeature(double& x, double& y, double& angle, double& scale) const
	{
		angle = unrectifiedAngle(x, y, angle);
		scale = unrectifiedScale(x, y, scale);
		Eigen::Vector3d point(x, y, 1.0);
		unrectifyPoint(point);
		x = point(0) / point(2);
		y = point(1) / point(2);
	}

	Eigen::Matrix3d getHomography() const
	{
		Eigen::Matrix3d N; // normalizing transform
		N << s, 0, -s * x0,
			 0, s, -s * y0,
			 0, 0, 1;
		Eigen::Matrix3d H; // homography in normalized coordinate space
		H << 1,  0,  0,
			 0,  1,  0,
			 h7, h8, 1;
		return N.inverse() * H * N;
	}
};

struct SIFTRectifyingHomography : public ScaleBasedRectifyingHomography
{
	// The orthogonal directions (in radians) of the vanishing points, used to
	// estimate the model, in the rectified image.
	double vanishing_point_dir1 = 0.0;
	double vanishing_point_dir2 = 0.0;
};

}
