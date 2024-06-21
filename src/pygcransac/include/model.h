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

#include <opencv2/core/core.hpp>
#include <vector>
#include <Eigen/Eigen>
#include "estimators/estimator.h"

namespace gcransac
{

class Model
{
public:
	Eigen::MatrixXd descriptor; // The descriptor of the current model
	Model(const Eigen::MatrixXd &descriptor_) : descriptor(descriptor_) {}
	Model() {}
};

class RigidTransformation : public Model
{
public:
	RigidTransformation() : Model(Eigen::MatrixXd(4, 4)) {}
	RigidTransformation(const RigidTransformation& other)
	{
		descriptor = other.descriptor;
	}
};

class Line2D : public Model
{
public:
	Line2D() :
		Model(Eigen::MatrixXd(3, 1))
	{}
	Line2D(const Line2D& other)
	{
		descriptor = other.descriptor;
	}
};

class FundamentalMatrix : public Model
{
public:
	FundamentalMatrix() :
		Model(Eigen::MatrixXd(3, 3))
	{}
	FundamentalMatrix(const FundamentalMatrix& other)
	{
		descriptor = other.descriptor;
	}
};

class EssentialMatrix : public Model
{
public:
	EssentialMatrix() :
		Model(Eigen::MatrixXd(3, 3))
	{}
	EssentialMatrix(const EssentialMatrix& other)
	{
		descriptor = other.descriptor;
	}
};

class Pose6D : public Model
{
public:
	Pose6D() :
		Model(Eigen::MatrixXd(3, 4))
	{}
	Pose6D(const Pose6D& other_)
	{
		descriptor = other_.descriptor;
	}
};

class Homography : public Model
{
public:
	Homography() : Model(Eigen::MatrixXd(3, 3)) {}
	Homography(const Homography& other)
	{
		descriptor = other.descriptor;
	}
};

class RadialHomography : public Model
{
public:
	RadialHomography() :
		Model(Eigen::MatrixXd(3, 7))
	{}

	RadialHomography(const RadialHomography& other)
	{
		descriptor = other.descriptor;
	}
};

struct ScaleBasedRectifyingHomography
{
	// model parameters
	double h7 = 0.0;
	double h8 = 0.0;
	// feature-class scale (varies between different classes of features)
	double alpha = 1.0;
	// normalization parameters
	double x0 = 0.0; // x-coordinate of the unnormalized sample set mean
	double y0 = 0.0; // y-coordinate of the unnormalized sample set mean
	double s = 1.0; // normalizing rescaling factor of sample set

	void normalize(Eigen::Vector3d& p) const
	{
		// generally, the homogeneous coordinate can be different than 1 (p(2) != 1)
		p(0) = s * p(2) * (p(0) - x0);
		p(1) = s * p(2) * (p(1) - y0);
	}

	void denormalize(Eigen::Vector3d& p) const
	{
		const auto inv_s = 1.0 / s;
		// generally, the homogeneous coordinate can be different than 1 (p(2) != 1)
		p(0) = inv_s * p(0) + x0 * p(2);
		p(1) = inv_s * p(1) + y0 * p(2);
	}

	void applyRectification(Eigen::Vector3d& p) const
	{
		p(2) = h7 * p(0) + h8 * p(1) + p(2);
	}

	void applyInverseRectification(Eigen::Vector3d& p) const
	{
		// negating h7 and h8 is equivalent to inverting the rectifying
		// homography matrix in this case
		p(2) = -h7 * p(0) - h8 * p(1) + p(2);
	}

	Eigen::Matrix3d getHomography() const
	{
		const double sh7 = s * h7;
		const double sh8 = s * h8;
		const double sx0h7 = x0 * sh7;
		const double sy0h8 = y0 * sh8;
		const double sx0h7y0h8 = sx0h7 + sy0h8;
		Eigen::Matrix3d result;
		result << 1.0 + sx0h7, x0 * sh8, 	-x0 * sx0h7y0h8,
				  y0 * sh7,	   1.0 + sy0h8, -y0 * sx0h7y0h8,
				  sh7,		   sh8, 		1.0 - sx0h7y0h8;
		return result;
	}
};

struct SIFTRectifyingHomography : public ScaleBasedRectifyingHomography
{
	// vanishing point in the original image used to estimate model
	Eigen::Vector3d vp1 = Eigen::Vector3d::Zero(); 
	// vanishing point in the original image, such that when rectified it is
	// orthogonal to the first vanishing point (also rectified).
	Eigen::Vector3d vp2 = Eigen::Vector3d::Zero();
};

}
