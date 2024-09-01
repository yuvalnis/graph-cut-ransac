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

#include <vector>
#include <memory>

namespace gcransac::sampler
{
// Purely virtual class used for the sampling consensus methods (e.g. Ransac,
// Prosac, MLESac, etc.)
class Sampler
{
public:

	virtual ~Sampler() {}

	virtual inline const std::string getName() const = 0;

	virtual void reset() = 0;

	virtual void update(
		const size_t* const subset_,
		const size_t& sample_size_,
		const size_t& iteration_number_, // TODO remove - no child classes use this parameter in their implementation
		const double& inlier_ratio_ // TODO remove - no child classes use this parameter in their implementation
	)
	{
		// do nothing
		return;
	}

	// Samples the input variable data and fills the std::vector subset with the
	// samples.
	virtual bool sample(
		const std::vector<size_t> &pool_,
		size_t * const subset_,
		size_t sample_size_
	) = 0;

	virtual bool isInitialized() const
	{
		return true;
	}
};

// template <size_t N> 
// class SamplerArray
// {
// private:

// 	std::array<std::unique_ptr<Sampler>, N> samplers;

// public:

// 	inline void update() const
// 	{
// 		// right now supports only UniformSamplers which have an empty update method
// 		return;
// 	}
// };

}
