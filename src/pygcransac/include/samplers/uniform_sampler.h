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
#include <opencv2/core/core.hpp>
#include "uniform_random_generator.h"
#include "sampler.h"

namespace gcransac
{
	namespace sampler
	{
		class UniformSampler : public Sampler
		{
		protected:
			const size_t m_sample_number;
			std::unique_ptr<utils::UniformRandomGenerator<size_t>> m_random_generator;

		public:
			explicit UniformSampler(const size_t& sample_number) :
				m_sample_number(sample_number)
			{
				m_random_generator = std::make_unique<utils::UniformRandomGenerator<size_t>>();
				m_random_generator->resetGenerator(0, m_sample_number);
			}

			const std::string getName() const { return "Uniform Sampler"; }

			void reset()
			{
				m_random_generator->resetGenerator(0, m_sample_number);
			}

			// Samples the input variable data and fills the std::vector subset with the
			// samples.
			bool sample(
				const std::vector<size_t> &pool_,
				size_t * const subset_,
				size_t sample_size_
			);
		};

		bool UniformSampler::sample(
			const std::vector<size_t> &pool_,
			size_t * const subset_,
			size_t sample_size_
		)
		{
			// If there are not enough points in the pool, interrupt the procedure.
			if (sample_size_ > pool_.size())
			{
				return false;
			}

			// Generate a unique random set of indices.
			m_random_generator->generateUniqueRandomSet(
				subset_, sample_size_, pool_.size() - 1
			);

			// Replace the temporary indices by the ones in the pool
			for (size_t i = 0; i < sample_size_; ++i)
			{
				subset_[i] = pool_[subset_[i]];
			}
			return true;
		}
	}
}