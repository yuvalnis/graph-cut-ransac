#pragma once

#include <stddef.h>
#include <array>
#include <algorithm>

namespace gcransac
{

template <size_t NumInlierTypes_t>
class Score
{
public:
	using NumInlierTypes = std::integral_constant<size_t, NumInlierTypes_t>;

	Score()
	{
		for (size_t i = 0; i < NumInlierTypes::value; i++)
		{
			m_num_inliers[i] = 0;
			m_values[i] = 0.0;
		}
		m_total_num_inliers = 0;
		m_values_sum = 0.0;
		finalized = false;
	}

	inline bool operator<(const Score& score) const
	{
		return value() < score.value();
	}

	inline bool operator>(const Score& score) const
	{
		return value() > score.value();
	}

	void finalize()
	{
		m_total_num_inliers = std::accumulate(m_num_inliers.begin(), m_num_inliers.end(), 0);
		m_values_sum = std::accumulate(m_values.begin(), m_values.end(), 0.0);
		finalized = true;
	}

	inline const size_t& num_inliers_by_type(const size_t& index) const
	{
		return m_num_inliers.at(index);
	}

	inline size_t& num_inliers_by_type(const size_t& index)
	{
		return m_num_inliers.at(index);
	}

	inline const double& value_by_type(const size_t& index) const
	{
		return m_values.at(index);
	}

	inline double& value_by_type(const size_t& index)
	{
		return m_values.at(index);
	}

	inline const size_t& num_inliers() const
	{
		if (!finalized)
		{
			throw std::runtime_error("Score was not finalized!");
		}
		return m_total_num_inliers;
	}

	inline const double& value() const
	{
		if (!finalized)
		{
			throw std::runtime_error("Score was not finalized!");
		}
		return m_values_sum;
	}

	inline const std::array<size_t, NumInlierTypes::value>& inlier_num_array() const
	{
		return m_num_inliers;
	}

private:

	std::array<size_t, NumInlierTypes::value> m_num_inliers; // Number of inliers, rectangular gain function
	std::array<double, NumInlierTypes::value> m_values;
	size_t m_total_num_inliers;
	double m_values_sum;
	bool finalized;
};

}
