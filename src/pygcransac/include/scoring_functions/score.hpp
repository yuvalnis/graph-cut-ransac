#pragma once

#include <stddef.h>
#include <array>
#include <algorithm>
#include <sstream>

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
	}

	inline bool operator<(const Score& score) const
	{
		return value() < score.value();
	}

	inline bool operator>(const Score& score) const
	{
		return value() > score.value();
	}

	void increment_inlier_num(const size_t& index)
	{
		verify_index(index);
		m_num_inliers.at(index) += 1;
		m_total_num_inliers++;
	}

	void increment_value(const size_t& index, const double& value)
	{
		verify_index(index);
		m_values.at(index) += value;
		m_values_sum += value;
	}

	void reset_value(const size_t& index, const double& value)
	{
		verify_index(index);
		m_values_sum -= m_values.at(index);
		m_values.at(index) = value;
		m_values_sum += value;
	}

	inline const size_t& num_inliers_by_type(const size_t& index) const
	{
		return m_num_inliers.at(index);
	}

	inline const double& value_by_type(const size_t& index) const
	{
		return m_values.at(index);
	}

	inline const size_t& num_inliers() const
	{
		return m_total_num_inliers;
	}

	inline const double& value() const
	{
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

	static void verify_index(const size_t& index)
	{
		if (index > (NumInlierTypes::value - 1))
		{
			std::stringstream err_msg;
			err_msg << "Invalid index: " << index << ". "
					<< "Number of inlier types: " << NumInlierTypes::value << ".\n";
			throw std::runtime_error(err_msg.str());
		}
	}
};

}
