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

namespace gcransac::utils
{

bool areCollinear(
	double x1, double y1, double x2, double y2, double x3, double y3,
	double tolerance
)
{
	// Compute the squared distances between each pair of points
    double dx12 = x1 - x2;
    double dy12 = y1 - y2;
    double dx13 = x1 - x3;
    double dy13 = y1 - y3;
    double dx23 = x2 - x3;
    double dy23 = y2 - y3;

    double d12_sq = dx12 * dx12 + dy12 * dy12;
    double d13_sq = dx13 * dx13 + dy13 * dy13;
    double d23_sq = dx23 * dx23 + dy23 * dy23;

    // Sum of the squares of the side lengths
    double sum_of_squares = d12_sq + d13_sq + d23_sq;

    // Compute the area of the triangle using the shoelace formula
    double area = 0.5 * std::abs(dx12 * dy13 - dy12 * dx13);

    // If all points are very close, assume collinearity
    if (sum_of_squares < 1e-6)
    {
        return true;
    }

    // Compute the measure, which approximates sin(smallest angle)
    double measure = (4.0 * area) / sum_of_squares;

    // Check if the measure is within the tolerance
    return measure < tolerance;
}

// Pivoting In-Place Gauss Elimination to solve problem A * x = b,
// where A is the known coefficient matrix, b is the inhomogeneous part and x is the unknown vector.
// Form: matrix_ = [A, b].
template<size_t _Size>
void gaussElimination(
    Eigen::Matrix<double, _Size, _Size + 1>& matrix_, // The matrix to which the elimination is applied
    Eigen::Matrix<double, _Size, 1>& result_) // The resulting null-space
{
    static_assert(_Size > 0, "Template argument _Size must be non-zero.");

    // Pivotisation
    for (size_t i = 0; i < _Size; i++)
    {
        for (size_t k = i + 1; k < _Size; k++)
        {
            if (abs(matrix_(i, i)) < abs(matrix_(k, i)))
            {
                for (size_t j = 0; j <= _Size; j++)
                {
                    std::swap(matrix_(i, j), matrix_(k, j));
                }
            }
        }
    }

    // loop to perform the gauss elimination
    for (size_t i = 0; i < _Size - 1; i++)
    {         
        for (size_t k = i + 1; k < _Size; k++)
        {
            const double temp = matrix_(k, i) / matrix_(i, i);
            for (size_t j = 0; j <= _Size; j++)
            {
                // make the elements below the pivot elements equal to zero or elimnate the variables
                matrix_(k, j) = matrix_(k, j) - temp * matrix_(i, j);
            }
        }
    }

    // back-substitution
    for (size_t i = 0; i < _Size; i++)                
    {                       
        // result_ is an array whose values correspond to the values of x,y,z..
        const auto row_idx = _Size - 1 - i;
        result_(row_idx) = matrix_(row_idx, _Size);                
        //make the variable to be calculated equal to the rhs of the last equation
        for (size_t col_idx = row_idx + 1; col_idx < _Size; col_idx++)
        {
            if (col_idx != row_idx)
            {
                //then subtract all the lhs values except the coefficient of the variable whose value is being calculated
                result_(row_idx) = result_(row_idx) - matrix_(row_idx, col_idx) * result_(col_idx);
            }
        }
        //now finally divide the rhs by the coefficient of the variable to be calculated
        result_(row_idx) = result_(row_idx) / matrix_(row_idx, row_idx);
    }
}

}
