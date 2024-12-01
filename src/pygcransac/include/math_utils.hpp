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
#include <string>
#include <sstream>
#include <algorithm>
#include <Eigen/Eigen>

namespace gcransac::utils
{

template<typename T>
constexpr T cube(T x)
{
    return x * x * x;
}

constexpr size_t nChoose2(size_t n)
{
    if (n == 0)
    {
        return 0;
    }
    return (n * (n - 1)) / 2;
}

constexpr double deg2rad(double angle)
{
    constexpr double kDegToRad{M_PI / 180.0};
    return angle * kDegToRad;
}

constexpr double rad2deg(double angle)
{
    constexpr double kRadToDeg{M_1_PI * 180.0};
    return angle * kRadToDeg;
}

inline double clipAngle(double angle)
{
    constexpr auto kTwoPI = 2.0 * M_PI;
    // Get the remainder when divided by 2π
    angle = std::fmod(angle, kTwoPI);
    if (angle < 0.0) {
    // If the angle is negative, bring it to the positive range
        angle += kTwoPI;  
    }
    return angle;
}

inline double minAngleDiff(double angle1, double angle2)
{
    constexpr auto kTwoPI = 2.0 * M_PI;
    auto diff = std::fabs(clipAngle(angle1) - clipAngle(angle2));
    return std::fmin(diff, kTwoPI - diff);
}

inline double linesAnglesDiff(double angle1, double angle2)
{
    auto diff1 = minAngleDiff(angle1, angle2);
    auto diff2 = minAngleDiff(angle1, angle2 - M_PI);
    return std::fmin(diff1, diff2);
}

inline Eigen::Vector3d lineFromPointAndAngle(double x, double y, double theta)
{
    const auto c = std::cos(theta);
    const auto s = std::sin(theta);
    return {s, -c, y * c - x * s};
}

class Point2D
{
public:

    Point2D() : m_x(0), m_y(0) {}
    Point2D(double x, double y) : m_x(x), m_y(y) {}
    inline const double& x() const { return m_x; }
    inline const double& y() const { return m_y; }

    // Overloading < operator to allow sorting points by lexicographical order
    bool operator<(const Point2D& p) const
    {
        return m_x < p.x() || (m_x == p.x() && m_y < p.y());
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "(" << m_x << ", " << m_y << ")";
        return oss.str();
    }

private:

    double m_x;
    double m_y;
};

inline bool areCollinear(
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

inline bool areCollinear(
    const Point2D& p1, const Point2D& p2, const Point2D& p3, double tolerance
)
{
    return areCollinear(p1.x(), p1.y(), p2.x(), p2.y(), p3.x(), p3.y(), tolerance);
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

/// @brief Computes the two-dimensional cross-product of the vectors OP and OQ
/// (with O as the origin).
/// @param O Point2D representing the origin. 
/// @param P Point2D to which the first vector is pointing from the origin.
/// @param Q Point2D to which the second vector is pointing from the origin.
/// @return The two-dimensional cross-product of OP and OQ.
inline double crossProduct(const Point2D& O, const Point2D& P, const Point2D& Q)
{
    return (P.x() - O.x()) * (Q.y() - O.y()) - (P.y() - O.y()) * (Q.x() - O.x());
}

/// @brief Computes the convex-hull of points. The set of points is ordered
/// lexicographically during the runtime of this function.
/// @param points a set of points, represented by a vector of Point2D.
/// @return A polygon, represented by a vector of Point2D, which forms the
/// convex-hull of points.
inline std::vector<Point2D> computeConvexHull(std::vector<Point2D>& points)
{
    const size_t n_points = points.size();
    if (n_points <= 1)
    {
        return points;
    }
    // Initialize result vector to twice the size the input as in the worst-case
    // it can hold all points twice.
    std::vector<Point2D> result(2 * n_points);
    // Sort points lexicographically.
    std::sort(points.begin(), points.end());
    // Build lower hull.
    size_t k{0};
	for (size_t i = 0; i < n_points; ++i)
    {
		while (k >= 2 && crossProduct(result[k-2], result[k-1], points[i]) <= 0)
        {
            k--;
        }
		result[k++] = points[i];
	}
	// Build upper hull.
    const size_t t = k + 1; 
	for (size_t i = n_points - 1; i > 0; --i)
    {
		while (k >= t && crossProduct(result[k-2], result[k-1], points[i-1]) <= 0)
        {
            k--;
        }
		result[k++] = points[i - 1];
	}
    // Resize result to fit convex-hull size and return.
	result.resize(k - 1);
    // Check special case where result has two vertices that are the same
    if (result.size() == 2)
    {
        bool close_x = std::abs(result[0].x() - result[1].x()) < 1e-9;
        bool close_y = std::abs(result[0].y() - result[1].y()) < 1e-9;
        if (close_x && close_y)
        {
            result.resize(1);
        }
    }
	return result;
}

/// @brief Checks if a 2D-point is contained within a convex polygon.
/// @param point a 2D-point
/// @param polygon a collection of 2D-points representing the vertices of a
/// convex polygon.
/// @return True, if the point is inside the polygon. False, otherwise.
inline bool pointInConvexPolygon(
    const Point2D& point,
    const std::vector<Point2D>& polygon
)
{
    const size_t n_verts = polygon.size();
    if (n_verts < 3)
    {
        return false;
    }

    bool positive = false;
    bool negative = false;
    for (size_t i = 0; i < n_verts; i++)
    {
        double cp = crossProduct(polygon[i], polygon[(i + 1) % n_verts], point);
        if (cp > 0)
        {
            positive = true;
        }
        else if (cp < 0)
        {
            negative = true;
        }
        if (positive && negative)
        {
            return false;
        }
    }
    return true;
}

}
