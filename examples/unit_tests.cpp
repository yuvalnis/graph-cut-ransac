#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "math_utils.hpp"

void testCollinearPointsPositiveCase()
{
    using namespace gcransac::utils;
    Point2D p0{1, 0};
    Point2D p1{2, 1};
    Point2D p2{3, 2};
    if (!areCollinear(p0, p1, p2, 1e-9))
    {
        std::stringstream error_msg;
        error_msg << "Incorrect result that points "
                  << "(" << p0.x() << ", " << p0.y() << "), "
                  << "(" << p1.x() << ", " << p1.y() << "), "
                  << "(" << p2.x() << ", " << p2.y() << ") "
                  << "are not collinear.\n";
        throw std::runtime_error(error_msg.str());
    }
}

void testCollinearPointsNegativeCase()
{
    using namespace gcransac::utils;
    Point2D p0{1, 0};
    Point2D p1{2, 1};
    Point2D p2{1, 4};
    if (areCollinear(p0, p1, p2, 1e-9))
    {
        std::stringstream error_msg;
        error_msg << "Incorrect result that points "
                  << "(" << p0.x() << ", " << p0.y() << "), "
                  << "(" << p1.x() << ", " << p1.y() << "), "
                  << "(" << p2.x() << ", " << p2.y() << ") "
                  << "are collinear.\n";
        throw std::runtime_error(error_msg.str());
    }
}

void testConvexHullAlgo()
{
    using namespace gcransac::utils;
    // Construct input
    std::vector<Point2D> points;
    points.emplace_back(0, 3);
    points.emplace_back(2, 2);
    points.emplace_back(1, 1);
    points.emplace_back(2, 1);
    points.emplace_back(3, 0);
    points.emplace_back(0, 0);
    points.emplace_back(3, 3);
    // Constuct expected output
    std::vector<Point2D> expected;
    expected.emplace_back(0, 0);
    expected.emplace_back(3, 0);
    expected.emplace_back(3, 3);
    expected.emplace_back(0, 3);
    // Find the convex hull
    std::vector<Point2D> actual = computeConvexHull(points);
    // Verify expected and actual output are the same size
    if (expected.size() != actual.size())
    {
        std::stringstream error_msg;
        error_msg << "Expected convex-hull size is " << expected.size()
                  << " but actual convex-hull size is " << actual.size()
                  << "\n";
        throw std::runtime_error(error_msg.str());
    }
    // Sort actual and expected outputs
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());
    // Verify both are equal
    for (size_t i = 0; i < actual.size(); i++)
    {
        const auto& p = actual.at(i);
        const auto& q = expected.at(i);
        if (p.x() != q.x() || p.y() != q.y())
        {
            std::stringstream error_msg;
            error_msg << "Actual convex-hull is different than expected. "
                      << "Point at index " << i
                      << " is (" << p.x() << ", " << p.y() << "), when "
                      << "(" << q.x() << ", " << q.y() << "), was expected.\n";
            throw std::runtime_error(error_msg.str());
        }
    }
}

void testPointInConvexPolygonPositiveCase()
{
    using namespace gcransac::utils;
    // Construct a convex polygon
    std::vector<Point2D> convex_polygon;
    convex_polygon.emplace_back(0, 0);
    convex_polygon.emplace_back(3, 0);
    convex_polygon.emplace_back(3, 3);
    convex_polygon.emplace_back(0, 3);
    // Construct a point expected to be inside of convex polygon
    const Point2D point{1, 2};
    if (!pointInConvexPolygon(point, convex_polygon))
    {
        std::stringstream error_msg;
        error_msg << "2D-point (" << point.x() << ", " << point.y() << ") "
                  << "is not inside the convex polygon with vertices:";
        for (size_t i = 0; i < convex_polygon.size(); i++)
        {
            const auto& vert = convex_polygon.at(i);
            error_msg << " (" << vert.x() << ", " << vert.y() << ")";
        }
        error_msg << std::endl;
        throw std::runtime_error(error_msg.str());
    }
}

void testPointInConvexPolygonNegativeCase()
{
    using namespace gcransac::utils;
    // Construct a convex polygon
    std::vector<Point2D> convex_polygon;
    convex_polygon.emplace_back(0, 0);
    convex_polygon.emplace_back(3, 0);
    convex_polygon.emplace_back(3, 3);
    convex_polygon.emplace_back(0, 3);
    // Construct a point expected to be inside of convex polygon
    const Point2D point{-1, 2};
    if (pointInConvexPolygon(point, convex_polygon))
    {
        std::stringstream error_msg;
        error_msg << "2D-point (" << point.x() << ", " << point.y() << ") "
                  << "is not outside the convex polygon with vertices:";
        for (size_t i = 0; i < convex_polygon.size(); i++)
        {
            const auto& vert = convex_polygon.at(i);
            error_msg << " (" << vert.x() << ", " << vert.y() << ")";
        }
        error_msg << std::endl;
        throw std::runtime_error(error_msg.str());
    }
}

void runTest(const std::string& desc, void (*test)())
{
    static size_t index{0};
    try
    {
        test();
        std::cout << "\033[32m" << "SUCCESS" << "\033[0m"
                  << " Test #" << ++index << ": " << desc << "\n";
    }
    catch (const std::runtime_error& e)
    {
        std::cout << "\033[31m"<< "FAILURE" << "\033[0m"
                  << " Test #" << ++index << ": " << desc << "\n"  
                  << "\t\t" << e.what() << "\n";
    }
}

int main()
{
    runTest("collinear positive case", testCollinearPointsPositiveCase);
    runTest("collinear negative case" ,testCollinearPointsNegativeCase);
    runTest("convex-hull", testConvexHullAlgo);
    runTest("point in convex-polygon positive case", testPointInConvexPolygonPositiveCase);
    runTest("point in convex-polygon negative case", testPointInConvexPolygonNegativeCase);
}
