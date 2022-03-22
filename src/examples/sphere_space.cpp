#include <iostream>

#include <geometric_control/manifolds/Sphere.hpp>
#include <utils_lib/FileManager.hpp>

using namespace geometric_control;
using namespace utils_lib;

int main(int argc, char** argv)
{
    manifolds::Sphere<2> s2;
    manifolds::Sphere sn;

    Eigen::VectorXd v1 = Eigen::Vector3d::Random(),
                    v2 = Eigen::VectorXd::Random(5);

    std::cout << s2.metric(v1) << std::endl;
    std::cout << "-" << std::endl;
    std::cout << sn.metric(v2) << std::endl;

    return 0;
}