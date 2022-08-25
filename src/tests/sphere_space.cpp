#include <iostream>

#include <geometric_control/manifolds/Sphere.hpp>
#include <utils_lib/FileManager.hpp>

using namespace geometric_control;
using namespace utils_lib;

int main(int argc, char** argv)
{
    Eigen::Vector2d x(0.933993247757551, 0.678735154857773);

    std::cout << "SPECIALIZED TEMPLATE" << std::endl;
    manifolds::Sphere<2> s2;

    // std::cout << "Embedding" << std::endl;
    // std::cout << s2.embedding(x).transpose() << std::endl;
    // std::cout << "Jacobian" << std::endl;
    // std::cout << s2.embeddingJacobian(x) << std::endl;
    // std::cout << "Hessian" << std::endl;
    // std::cout << s2.embeddingHessian(x) << std::endl;
    // std::cout << "Metric" << std::endl;
    // std::cout << s2.pullMetric(x) << std::endl;
    // std::cout << "Metric Grad" << std::endl;
    // std::cout << s2.pullMetricGrad(x) << std::endl;
    // std::cout << "Christoffel" << std::endl;
    // std::cout << s2.leviCivitaChristoffel(x) << std::endl;
    // std::cout << tools::leviCivitaConnection(tools::TensorCast(s2.pullMetric(x).inverse()), s2.pullMetricGrad(x)) << std::endl;

    // std::cout << "GENERIC TEMPLATE" << std::endl;
    // manifolds::Sphere sn;

    // std::cout << "Embedding" << std::endl;
    // std::cout << sn.embedding(x).transpose() << std::endl;
    // std::cout << "Jacobian" << std::endl;
    // std::cout << sn.embeddingJacobian(x) << std::endl;
    // std::cout << "Hessian" << std::endl;
    // std::cout << sn.embeddingHessian(x) << std::endl;
    // std::cout << "Metric" << std::endl;
    // std::cout << sn.pullMetric(x) << std::endl;
    // std::cout << "Christoffel" << std::endl;
    // std::cout << sn.leviCivitaChristoffel(x) << std::endl;

    // Eigen::VectorXd v1 = Eigen::Vector3d::Random(),
    //                 v2 = Eigen::VectorXd::Random(5);

    // std::cout << s2.metric(v1) << std::endl;
    // std::cout << "-" << std::endl;
    // std::cout << sn.metric(v2) << std::endl;

    return 0;
}