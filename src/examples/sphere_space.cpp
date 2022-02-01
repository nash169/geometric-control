#include <algorithm>
#include <iostream>
// #include <span>
#include <vector>

#include <geometric_control/GeometricControl.hpp>
#include <geometric_control/tools/helper.hpp>
#include <utils_lib/FileManager.hpp>

using namespace geometric_control;
using namespace utils_lib;

int main(int argc, char** argv)
{
    // Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 3),
    //                 B = Eigen::MatrixXd::Random(3, 3);

    // std::cout << A * B << std::endl;

    // std::cout << "-------------------------\n";

    // Eigen::Tensor<double, 2> At = tools::TensorCast(A, 3, 3),
    //                          Bt = tools::TensorCast(B, 3, 3);

    // Eigen::array<Eigen::IndexPair<int>, 1> prod_dims = {Eigen::IndexPair<int>(1, 0)};

    // std::cout << At.contract(Bt, prod_dims) << std::endl;

    std::vector<double> v = {1, 2, 3, 4, 5};

    std::cout << "vector: ";
    for (auto& i : v)
        std::cout << i << " ";
    std::cout << std::endl;

    std::iter_swap(v.begin() + 2, v.begin() + 3);
    std::swap(v[2], v[3]);

    std::cout << "vector swapped: ";
    for (auto& i : v)
        std::cout << i << " ";
    std::cout << std::endl;

    size_t dim = 2;

    manifolds::SphereN<2> sphere2;
    manifolds::SphereN sphere_n;

    manifolds::Sphere sphere(dim);

    // Eigen::Vector2d x(0.933993247757551, 0.678735154857773);
    Eigen::VectorXd x = Eigen::VectorXd::Random(dim), y = Eigen::VectorXd::Random(dim);

    // std::span<double> s = {x.data(), dim};

    // std::cout << "span: ";
    // for (auto const i : s)
    //     std::cout << i << " ";
    // std::cout << std::endl;

    // std::cout << x.transpose() << std::endl;

    // std::cout << "-------------------------\n";

    // std::cout << sphere.embedding(x).transpose() << std::endl;
    std::cout << sphere2.embedding(x).transpose() << std::endl;
    std::cout << sphere_n.embedding(x).transpose() << std::endl;

    // std::cout << "-------------------------\n";

    // std::cout << sphere.jacobian(x) << std::endl;
    // std::cout << sphere2.jacobian(x) << std::endl;

    // std::cout << "-------------------------\n";

    // std::cout << "Metric" << std::endl;
    // std::cout << sphere.metric(x) << std::endl;
    // std::cout << sphere2.metric(x) << std::endl;

    // std::cout << "-------------------------\n";

    // // std::cout << sphere.jacobian(x).transpose() * sphere.jacobian(x) << std::endl;

    // // std::cout << "-------------------------\n";

    // std::cout << "Metric Grad" << std::endl;
    // std::cout << sphere.metricGrad(x) << std::endl;
    // std::cout << sphere2.metricGrad(x) << std::endl;

    // std::cout << "-------------------------\n";

    // std::cout << sphere.christoffel(x) << std::endl;
    // std::cout << sphere2.christoffel(x) << std::endl;

    // std::cout << "-------------------------\n";

    // std::cout << sphere2.hessian(x) << std::endl;

    // Eigen::Tensor<double, 2> h(3, 3);
    // h.setZero();
    // h(0, 0) = 1;
    // h(1, 1) = 1;
    // h(2, 2) = 1;
    // Eigen::Tensor<double, 2> J = geometric_control::tools::TensorCast(sphere2.jacobian(x));
    // Eigen::Tensor<double, 3> H = sphere2.hessian(x);

    // Eigen::array<Eigen::IndexPair<int>, 1> c1 = {Eigen::IndexPair<int>(0, 0)};
    // Eigen::array<Eigen::IndexPair<int>, 1> c2 = {Eigen::IndexPair<int>(2, 0)};
    // Eigen::array<Eigen::IndexPair<int>, 1> c3 = {Eigen::IndexPair<int>(1, 0)};
    // Eigen::Tensor<double, 3> temp = H.contract(h, c1);
    // // std::cout << H.dimensions() << std::endl;
    // // std::cout << temp.dimensions() << std::endl;
    // // std::cout << J.dimensions() << std::endl;

    // std::cout << "-------------------------\n";

    // std::cout << H.contract(h, c1).contract(J, c2) + J.contract(h, c1).contract(H, c3) << std::endl;

    // std::cout << sphere2.jacobian(x).transpose() * geometric_control::tools::MatrixCast(h, 3, 3) * sphere2.jacobian(x) << std::endl;
    // std::cout << J.contract(h, c1).contract(J, c2) << std::endl;

    // std::cout << "-------------------------\n";

    // std::cout << sphere2.distanceHess(x, y) << std::endl;

    return 0;
}