#include <algorithm>
#include <iostream>
// #include <span>
#include <vector>

#include <geometric_control/GeometricControl.hpp>
#include <geometric_control/tools/helper.hpp>

using namespace geometric_control;

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

    size_t dim = 3;

    manifolds::Sphere sphere(dim);

    // Eigen::Vector2d x(0.933993247757551, 0.678735154857773);
    Eigen::VectorXd x = Eigen::VectorXd::Random(dim);

    // std::span<double> s = {x.data(), dim};

    // std::cout << "span: ";
    // for (auto const i : s)
    //     std::cout << i << " ";
    // std::cout << std::endl;

    std::cout << x.transpose() << std::endl;

    std::cout << "-------------------------\n";

    std::cout << sphere.embedding(x).transpose() << std::endl;

    std::cout << "-------------------------\n";

    std::cout << sphere.jacobian(x) << std::endl;

    std::cout << "-------------------------\n";

    std::cout << sphere.metric(x) << std::endl;

    std::cout << "-------------------------\n";

    std::cout << sphere.jacobian(x).transpose() * sphere.jacobian(x) << std::endl;

    std::cout << "-------------------------\n";

    std::cout << sphere.metricGrad(x) << std::endl;

    std::cout << "-------------------------\n";

    std::cout << sphere.christoffel(x) << std::endl;

    return 0;
}