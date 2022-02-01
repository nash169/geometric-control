#include <iostream>

#include <geometric_control/GeometricControl.hpp>
#include <utils_lib/FileManager.hpp>

using namespace geometric_control;
using namespace utils_lib;

struct Params {
    struct kernel : public defaults::kernel {
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, 2);
    };
};

int main(int argc, char** argv)
{
    // Data
    double box[] = {-50, 50, -50, 50};
    size_t resolution = 100, num_samples = resolution * resolution, dim = sizeof(box) / sizeof(*box) / 2, num_deform = 1;

    Eigen::MatrixXd gridX = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1),
                    gridY = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution),
                    X(num_samples, dim), Xi(num_deform, dim);
    // Test points
    X << Eigen::Map<Eigen::VectorXd>(gridX.data(), gridX.size()), Eigen::Map<Eigen::VectorXd>(gridY.data(), gridY.size());

    // Deformation points
    Xi << 0, 0;

    // Deformation intensity
    Eigen::VectorXd w(num_deform);
    w << 1;

    // Deformed space
    using Kernel_t = kernels::SquaredExp<Params>;
    manifolds::KernelDeformed<Params, Kernel_t> deformedSpace(dim);
    deformedSpace.setDeformations(Xi, w);

    // Sample space
    Eigen::MatrixXd samples(num_samples, dim + 1);
    for (size_t i = 0; i < num_samples; i++)
        samples.row(i) = deformedSpace.embedding(X.row(i));

    // Jacobian
    std::cout << "JACOBIAN" << std::endl;
    std::cout << deformedSpace.jacobian(X.row(5056)) << std::endl;

    // Metric
    std::cout << "METRIC" << std::endl;
    std::cout << deformedSpace.metric(X.row(5056)) << std::endl;

    std::cout << "RECONSTRUCTED METRIC" << std::endl;
    std::cout << deformedSpace.jacobian(X.row(5056)).transpose() * deformedSpace.jacobian(X.row(5056)) << std::endl;

    // Gradient metric
    std::cout << "GRADIENT METRIC" << std::endl;
    // std::cout << deformedSpace.metricGrad(X.row(5056)) << std::endl;

    // Christoffel symbols
    std::cout << "CHRISTOFFEL SYMBOLS" << std::endl;
    std::cout << deformedSpace.christoffel(X.row(5056)) << std::endl;

    utils_lib::FileManager io_manager;
    io_manager.setFile("rsc/kernel_space.csv");
    io_manager.write("SAMPLES", samples);

    return 0;
}