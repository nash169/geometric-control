#include <iostream>

// #include <geometric_control/GeometricControl.hpp>
#include <geometric_control/dynamics/MassSpringDamper.hpp>
#include <geometric_control/manifolds/Sphere.hpp>

#include <integrator_lib/Integrate.hpp>
#include <utils_lib/FileManager.hpp>

using namespace geometric_control;
using namespace utils_lib;
using namespace integrator_lib;

struct Params {
    struct integrator : public defaults::integrator {
    };
};

int main(int argc, char** argv)
{
    // Data
    double box[] = {0, M_PI, 0, 2 * M_PI};
    size_t resolution = 100, num_samples = resolution * resolution, dim = sizeof(box) / sizeof(*box) / 2;

    Eigen::MatrixXd gridX = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1),
                    gridY = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution),
                    X(num_samples, dim);
    // Test points
    X << Eigen::Map<Eigen::VectorXd>(gridX.data(), gridX.size()), Eigen::Map<Eigen::VectorXd>(gridY.data(), gridY.size());

    // Attractor
    Eigen::Vector2d a(1.5, 3);

    // Dynamics
    double time = 0, max_time = 20, dt = 0.001;
    size_t num_steps = std::ceil(max_time / dt) + 1, index = 0;

    dynamics::MassSpringDamper<manifolds::Sphere> msd(dim);
    msd.setAttractor(a);
    msd.setDissipativeFactor(Eigen::MatrixXd::Identity(dim, dim) * 5);
    msd.setPotentialFactor(Eigen::MatrixXd::Identity(dim, dim) * 2);

    integrator::ForwardEuler<Params> integrator;
    integrator.setStep(dt);

    Eigen::VectorXd x = Eigen::Vector2d(1, 4), v = Eigen::Vector2d(-1, 1);
    std::cout << msd(x, v).transpose() << std::endl;
    std::cout << msd.manifold().embedding(a).transpose() << std::endl;
    Eigen::MatrixXd record = Eigen::MatrixXd::Zero(num_steps, 1 + 2 * dim);

    Eigen::VectorXd potential(num_samples);
    Eigen::MatrixXd embedding(num_samples, dim + 1), potentialGrad(num_samples, dim);

    for (size_t i = 0; i < num_samples; i++) {
        embedding.row(i) = msd.manifold().embedding(X.row(i));
        potential(i) = msd.potentialEnergy(X.row(i));
        potentialGrad.row(i) = msd.potentialGrad(X.row(i)) / msd.potentialGrad(X.row(i)).norm();
    }

    record.row(0)(0) = time;
    record.row(0).segment(1, dim) = x;
    record.row(0).segment(dim + 1, dim) = v;

    while (time < max_time && index < num_steps - 1) {
        // Velocity
        v = v + dt * msd(x, v);

        // Position
        x = x + v * dt;

        // Step forward
        time += dt;
        index++;

        // Record
        record.row(index)(0) = time;
        record.row(index).segment(1, dim) = x;
        record.row(index).segment(dim + 1, dim) = v;
    }

    Eigen::MatrixXd projection(record.rows(), dim + 1);

    for (size_t i = 0; i < record.rows(); i++)
        projection.row(i) = msd.manifold().embedding(record.row(i).segment(1, dim));

    FileManager io_manager;
    io_manager.setFile("rsc/sphere_msd.csv");
    io_manager.write("CHART", X, "EMBEDDING", embedding, "FUNCTION", potential, "GRAD", potentialGrad, "RECORD", record, "PROJECTION", projection);

    return 0;
}