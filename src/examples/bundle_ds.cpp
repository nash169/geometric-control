#include <iostream>
#include <utils_cpp/FileManager.hpp>

// Bundle DS
#include <geometric_control/dynamics/BundleDynamics.hpp>

// Base manifold
#include <geometric_control/manifolds/SphereN.hpp>

// Tasks
#include <geometric_control/tasks/DissipativeEnergy.hpp>
#include <geometric_control/tasks/PotentialEnergy.hpp>

using namespace geometric_control;
using namespace utils_cpp;

int main(int argc, char** argv)
{
    // Data
    double box[] = {0, M_PI, 0, 2 * M_PI};
    constexpr size_t dim = 2;
    size_t resolution = 100, num_samples = resolution * resolution;

    Eigen::MatrixXd gridX = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1),
                    gridY = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution),
                    X(num_samples, dim);
    // Test points
    X << Eigen::Map<Eigen::VectorXd>(gridX.data(), gridX.size()), Eigen::Map<Eigen::VectorXd>(gridY.data(), gridY.size());

    // Define base manifold
    using Manifold = manifolds::SphereN<dim>;

    // Create DS on a specific manifold
    dynamics::BundleDynamics<Manifold> ds;

    // Assign potential and dissipative task to the DS
    ds.addTasks(std::make_unique<tasks::PotentialEnergy<Manifold>>(), std::make_unique<tasks::DissipativeEnergy<Manifold>>());

    // Attractor
    Eigen::Vector3d a = Manifold().embedding(Eigen::Vector2d(1.5, 3));

    // Set tasks' properties
    static_cast<tasks::PotentialEnergy<Manifold>&>(ds.task(0)).setStiffness(Eigen::Matrix3d::Identity()).setAttractor(a);
    static_cast<tasks::DissipativeEnergy<Manifold>&>(ds.task(1)).setDissipativeFactor(Eigen::Matrix3d::Identity());

    // Embedding
    Eigen::VectorXd potential(num_samples);
    Eigen::MatrixXd embedding(num_samples, dim + 1);

    for (size_t i = 0; i < num_samples; i++) {
        embedding.row(i) = Manifold().embedding(X.row(i));
        potential(i) = ds.task(0).map(embedding.row(i))[0];
    }

    // Dynamics
    double time = 0, max_time = 20, dt = 0.001;
    size_t num_steps = std::ceil(max_time / dt) + 1, index = 0;
    Eigen::Vector3d x = Manifold().embedding(Eigen::Vector2d(1, 4)),
                    v = Manifold().jacobian(Eigen::Vector2d(1, 4)) * Eigen::Vector2d(-1, 1);

    // Record
    Eigen::MatrixXd record = Eigen::MatrixXd::Zero(num_steps, 1 + 2 * (dim + 1));
    record.row(0)(0) = time;
    record.row(0).segment(1, dim + 1) = x;
    record.row(0).segment(dim + 2, dim + 1) = v;

    while (time < max_time && index < num_steps - 1) {
        // Velocity
        v = v + dt * Manifold().projector(x, ds(x, v));

        // Position
        x = Manifold().retraction(x, v, dt);

        // Step forward
        time += dt;
        index++;

        // Record
        record.row(index)(0) = time;
        record.row(index).segment(1, dim + 1) = x;
        record.row(index).segment(dim + 2, dim + 1) = v;
    }

    FileManager io_manager;
    io_manager.setFile("rsc/sphere_bundle.csv");
    io_manager.write("RECORD", record, "EMBEDDING", embedding, "POTENTIAL", potential);

    return 0;
}