#include <iostream>
#include <random>
#include <utils_lib/FileManager.hpp>

// Bundle DS
#include <geometric_control/dynamics/BundleDynamics.hpp>

// Base manifold
#include <geometric_control/manifolds/Euclidean.hpp>
#include <geometric_control/manifolds/Sphere.hpp>

// Tasks
#include <geometric_control/tasks/DissipativeEnergy.hpp>
#include <geometric_control/tasks/ObstacleAvoidance.hpp>
#include <geometric_control/tasks/PotentialEnergy.hpp>

using namespace geometric_control;
using namespace utils_lib;

// Define base manifold
using Manifold = manifolds::Sphere<2>;
// using Manifold = manifolds::Euclidean<2>;

// Define the nodes in the tree dynamics
using TreeManifoldsImpl = TreeManifolds<Manifold>;

// Tree Mapping
template <typename ParentManifold>
class ManifoldsMapping : public TreeManifoldsImpl {
    Eigen::VectorXd map(const Eigen::VectorXd& x, Manifold& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, Manifold& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, Manifold& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

public:
    ParentManifold* _manifold;
};

// Parent Manifold map specialization
template <>
Eigen::VectorXd ManifoldsMapping<Manifold>::map(const Eigen::VectorXd& x, Manifold& manifold)
{
    return Eigen::VectorXd::Ones(x.size());
}

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

    // Create DS on a specific manifold
    dynamics::BundleDynamics<Manifold, TreeManifoldsImpl, ManifoldsMapping> ds;

    // Assign potential and dissipative task to the DS
    ds.addTasks(std::make_unique<tasks::DissipativeEnergy<Manifold>>());
    static_cast<tasks::DissipativeEnergy<Manifold>&>(ds.task(0)).setDissipativeFactor(5 * Eigen::Matrix3d::Identity());

    ds.addTasks(std::make_unique<tasks::PotentialEnergy<Manifold>>());
    Eigen::Vector3d a = ds.manifold().embedding(Eigen::Vector2d(1.5, 3));
    static_cast<tasks::PotentialEnergy<Manifold>&>(ds.task(1)).setStiffness(Eigen::Matrix3d::Identity()).setAttractor(a);

    // Generate random obstacles over the sphere
    size_t num_obstacles = 40;
    double radius_obstacles = 0.1;
    Eigen::MatrixXd center_obstacles(num_obstacles, ds.manifold().eDim());

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr_x1(0, M_PI), distr_x2(0, 2 * M_PI);

    for (size_t i = 0; i < num_obstacles; i++) {
        ds.addTasks(std::make_unique<tasks::ObstacleAvoidance<Manifold>>());
        Eigen::Vector2d oCenter(distr_x1(eng), distr_x2(eng));
        static_cast<tasks::ObstacleAvoidance<Manifold>&>(ds.task(i + 2))
            .setRadius(radius_obstacles)
            .setCenter(oCenter)
            .setMetricParams(1, 3);
        center_obstacles.row(i) = ds.manifold().embedding(oCenter);
    }

    // Embedding & potential
    Eigen::VectorXd potential(num_samples);
    Eigen::MatrixXd embedding(num_samples, dim + 1);

    for (size_t i = 0; i < num_samples; i++) {
        embedding.row(i) = ds.manifold().embedding(X.row(i));
        potential(i) = ds.task(1).map(embedding.row(i))[0];
    }

    // Dynamics
    double time = 0, max_time = 60, dt = 0.001;
    size_t num_steps = std::ceil(max_time / dt) + 1, index = 0;
    Eigen::Matrix3d rot_mat{{1, 0, 0}, {0, 0, -1}, {0, 1, 0}};
    Eigen::Matrix3d rot_mat2{{1, 0, 0}, {0, 0, 1}, {0, -1, 0}};
    Eigen::Vector3d x1 = ds.manifold().embedding(Eigen::Vector2d(0.7, 6)),
                    v1 = ds.manifold().project(x1, (ds.manifold().embedding(Eigen::Vector2d(1.5, 3)) - x1) * 0.005);
    Eigen::Vector3d x2 = ds.manifold().embedding(Eigen::Vector2d(0.7, 6)),
                    v2 = ds.manifold().project(x2, rot_mat * (ds.manifold().embedding(Eigen::Vector2d(1.5, 3)) - x2) * 2.05);
    Eigen::Vector3d x3 = ds.manifold().embedding(Eigen::Vector2d(0.7, 6)),
                    v3 = ds.manifold().project(x3, rot_mat2 * (ds.manifold().embedding(Eigen::Vector2d(1.5, 3)) - x3) * 2.05);

    // v = ds.manifold().jacobian(Eigen::Vector2d(1, 4)) * Eigen::Vector2d(-1, 1);

    // Record
    Eigen::MatrixXd record = Eigen::MatrixXd::Zero(num_steps, 1 + 2 * (dim + 1));
    record.row(0)(0) = time;
    record.row(0).segment(1, dim + 1) = x1;
    record.row(0).segment(dim + 2, dim + 1) = v1;

    // Record 2
    Eigen::MatrixXd record2 = Eigen::MatrixXd::Zero(num_steps, 1 + 2 * (dim + 1));
    record2.row(0)(0) = time;
    record2.row(0).segment(1, dim + 1) = x2;
    record2.row(0).segment(dim + 2, dim + 1) = v2;

    // Record 3
    Eigen::MatrixXd record3 = Eigen::MatrixXd::Zero(num_steps, 1 + 2 * (dim + 1));
    record3.row(0)(0) = time;
    record3.row(0).segment(1, dim + 1) = x3;
    record3.row(0).segment(dim + 2, dim + 1) = v3;

    // std::cout << Manifold().embedding(Eigen::Vector2d(1.2, 3.5)).transpose() << std::endl;

    {
        Timer timer;
        ds(x1, v1);
    }

    while (time < max_time && index < num_steps - 1) {
        // Velocity
        // No need to project the acceleration because it is already
        // in the tangent space. Projecting in principle should not alter
        // the solution but in practice it creates numerical instability
        // close to the attractor.
        v1 = v1 + dt * ds(x1, v1);
        v2 = v2 + dt * ds(x2, v2);
        v3 = v3 + dt * ds(x3, v3);
        // v = v + dt * Manifold().project(x, ds(x, v));

        // Position
        // It also possible to avoid retraction because the acceleration profile
        // keeps the trajectory close to the manifold. Although without retraction
        // the trajectory seems to slightly leave the manifold.
        x1 = ds.manifold().retract(x1, v1, dt);
        x2 = ds.manifold().retract(x2, v2, dt);
        x3 = ds.manifold().retract(x3, v3, dt);
        // x = x + dt * v;

        // Step forward
        time += dt;
        index++;

        // Record
        record.row(index)(0) = time;
        record.row(index).segment(1, dim + 1) = x1;
        record.row(index).segment(dim + 2, dim + 1) = v1;

        record2.row(index)(0) = time;
        record2.row(index).segment(1, dim + 1) = x2;
        record2.row(index).segment(dim + 2, dim + 1) = v2;

        record3.row(index)(0) = time;
        record3.row(index).segment(1, dim + 1) = x3;
        record3.row(index).segment(dim + 2, dim + 1) = v3;
    }

    FileManager io_manager;
    io_manager.setFile("outputs/sphere_bundle.csv");
    io_manager.write("RECORD", record, "EMBEDDING", embedding, "POTENTIAL", potential, "TARGET", a, "RADIUS", radius_obstacles, "CENTER", center_obstacles, "TWO", record2, "THREE", record3);

    return 0;
}