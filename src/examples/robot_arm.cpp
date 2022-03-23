#include <iostream>

#include <beautiful_bullet/Simulator.hpp>
#include <beautiful_bullet/graphics/MagnumGraphics.hpp>
#include <utils_lib/FileManager.hpp>

// Bundle DS
#include <geometric_control/dynamics/BundleDynamics.hpp>

// Manifolds
#include <geometric_control/manifolds/Euclidean.hpp>
#include <geometric_control/manifolds/SpecialEuclidean.hpp>
#include <geometric_control/manifolds/Sphere.hpp>

// Tasks
#include <geometric_control/tasks/DissipativeEnergy.hpp>
#include <geometric_control/tasks/ObstacleAvoidance.hpp>
#include <geometric_control/tasks/PotentialEnergy.hpp>

using namespace geometric_control;
using namespace beautiful_bullet;
using namespace utils_lib;

// Define the nodes in the tree dynamics
using TreeManifoldsImpl = TreeManifolds<bodies::MultiBody, manifolds::SpecialEuclidean<3>, manifolds::Sphere<2>>;

// Tree Mapping
template <typename ParentManifold>
class ManifoldsMapping : public TreeManifoldsImpl {
    Eigen::VectorXd map(const Eigen::VectorXd& x, manifolds::Sphere<2>& manifold) override
    {
        return Eigen::VectorXd::Zero(x.size());
    }

    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, manifolds::Sphere<2>& manifold) override
    {
        return Eigen::MatrixXd::Zero(x.size(), x.size());
    }

    Eigen::Tensor<double, 3> hessian(const Eigen::VectorXd& x, manifolds::Sphere<2>& manifold) override
    {
        Eigen::Tensor<double, 3> hess(x.size(), x.size(), x.size());
        hess.setZero();
        return hess;
    }

    Eigen::VectorXd map(const Eigen::VectorXd& x, manifolds::SpecialEuclidean<3>& manifold) override
    {
        return Eigen::VectorXd::Zero(x.size());
    }

    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, manifolds::SpecialEuclidean<3>& manifold) override
    {
        return Eigen::MatrixXd::Zero(x.size(), x.size());
    }

    Eigen::Tensor<double, 3> hessian(const Eigen::VectorXd& x, manifolds::SpecialEuclidean<3>& manifold) override
    {
        Eigen::Tensor<double, 3> hess(x.size(), x.size(), x.size());
        hess.setZero();
        return hess;
    }

    Eigen::VectorXd map(const Eigen::VectorXd& x, bodies::MultiBody& manifold) override
    {
        return Eigen::VectorXd::Zero(x.size());
    }

    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, bodies::MultiBody& manifold) override
    {
        return Eigen::MatrixXd::Zero(x.size(), x.size());
    }

    Eigen::Tensor<double, 3> hessian(const Eigen::VectorXd& x, bodies::MultiBody& manifold) override
    {
        Eigen::Tensor<double, 3> hess(x.size(), x.size(), x.size());
        hess.setZero();
        return hess;
    }
};

// Parent Manifold map specialization
template <>
Eigen::VectorXd ManifoldsMapping<manifolds::Sphere<2>>::map(const Eigen::VectorXd& x, manifolds::Sphere<2>& manifold)
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
    Simulator simulator;
    // simulator.setGraphics(std::make_unique<graphics::MagnumGraphics>());
    simulator.addGround();
    bodies::MultiBody iiwa("rsc/iiwa/urdf/iiwa14.urdf");
    simulator.add(iiwa.activateGravity());
    // simulator.run();

    // dynamics::BundleDynamics<bodies::MultiBody, TreeManifoldsImpl, ManifoldsMapping> ds;

    // // Assign potential and dissipative task to the DS
    // ds.addTasks(
    //     std::make_unique<tasks::PotentialEnergy<Manifold>>(),
    //     std::make_unique<tasks::DissipativeEnergy<Manifold>>(),
    //     std::make_unique<tasks::ObstacleAvoidance<Manifold>>());

    // // Attractor
    // Eigen::Vector3d a = Manifold().embedding(Eigen::Vector2d(1.5, 3));

    // // Set tasks' properties
    // static_cast<tasks::PotentialEnergy<Manifold>&>(ds.task(0)).setStiffness(Eigen::Matrix3d::Identity()).setAttractor(a);
    // static_cast<tasks::DissipativeEnergy<Manifold>&>(ds.task(1)).setDissipativeFactor(10 * Eigen::Matrix3d::Identity());
    // static_cast<tasks::ObstacleAvoidance<Manifold>&>(ds.task(2)).setRadius(0.4).setCenter(Eigen::Vector2d(1.2, 3.5)).setMetricParams(1, 1);

    // // Embedding
    // Eigen::VectorXd potential(num_samples);
    // Eigen::MatrixXd embedding(num_samples, dim + 1);

    // for (size_t i = 0; i < num_samples; i++) {
    //     embedding.row(i) = Manifold().embedding(X.row(i));
    //     potential(i) = ds.task(0).map(embedding.row(i))[0];
    // }

    // // Dynamics
    // double time = 0, max_time = 30, dt = 0.001;
    // size_t num_steps = std::ceil(max_time / dt) + 1, index = 0;
    // Eigen::Vector3d x = Manifold().embedding(Eigen::Vector2d(0.7, 5)),
    //                 v = Manifold().projector(x, (Manifold().embedding(Eigen::Vector2d(1.5, 3)) - x) * 0.01);
    // // v = Manifold().jacobian(Eigen::Vector2d(1, 4)) * Eigen::Vector2d(-1, 1);

    // // Record
    // Eigen::MatrixXd record = Eigen::MatrixXd::Zero(num_steps, 1 + 2 * (dim + 1));
    // record.row(0)(0) = time;
    // record.row(0).segment(1, dim + 1) = x;
    // record.row(0).segment(dim + 2, dim + 1) = v;

    // std::cout << Manifold().embedding(Eigen::Vector2d(1.2, 3.5)).transpose() << std::endl;

    // while (time < max_time && index < num_steps - 1) {
    //     // Velocity
    //     v = v + dt * Manifold().projector(x, ds(x, v));

    //     // Position
    //     x = Manifold().retraction(x, v, dt);

    //     // Step forward
    //     time += dt;
    //     index++;

    //     // Record
    //     record.row(index)(0) = time;
    //     record.row(index).segment(1, dim + 1) = x;
    //     record.row(index).segment(dim + 2, dim + 1) = v;
    // }

    // FileManager io_manager;
    // io_manager.setFile("rsc/sphere_bundle.csv");
    // io_manager.write("RECORD", record, "EMBEDDING", embedding, "POTENTIAL", potential);

    return 0;
}