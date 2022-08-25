#include <iostream>
#include <random>
#include <utils_lib/FileManager.hpp>
#include <utils_lib/Timer.hpp>

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
using S2 = manifolds::Sphere<2>;
using R3 = manifolds::Euclidean<3>;

// Define the nodes in the tree dynamics
using TreeManifoldsImpl = TreeManifolds<R3, S2>;

// Tree Mapping
template <typename ParentManifold>
class ManifoldsMapping : public TreeManifoldsImpl {
    Eigen::VectorXd map(const Eigen::VectorXd& x, S2& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, S2& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, S2& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

    Eigen::VectorXd map(const Eigen::VectorXd& x, R3& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, R3& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, R3& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

public:
    std::shared_ptr<ParentManifold> _manifold;
};

// R3 -> S2
template <>
Eigen::VectorXd ManifoldsMapping<R3>::map(const Eigen::VectorXd& x, S2& manifold)
{
    return x.normalized();
}
template <>
Eigen::MatrixXd ManifoldsMapping<R3>::jacobian(const Eigen::VectorXd& x, S2& manifold)
{
    return Eigen::MatrixXd::Identity(x.size(), x.size()) / x.norm() - x * x.transpose() / std::pow(x.norm(), 3);
}
template <>
Eigen::MatrixXd ManifoldsMapping<R3>::hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, S2& manifold)
{
    return 3 * x.dot(v) / std::pow(x.norm(), 5) * (x * x.transpose())
        - (v * x.transpose() + x * v.transpose()) / std::pow(x.norm(), 3)
        - x.dot(v) / std::pow(x.norm(), 3) * Eigen::MatrixXd::Identity(x.size(), x.size());
}

int main(int argc, char** argv)
{
    // Create Euclidean 3D Space
    dynamics::BundleDynamics<R3, TreeManifoldsImpl, ManifoldsMapping> ds;

    // Create DS on a specific manifold
    dynamics::BundleDynamics<S2, TreeManifoldsImpl, ManifoldsMapping> s2;
    // s2.manifoldShared()->setRadius(0.5);
    // s2.manifoldShared()->setCenter(Eigen::Vector3d::Random());

    // Dissipative task over S2
    s2.addTasks(std::make_unique<tasks::DissipativeEnergy<S2>>());
    static_cast<tasks::DissipativeEnergy<S2>&>(s2.task(0)).setDissipativeFactor(5 * Eigen::Matrix3d::Identity());

    // Potential task over S2
    s2.addTasks(std::make_unique<tasks::PotentialEnergy<S2>>());
    Eigen::Vector3d a1 = s2.manifold().embedding(Eigen::Vector2d(1.5, 3));
    static_cast<tasks::PotentialEnergy<S2>&>(s2.task(1)).setStiffness(Eigen::Matrix3d::Identity()).setAttractor(a1);

    // Obstacles tasks over S2
    size_t num_obstacles = 2;
    double radius_obstacles = 0.1;
    Eigen::MatrixXd center_obstacles(num_obstacles, s2.manifold().eDim());

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr_x1(0, M_PI), distr_x2(0, 2 * M_PI);

    for (size_t i = 0; i < num_obstacles; i++) {
        s2.addTasks(std::make_unique<tasks::ObstacleAvoidance<S2>>());
        Eigen::Vector2d oCenter(distr_x1(eng), distr_x2(eng));
        // static_cast<tasks::ObstacleAvoidance<Manifold>&>(s2.task(i + 2))
        //     .setRadius(radius_obstacles)
        //     .setCenter(oCenter)
        //     .setMetricParams(1, 3);
        center_obstacles.row(i) = s2.manifold().embedding(oCenter);
    }

    // Dissipative task over R3
    ds.addTasks(std::make_unique<tasks::DissipativeEnergy<R3>>());
    Eigen::Matrix3d D = Eigen::MatrixXd::Identity(ds.manifoldShared()->dim(), ds.manifoldShared()->dim());
    static_cast<tasks::DissipativeEnergy<R3>&>(ds.task(0)).setDissipativeFactor(D);

    // Potential task over R3
    Eigen::Vector3d a2 = s2.manifold().center();
    ds.addTasks(std::make_unique<tasks::PotentialEnergy<R3>>());
    static_cast<tasks::PotentialEnergy<R3>&>(ds.task(1)).setStiffness(Eigen::Matrix3d::Identity()).setAttractor(a2);

    // Tree structure
    ds.addBundles(&s2);

    // Simulation
    double time = 0, max_time = 100, dt = 0.001;
    size_t num_steps = std::ceil(max_time / dt) + 1, index = 0;
    Eigen::Vector3d x = Eigen::Vector3d(1, 1, 1),
                    v = Eigen::Vector3d::Zero();

    Eigen::MatrixXd record = Eigen::MatrixXd::Zero(num_steps, 1 + 2 * ds.manifoldShared()->dim());
    record.row(0)(0) = time;
    record.row(0).segment(1, ds.manifoldShared()->dim()) = x;
    record.row(0).tail(ds.manifoldShared()->dim()) = v;

    while (time < max_time && index < num_steps - 1) {
        // Integration
        v = v + dt * ds(x, v);
        x = x + dt * v;

        // Step forward
        time += dt;
        index++;

        // Record
        record.row(index)(0) = time;
        record.row(index).segment(1, ds.manifoldShared()->dim()) = x;
        record.row(index).tail(ds.manifoldShared()->dim()) = v;
    }

    // Get S2 embedding and potential function for visualization
    double box[] = {0, M_PI, 0, 2 * M_PI};
    constexpr size_t dim = 2;
    size_t resolution = 100, num_samples = resolution * resolution;

    Eigen::MatrixXd gridX = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1),
                    gridY = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution),
                    X(num_samples, dim);
    X << Eigen::Map<Eigen::VectorXd>(gridX.data(), gridX.size()), Eigen::Map<Eigen::VectorXd>(gridY.data(), gridY.size());

    Eigen::VectorXd potential(num_samples);
    Eigen::MatrixXd embedding(num_samples, dim + 1);
    for (size_t i = 0; i < num_samples; i++) {
        embedding.row(i) = s2.manifold().embedding(X.row(i));
        potential(i) = s2.task(1).map(embedding.row(i))[0];
    }

    // Save data
    FileManager io_manager;
    io_manager.setFile("outputs/euclidean_bundle.csv");
    io_manager.write("RECORD", record, "EMBEDDING", embedding, "POTENTIAL", potential, "TARGET", a, "RADIUS", radius_obstacles, "CENTER", center_obstacles);

    return 0;
}