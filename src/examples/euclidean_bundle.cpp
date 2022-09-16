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
    return x.normalized(); // + Eigen::Vector3d(0.7, 0.0, 0.5);
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

// Potential and Dissipative Energy R3 Specialization
class PotentialEnergyR3 : public tasks::PotentialEnergy<R3> {
public:
    PotentialEnergyR3() : tasks::PotentialEnergy<R3>(), _a(30), _b(20), _r(0.3), _c(Eigen::Vector3d(0.7, 0.0, 0.5)) {}

    Eigen::MatrixXd weight(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
    {
        double d = (x - _c).norm() / _r;
        return tools::makeMatrix(1 / (1 + std::exp(_a - _b * d)));
        // return tools::makeMatrix(1);
    }

protected:
    // Modulation params
    double _a, _b;

    // Sphere params
    double _r;
    Eigen::Vector3d _c;
};

class DissipativeEnergyR3 : public tasks::DissipativeEnergy<R3> {
public:
    DissipativeEnergyR3() : tasks::DissipativeEnergy<R3>(), _a(30), _b(20), _r(0.3), _c(Eigen::Vector3d(0.7, 0.0, 0.5)) {}

    Eigen::MatrixXd weight(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
    {
        double d = (x - _c).norm() / _r;
        return (0.1 + 1 / (1 + std::exp(_a - _b * d))) * Eigen::MatrixXd::Identity(x.rows(), x.rows());
    }

protected:
    // Modulation params
    double _a, _b;

    // Sphere params
    double _r;
    Eigen::Vector3d _c;
};

// Potential and Dissipative Energy S2 Specialization
class PotentialEnergyS2 : public tasks::PotentialEnergy<S2> {
public:
    PotentialEnergyS2() : tasks::PotentialEnergy<S2>(), _tau(1e-2), _r(0.3), _c(Eigen::Vector3d(0.7, 0.0, 0.5)) {}

    Eigen::MatrixXd weight(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
    {
        double d = (x - _c).norm() / _r;
        return tools::makeMatrix(std::exp(-std::pow(d - 1, 2) / _tau));
        // return tools::makeMatrix(0);
    }

protected:
    // Modulation params
    double _tau;

    // Sphere params
    double _r;
    Eigen::Vector3d _c;
};

class DissipativeEnergyS2 : public tasks::DissipativeEnergy<S2> {
public:
    DissipativeEnergyS2() : tasks::DissipativeEnergy<S2>(), _tau(1e-2), _r(0.3), _c(Eigen::Vector3d(0.7, 0.0, 0.5)) {}

    Eigen::MatrixXd weight(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
    {
        double d = (x - _c).norm() / _r;
        return std::exp(-std::pow(d - 1, 2) / _tau) * Eigen::MatrixXd::Identity(x.rows(), x.rows());
    }

protected:
    // Modulation params
    double _tau;

    // Sphere params
    double _r;
    Eigen::Vector3d _c;
};

int main(int argc, char** argv)
{
    /*=====================
    |
    |   Sphere 2D Space
    |
    ======================*/
    dynamics::BundleDynamics<S2, TreeManifoldsImpl, ManifoldsMapping> s2;
    double s2_radius = 0.3;
    Eigen::Vector3d s2_center = Eigen::Vector3d(0.7, 0.0, 0.5);
    s2.manifoldShared()->setRadius(s2_radius);
    s2.manifoldShared()->setCenter(s2_center);

    // Dissipative task
    s2.addTasks(std::make_unique<tasks::DissipativeEnergy<S2>>());
    static_cast<tasks::DissipativeEnergy<S2>&>(s2.task(0)).setDissipativeFactor(5 * Eigen::Matrix3d::Identity());

    // Potential task
    s2.addTasks(std::make_unique<tasks::PotentialEnergy<S2>>());
    Eigen::Vector3d a = s2_radius * Eigen::Vector3d(0, -1, 1).normalized() + s2_center;
    static_cast<tasks::PotentialEnergy<S2>&>(s2.task(1)).setStiffness(1 * Eigen::Matrix3d::Identity()).setAttractor(a);

    // Obstacles tasks
    size_t num_obstacles = 1;
    double radius_obstacles = 0.05;
    Eigen::MatrixXd center_obstacles(num_obstacles, s2.manifold().eDim());

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr_x1(0, M_PI), distr_x2(0, 2 * M_PI);

    for (size_t i = 0; i < num_obstacles; i++) {
        // s2.addTasks(std::make_unique<tasks::ObstacleAvoidance<S2>>());
        // Eigen::Vector2d oCenter(distr_x1(eng), distr_x2(eng));
        Eigen::Vector2d oCenter = Eigen::Vector2d(1.9, 2.0);
        // static_cast<tasks::ObstacleAvoidance<S2>&>(s2.task(i + 2))
        //     .setRadius(radius_obstacles)
        //     .setCenter(oCenter)
        //     .setMetricParams(1, 3);
        center_obstacles.row(i) = s2.manifold().embedding(oCenter);
    }

    /*=====================
    |
    |   Euclidean 3D Space
    |
    ======================*/
    dynamics::BundleDynamics<R3, TreeManifoldsImpl, ManifoldsMapping> r3;

    // Dissipative task
    r3.addTasks(std::make_unique<tasks::DissipativeEnergy<R3>>());
    Eigen::Matrix3d D = 1e-8 * Eigen::MatrixXd::Identity(r3.manifoldShared()->dim(), r3.manifoldShared()->dim());
    //                 U = tools::frameMatrix(s2_center);

    // Eigen::Matrix3d D;
    // D << 1e-8, 0, 0,
    //     0, 1e-8, 0,
    //     0, 0, 1e-8;
    static_cast<tasks::DissipativeEnergy<R3>&>(r3.task(0)).setDissipativeFactor(D);

    // // Potential task
    // Eigen::Vector3d a2 = s2.manifold().center();
    // r3.addTasks(std::make_unique<PotentialEnergyR3>());
    // static_cast<PotentialEnergyR3&>(r3.task(1)).setStiffness(1 * Eigen::Matrix3d::Identity()).setAttractor(a2);

    /*=====================
    |
    |   Build Tree
    |
    ======================*/
    r3.addBundles(&s2);

    /*=====================
    |
    |   Simulation
    |
    ======================*/
    double time = 0, max_time = 100, dt = 0.001;
    size_t num_steps = std::ceil(max_time / dt) + 1, index = 0;
    Eigen::Vector3d x = (s2_radius + 0.05) * Eigen::Vector3d(-1, 0, 1).normalized() + s2_center,
                    v = Eigen::Vector3d::Zero();

    {
        Timer timer;
        r3(x, v);
    }

    Eigen::MatrixXd record = Eigen::MatrixXd::Zero(num_steps, 1 + 2 * r3.manifoldShared()->dim());
    record.row(0)(0) = time;
    record.row(0).segment(1, r3.manifoldShared()->dim()) = x;
    record.row(0).tail(r3.manifoldShared()->dim()) = v;

    while (time < max_time && index < num_steps - 1) {
        // Integration
        v = v + dt * r3(x, v);
        x = x + dt * v;

        // Step forward
        time += dt;
        index++;

        // Record
        record.row(index)(0) = time;
        record.row(index).segment(1, r3.manifoldShared()->dim()) = x;
        record.row(index).tail(r3.manifoldShared()->dim()) = v;
    }

    /*=====================
    |
    |   Visualization Data
    |
    ======================*/
    double box[] = {0, M_PI, 0, 2 * M_PI};
    constexpr size_t dim = 2;
    size_t resolution = 100, num_samples = resolution * resolution;

    Eigen::MatrixXd gridX = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1),
                    gridY = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution),
                    X(num_samples, dim);
    X << Eigen::Map<Eigen::VectorXd>(gridX.data(), gridX.size()), Eigen::Map<Eigen::VectorXd>(gridY.data(), gridY.size());

    // Get S2 embedding and potential function for visualization
    Eigen::VectorXd potential(num_samples);
    Eigen::MatrixXd embedding(num_samples, dim + 1);
    for (size_t i = 0; i < num_samples; i++) {
        embedding.row(i) = s2.manifold().embedding(X.row(i));
        potential(i) = s2.task(1).map(embedding.row(i))[0];
    }

    /*=====================
    |
    |   Save Data
    |
    ======================*/
    FileManager io_manager;
    io_manager.setFile("outputs/euclidean_bundle.csv");
    io_manager.write("RECORD", record, "EMBEDDING", embedding, "POTENTIAL", potential, "TARGET", a, "RADIUS", radius_obstacles, "CENTER", center_obstacles);

    return 0;
}