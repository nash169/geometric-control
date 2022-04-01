#include <iostream>

#include <beautiful_bullet/Simulator.hpp>
#include <beautiful_bullet/graphics/MagnumGraphics.hpp>
#include <utils_lib/FileManager.hpp>
#include <utils_lib/Timer.hpp>

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

// create the manifold multibody
class FrankaRobot : public bodies::MultiBody {
public:
    FrankaRobot() : bodies::MultiBody("rsc/iiwa/urdf/iiwa14.urdf"), _frame("lbr_iiwa_link_7") {}

    static constexpr int dim() { return 7; }
    static constexpr int eDim() { return 7; }

    Eigen::VectorXd map(const Eigen::VectorXd& q)
    {
        return this->setState(q).framePose(_frame);
    }

    Eigen::MatrixXd jacobian(const Eigen::VectorXd& q)
    {
        return this->setState(q).jacobian(_frame);
    }

    Eigen::MatrixXd hessian(const Eigen::VectorXd& q, const Eigen::VectorXd& v)
    {
        return this->setState(q).setVelocity(v).hessian(_frame);
    }

protected:
    std::string _frame;
};

// Define the nodes in the tree dynamics
using TreeManifoldsImpl = TreeManifolds<FrankaRobot, manifolds::SpecialEuclidean<3>, manifolds::Sphere<2>>;

// Tree Mapping (non-specialize; this should be automatized)
template <typename ParentManifold>
class ManifoldsMapping : public TreeManifoldsImpl {
    Eigen::VectorXd map(const Eigen::VectorXd& x, FrankaRobot& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, FrankaRobot& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, FrankaRobot& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

    Eigen::VectorXd map(const Eigen::VectorXd& x, manifolds::SpecialEuclidean<3>& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, manifolds::SpecialEuclidean<3>& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, manifolds::SpecialEuclidean<3>& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

    Eigen::VectorXd map(const Eigen::VectorXd& x, manifolds::Sphere<2>& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, manifolds::Sphere<2>& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, manifolds::Sphere<2>& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

public:
    ParentManifold* _manifold;
};

// Robot -> SE3
template <>
Eigen::VectorXd ManifoldsMapping<FrankaRobot>::map(const Eigen::VectorXd& x, manifolds::SpecialEuclidean<3>& manifold)
{
    return _manifold->map(x);
}
template <>
Eigen::MatrixXd ManifoldsMapping<FrankaRobot>::jacobian(const Eigen::VectorXd& x, manifolds::SpecialEuclidean<3>& manifold)
{
    return _manifold->jacobian(x);
}
template <>
Eigen::MatrixXd ManifoldsMapping<FrankaRobot>::hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, manifolds::SpecialEuclidean<3>& manifold)
{
    return _manifold->hessian(x, v);
}

// SE3 -> S2
template <>
Eigen::VectorXd ManifoldsMapping<manifolds::SpecialEuclidean<3>>::map(const Eigen::VectorXd& x, manifolds::Sphere<2>& manifold)
{
    return x.head(3);
}
template <>
Eigen::MatrixXd ManifoldsMapping<manifolds::SpecialEuclidean<3>>::jacobian(const Eigen::VectorXd& x, manifolds::Sphere<2>& manifold)
{
    Eigen::MatrixXd j = Eigen::MatrixXd::Zero(3, x.size());
    j.block(0, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
    return j;
}
template <>
Eigen::MatrixXd ManifoldsMapping<manifolds::SpecialEuclidean<3>>::hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, manifolds::Sphere<2>& manifold)
{
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(3, x.size());
    h.block(0, 0, 3, 3).diagonal() = v.head(3);
    return h;
}

int main(int argc, char** argv)
{
    // Create Bundle DS for each manifold
    dynamics::BundleDynamics<FrankaRobot, TreeManifoldsImpl, ManifoldsMapping> robot;
    dynamics::BundleDynamics<manifolds::SpecialEuclidean<3>, TreeManifoldsImpl, ManifoldsMapping> se3;
    dynamics::BundleDynamics<manifolds::Sphere<2>, TreeManifoldsImpl, ManifoldsMapping> s2;

    // Bundle tree structure
    robot.addBundles(&se3);
    se3.addBundles(&s2);

    // Add tasks to the leaf Bundle Dynamics over S2
    s2.addTasks(
        std::make_unique<tasks::PotentialEnergy<manifolds::Sphere<2>>>(),
        std::make_unique<tasks::DissipativeEnergy<manifolds::Sphere<2>>>());
    // std::make_unique<tasks::ObstacleAvoidance<manifolds::Sphere<2>>>());

    // Set tasks' properties
    Eigen::Vector3d a = manifolds::Sphere<2>().embedding(Eigen::Vector2d(1.5, 3));
    static_cast<tasks::PotentialEnergy<manifolds::Sphere<2>>&>(s2.task(0)).setStiffness(Eigen::Matrix3d::Identity()).setAttractor(a);
    static_cast<tasks::DissipativeEnergy<manifolds::Sphere<2>>&>(s2.task(1)).setDissipativeFactor(10 * Eigen::Matrix3d::Identity());
    // static_cast<tasks::ObstacleAvoidance<manifolds::Sphere<2>>&>(s2.task(2)).setRadius(0.4).setCenter(Eigen::Vector2d(1.2, 3.5)).setMetricParams(1, 1);

    // Record
    double time = 0, max_time = 30, dt = 0.001;
    size_t dim = 7, num_steps = std::ceil(max_time / dt) + 1, index = 0;
    Eigen::MatrixXd record = Eigen::MatrixXd::Zero(num_steps, 1 + 2 * (dim + 1));
    Eigen::VectorXd x(dim),
        v = Eigen::VectorXd::Random(7);
    x << 0., 0.7, 0.4, 0.6, 0.3, 0.5, 0.1;

    record.row(0)(0) = time;
    record.row(0).segment(1, dim + 1) = x;
    record.row(0).segment(dim + 2, dim + 1) = v;

    // std::cout << "hello" << std::endl;
    // std::cout << robot(x, v) << std::endl;

    while (time < max_time && index < num_steps - 1) {
        // Velocity
        v = v + dt * robot(x, v);

        // Position
        x = x + dt * v;

        // Step forward
        time += dt;
        index++;

        // Record
        record.row(index)(0) = time;
        record.row(index).segment(1, dim + 1) = x;
        record.row(index).segment(dim + 2, dim + 1) = v;
    }

    FileManager io_manager;
    io_manager.setFile("rsc/robot_bundle.csv").write(record);

    return 0;
}