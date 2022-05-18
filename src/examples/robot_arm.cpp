#include <iostream>

#include <beautiful_bullet/Simulator.hpp>
#include <beautiful_bullet/graphics/MagnumGraphics.hpp>
#include <utils_lib/FileManager.hpp>
#include <utils_lib/Timer.hpp>

// QP
#include <geometric_control/optimization/IDSolver.hpp>

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

#include <geometric_control/tools/math.hpp>

using namespace geometric_control;
using namespace beautiful_bullet;
using namespace utils_lib;

// create the manifold multibody
class FrankaRobot : public bodies::MultiBody {
public:
    FrankaRobot() : bodies::MultiBody("rsc/iiwa/urdf/iiwa14.urdf"), _pose(-1.1, 0, 0, 0, 0, 0), _frame("lbr_iiwa_link_7")
    {
        setPosition(_pose(0), _pose(1), _pose(2));
    }

    static constexpr int dim() { return 7; }
    static constexpr int eDim() { return 7; }

    Eigen::VectorXd map(const Eigen::VectorXd& q)
    {
        return this->setState(q).framePose(_frame) + _pose;
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
    Eigen::Matrix<double, 6, 1> _pose;
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

    // S2
    double s2_radius = 1;
    Eigen::Vector3d s2_center(0, 0, 0); // (0.7, 0.0, 0.5);
    s2.manifold().setRadius(s2_radius).setCenter(s2_center);

    double box[] = {0, M_PI, 0, 2 * M_PI};
    size_t resolution = 100, num_samples = resolution * resolution;
    Eigen::MatrixXd gridX = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1),
                    gridY = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution),
                    X(num_samples, 2);
    X << Eigen::Map<Eigen::VectorXd>(gridX.data(), gridX.size()), Eigen::Map<Eigen::VectorXd>(gridY.data(), gridY.size());
    Eigen::VectorXd potential(num_samples);
    Eigen::MatrixXd embedding(num_samples, 3);

    s2.addTasks(std::make_unique<tasks::DissipativeEnergy<manifolds::Sphere<2>>>());
    static_cast<tasks::DissipativeEnergy<manifolds::Sphere<2>>&>(s2.task(0)).setDissipativeFactor(5 * Eigen::Matrix3d::Identity());

    Eigen::Vector3d attractor = s2.manifold().embedding(Eigen::Vector2d(1.5, 3));
    s2.addTasks(std::make_unique<tasks::PotentialEnergy<manifolds::Sphere<2>>>());
    static_cast<tasks::PotentialEnergy<manifolds::Sphere<2>>&>(s2.task(1)).setStiffness(Eigen::Matrix3d::Identity()).setAttractor(attractor);
    for (size_t i = 0; i < num_samples; i++) {
        embedding.row(i) = s2.manifold().embedding(X.row(i));
        potential(i) = s2.task(1).map(embedding.row(i))[0];
    }

    size_t num_obstacles = 1;
    double obs_radius = 0.1;
    Eigen::MatrixXd obs_centers(num_obstacles, 2), obs_centers3d(num_obstacles, 3);
    obs_centers << 1.2, 3.5;

    for (size_t i = 0; i < num_obstacles; i++) {
        // s2.addTasks(std::make_unique<tasks::ObstacleAvoidance<manifolds::Sphere<2>>>());
        // static_cast<tasks::ObstacleAvoidance<manifolds::Sphere<2>>&>(s2.task(i + 2))
        //     .setRadius(obs_radius)
        //     .setCenter(obs_centers.row(i))
        //     .setMetricParams(1, 3);
        obs_centers3d.row(i) = s2.manifold().embedding(obs_centers.row(i));
    }

    Eigen::Vector3d x = s2_radius * Eigen::Vector3d(-1, 0, 1).normalized() + s2_center,
                    v = Eigen::Vector3d::Zero(3),
                    a = Eigen::Vector3d::Zero(3);

    // SE3 (SO3)
    Eigen::MatrixXd R = geometric_control::tools::frameMatrix(-(x - s2_center).normalized());

    // ROBOT
    Eigen::VectorXd ref_q = Eigen::VectorXd::Zero(7),
                    q = robot.manifold().inverseKinematics(x, R, "lbr_iiwa_link_7", &ref_q),
                    dq = Eigen::VectorXd::Zero(7);
    robot.manifold().setState(q);

    // Simulation
    const double T = 50, dt = 1e-3;
    const size_t num_steps = std::ceil(T / dt) + 1;

    double t = 0;
    size_t step = 0;

    Simulator simulator;
    simulator.addGround();

    // Sphere manifold
    bodies::SphereParams paramsSphere;
    paramsSphere.setRadius(s2_radius - 0.2).setMass(0.0).setFriction(0.5).setColor("grey");
    bodies::RigidBody sphere("sphere", paramsSphere);

    // For the moment create a duplicate robot (this has to be fixed)
    bodies::MultiBody iiwa("rsc/iiwa/urdf/iiwa14.urdf");
    simulator.add(
        sphere.setPosition(s2_center(0), s2_center(1), s2_center(2)),
        iiwa.activateGravity().setPosition(-1.1, 0, 0).setState(q));

    simulator.setGraphics(std::make_unique<graphics::MagnumGraphics>());
    simulator.initGraphics();
    // simulator.run();

    // QP Solver
    optimization::IDSolver qp(7, 6, true);

    qp.setJointPositionLimits(iiwa.lowerLimits(), iiwa.upperLimits());
    qp.setJointVelocityLimits(iiwa.velocityLimits());
    qp.setJointAccelerationLimits(100 * Eigen::VectorXd::Ones(7));
    qp.setJointTorqueLimits(iiwa.torquesLimits());
    Eigen::VectorXd tau = iiwa.torques();

    // Record
    size_t dim = 3;
    Eigen::MatrixXd record = Eigen::MatrixXd::Zero(num_steps, 1 + 2 * dim);
    record.row(0)(0) = t;
    record.row(0).segment(1, dim) = x; // q;
    record.row(0).segment(dim + 1, dim) = v; // dq;

    {
        Timer timer;
        std::cout << robot(q, dq).transpose() << std::endl;
    }

    while (t < T && step < num_steps - 1) {
        // // Dynamics integration (robot)
        // dq = dq + dt * robot(q, dq);
        // q = q + dt * dq;

        // Dynamics integration (s2)
        a = s2(x, v);
        v = v + dt * a;
        x = s2.manifold().retract(x, v, dt); // x + dt * v;

        Eigen::Matrix<double, 6, 1> err;
        err << a, 3 * geometric_control::tools::rotationError(iiwa.frameOrientation(), geometric_control::tools::frameMatrix(-x));

        bool result = qp.step(tau, iiwa.state(), iiwa.velocity(), err,
            iiwa.jacobian(), iiwa.hessian(), iiwa.inertiaMatrix(), iiwa.nonLinearEffects(),
            dt);

        // Step forward
        // simulator.agents()[0].setState(q);
        simulator.agents()[0].setTorques(tau);
        simulator.step(t);
        t += dt;
        step++;

        // Record
        record.row(step)(0) = t;
        record.row(step).segment(1, dim) = x; // q;
        record.row(step).tail(dim) = v; // dq;
    }

    FileManager io_manager;
    io_manager.setFile("outputs/robot_bundle.csv").write(record);
    io_manager.write("RECORD", record, "TARGET", attractor, "RADIUS", obs_radius, "CENTER", obs_centers3d, "EMBEDDING", embedding, "POTENTIAL", potential);

    return 0;
}