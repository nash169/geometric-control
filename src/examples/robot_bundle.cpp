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

#include <control_lib/spatial/SE3.hpp>

using namespace geometric_control;
using namespace beautiful_bullet;
using namespace utils_lib;
using namespace control_lib;

// Robot Configuration Manifold
class Robot : public bodies::MultiBody {
public:
    Robot() : bodies::MultiBody("rsc/iiwa/urdf/iiwa14.urdf"), _frame("lbr_iiwa_link_7") {}

    static constexpr int dim() { return 7; }
    static constexpr int eDim() { return 7; }

    Eigen::VectorXd map(const Eigen::VectorXd& q) { return this->setState(q).framePose(_frame); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& q) { return this->setState(q).jacobian(_frame); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& q, const Eigen::VectorXd& v) { return this->setState(q).setVelocity(v).jacobianDerivative(_frame); }

protected:
    std::string _frame;
};

// Special Euclidean Group
using SE3 = manifolds::SpecialEuclidean<3>;

// 2D Sphere
using S2 = manifolds::Sphere<2>;

// Define the nodes in the tree dynamics
using TreeManifoldsImpl = TreeManifolds<Robot, SE3, S2>;

// Tree Mapping (non-specialize; this should be automatized)
template <typename ParentManifold>
class ManifoldsMapping : public TreeManifoldsImpl {
    Eigen::VectorXd map(const Eigen::VectorXd& x, Robot& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, Robot& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, Robot& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

    Eigen::VectorXd map(const Eigen::VectorXd& x, SE3& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, SE3& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, SE3& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

    Eigen::VectorXd map(const Eigen::VectorXd& x, S2& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, S2& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, S2& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

public:
    std::shared_ptr<ParentManifold> _manifold;
    // ParentManifold* _manifold;
};

// Robot -> SE3
template <>
Eigen::VectorXd ManifoldsMapping<Robot>::map(const Eigen::VectorXd& x, SE3& manifold)
{
    return _manifold->map(x);
}
template <>
Eigen::MatrixXd ManifoldsMapping<Robot>::jacobian(const Eigen::VectorXd& x, SE3& manifold)
{
    return _manifold->jacobian(x);
}
template <>
Eigen::MatrixXd ManifoldsMapping<Robot>::hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, SE3& manifold)
{
    return _manifold->hessian(x, v);
}

// SE3 -> S2
template <>
Eigen::VectorXd ManifoldsMapping<SE3>::map(const Eigen::VectorXd& x, S2& manifold)
{
    return x.head(3) / x.head(3).norm();
}
template <>
Eigen::MatrixXd ManifoldsMapping<SE3>::jacobian(const Eigen::VectorXd& x, S2& manifold)
{
    Eigen::MatrixXd j = Eigen::MatrixXd::Zero(3, x.size());
    j.block(0, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
    return j;
}
template <>
Eigen::MatrixXd ManifoldsMapping<SE3>::hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, S2& manifold)
{
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(3, x.size());
    h.block(0, 0, 3, 3).diagonal() = v.head(3);
    return h;
}

int main(int argc, char** argv)
{
    // Create Bundle DS for each manifold
    dynamics::BundleDynamics<Robot, TreeManifoldsImpl, ManifoldsMapping> robot;
    dynamics::BundleDynamics<SE3, TreeManifoldsImpl, ManifoldsMapping> se3;
    dynamics::BundleDynamics<S2, TreeManifoldsImpl, ManifoldsMapping> s2;

    // Bundle tree structure
    robot.addBundles(&se3);
    se3.addBundles(&s2);

    // S2
    double s2_radius = 0.3;
    Eigen::Vector3d s2_center(0.7, 0.0, 0.5);
    s2.manifold().setRadius(s2_radius).setCenter(s2_center);

    double box[] = {0, M_PI, 0, 2 * M_PI};
    size_t resolution = 100, num_samples = resolution * resolution;
    Eigen::MatrixXd gridX = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1),
                    gridY = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution),
                    X(num_samples, 2);
    X << Eigen::Map<Eigen::VectorXd>(gridX.data(), gridX.size()), Eigen::Map<Eigen::VectorXd>(gridY.data(), gridY.size());
    Eigen::VectorXd potential(num_samples);
    Eigen::MatrixXd embedding(num_samples, 3);

    s2.addTasks(std::make_unique<tasks::DissipativeEnergy<S2>>());
    static_cast<tasks::DissipativeEnergy<S2>&>(s2.task(0)).setDissipativeFactor(5 * Eigen::Matrix3d::Identity());

    Eigen::Vector3d attractor = s2.manifold().embedding(Eigen::Vector2d(2.5, 2.2));
    s2.addTasks(std::make_unique<tasks::PotentialEnergy<S2>>());
    static_cast<tasks::PotentialEnergy<S2>&>(s2.task(1)).setStiffness(Eigen::Matrix3d::Identity()).setAttractor(attractor);
    for (size_t i = 0; i < num_samples; i++) {
        embedding.row(i) = s2.manifold().embedding(X.row(i));
        potential(i) = s2.task(1).map(embedding.row(i))[0];
    }

    // size_t num_obstacles = 0;
    // double obs_radius = 0.1;
    // Eigen::MatrixXd obs_centers(num_obstacles, 2), obs_centers3d(num_obstacles, 3);
    // // obs_centers << 1.2, 3.5;

    // for (size_t i = 0; i < num_obstacles; i++) {
    //     s2.addTasks(std::make_unique<tasks::ObstacleAvoidance<S2>>());
    //     static_cast<tasks::ObstacleAvoidance<S2>&>(s2.task(i + 2))
    //         .setRadius(obs_radius)
    //         .setCenter(obs_centers.row(i))
    //         .setMetricParams(1, 3);
    //     obs_centers3d.row(i) = s2.manifold().embedding(obs_centers.row(i));
    // }

    Eigen::Vector3d x = s2_radius * Eigen::Vector3d(-1, 0, 1).normalized() + s2_center,
                    v = Eigen::Vector3d::Zero(3),
                    a = Eigen::Vector3d::Zero(3);

    // SE3 (SO3)
    Eigen::MatrixXd R = geometric_control::tools::frameMatrix(-(x - s2_center).normalized());

    // ROBOT
    Eigen::VectorXd q_ref(7);
    q_ref << 0, 0, 0, -M_PI / 4, 0, M_PI / 4, 0;
    Eigen::VectorXd q = robot.manifold().inverseKinematics(x, R, "lbr_iiwa_link_7", &q_ref),
                    dq = Eigen::VectorXd::Zero(7);
    // robot.manifold().setPosition(-1.1, 0, 0).setState(q);
    robot.manifold().setState(q_ref);
    std::cout << robot.manifold().framePosition().transpose() << std::endl;

    robot.addTasks(std::make_unique<tasks::DissipativeEnergy<Robot>>());
    static_cast<tasks::DissipativeEnergy<Robot>&>(robot.task(0)).setDissipativeFactor(0.5 * Eigen::MatrixXd::Identity(7, 7));

    // Simulation
    const double T = 30, dt = 1e-3;
    const size_t num_steps = std::ceil(T / dt) + 1;

    double t = 0;
    size_t steps = 0;

    Simulator simulator;
    simulator.addGround();

    // Sphere manifold
    bodies::SphereParams paramsSphere;
    paramsSphere.setRadius(s2_radius).setMass(0.0).setFriction(0.5).setColor("grey");
    std::shared_ptr<bodies::RigidBody> sphere = std::make_shared<bodies::RigidBody>("sphere", paramsSphere);
    sphere->setPosition(s2_center(0), s2_center(1), s2_center(2));

    bodies::MultiBodyPtr robotPtr = robot.manifoldShared();
    robotPtr->activateGravity();
    simulator.add(robotPtr, sphere);
    simulator.setGraphics(std::make_unique<graphics::MagnumGraphics>());

    // // QP Solver
    // optimization::IDSolver qp(7, 6, true);

    // qp.setJointPositionLimits(robotPtr->positionLower(), robotPtr->positionUpper());
    // qp.setJointVelocityLimits(robotPtr->velocityUpper());
    // qp.setJointAccelerationLimits(50 * Eigen::VectorXd::Ones(7));
    // qp.setJointTorqueLimits(robotPtr->effortUpper());
    // Eigen::VectorXd tau = robotPtr->effort();

    // // Record
    // size_t dim = 3;
    // Eigen::MatrixXd record = Eigen::MatrixXd::Zero(num_steps, 1 + 2 * dim);
    // record.row(0)(0) = t;
    // record.row(0).segment(1, dim) = x; // q;
    // record.row(0).segment(dim + 1, dim) = v; // dq;

    // Eigen::MatrixXd end_effector = Eigen::MatrixXd::Zero(num_steps, 3);

    // {
    //     Timer timer;
    //     std::cout << robot(q, dq).transpose() << std::endl;
    // }
    simulator.run();

    // // Eigen::Vector3d pos_des = x + Eigen::Vector3d(0.2, 0, 0);
    // // Eigen::Matrix3d rot_des = Eigen::Matrix3d::Identity();
    // // spatial::SE3 sDes(rot_des, pos_des);
    // Eigen::Matrix<double, 6, 7> jac = robotPtr->jacobian(),
    //                             hess = Eigen::MatrixXd::Zero(6, 7),
    //                             J, dJ;

    // Eigen::VectorXd temp = Eigen::VectorXd::Zero(6);
    // temp.head(3) = simulator.agents()[0].framePosition() + Eigen::Vector3d(0.0, 0.2, 0);

    // std::cout << "Robot Pose" << std::endl;
    // std::cout << simulator.agents()[0].framePose().transpose() << std::endl;
    // std::cout << temp.transpose() << std::endl;

    // simulator.initGraphics();

    // while (t < T && steps < num_steps - 1) {
    //     // // Dynamics integration (robot)
    //     // dq = dq + dt * robot(q, dq);
    //     // q = q + dt * dq;

    //     // Dynamics integration (s2)
    //     Eigen::Vector3d robot_pos = robotPtr->framePosition();
    //     Eigen::Matrix3d robot_rot = robotPtr->frameOrientation();

    //     a = s2(x, v); // s2(robot_pos, simulator.agents()[0].frameVelocity().head(3));
    //     v = v + dt * s2(x, v);
    //     x = s2.manifold().retract(x, v, dt); // x + dt * v;

    //     Eigen::Matrix<double, 6, 1> err;
    //     // err = 3 * (sDes - spatial::SE3(robot_rot, robot_pos)) - 3 * simulator.agents()[0].frameVelocity();

    //     Eigen::MatrixXd wRb(6, 6);
    //     wRb.setConstant(0.0);
    //     wRb.block(0, 0, 3, 3) = robot_rot;
    //     wRb.block(3, 3, 3, 3) = robot_rot;
    //     J = wRb * robotPtr->jacobian();
    //     dJ = wRb * robotPtr->jacobianDerivative();

    //     err.head(3) = 5 * a + 1 * (robot_pos - robot_pos.normalized());
    //     err.tail(3) = Eigen::Vector3d(0, 0, 0); // 3 * geometric_control::tools::rotationError(simulator.agents()[0].frameOrientation(), geometric_control::tools::frameMatrix(-x));

    //     // err = -3 * (simulator.agents()[0].framePose() - temp);
    //     // err.tail(3).setZero();
    //     err -= 0.3 * J * robotPtr->velocity();

    //     bool result = qp.step(tau, robotPtr->state(), robotPtr->velocity(), err,
    //         J, dJ, robotPtr->inertiaMatrix(), robotPtr->nonLinearEffects(),
    //         dt);

    //     // Step forward
    //     // robotPtr->setState(q);
    //     // robotPtr->setTorques(tau);
    //     simulator.step(steps);
    //     t += dt;
    //     steps++;

    //     // hess = (simulator.agents()[0].jacobian() - jac) / dt;
    //     // jac = simulator.agents()[0].jacobian();

    //     // Record
    //     record.row(steps)(0) = t;
    //     record.row(steps).segment(1, dim) = x; // q;
    //     record.row(steps).tail(dim) = v; // dq;

    //     end_effector.row(steps) = robot_pos;
    // }

    // FileManager io_manager;
    // io_manager.setFile("outputs/robot_bundle.csv").write(record);
    // io_manager.write("RECORD", record, "TARGET", attractor, "RADIUS", obs_radius, "CENTER", obs_centers3d, "EMBEDDING", embedding, "POTENTIAL", potential,
    //     "EFFECTOR", end_effector);

    return 0;
}