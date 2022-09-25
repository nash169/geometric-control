#include <iostream>
#include <memory>

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
#include <geometric_control/tasks/StateLimits.hpp>

#include <geometric_control/tools/math.hpp>

// Control
#include <control_lib/controllers/Feedback.hpp>
#include <control_lib/controllers/QuadraticProgramming.hpp>
#include <control_lib/spatial/RN.hpp>
#include <control_lib/spatial/SE3.hpp>
#include <control_lib/spatial/SO3.hpp>

using namespace geometric_control;
using namespace beautiful_bullet;
using namespace utils_lib;
using namespace control_lib;

// Controller params
struct Params {
    struct controller : public defaults::controller {
    };

    struct feedback : defaults::feedback {
        // Output dimension
        PARAM_SCALAR(size_t, d, 3);
    };

    struct quadratic_programming : public defaults::quadratic_programming {
        // State dimension
        PARAM_SCALAR(size_t, nP, 7);

        // Control input dimension
        PARAM_SCALAR(size_t, nC, 7);

        // Slack variable dimension
        PARAM_SCALAR(size_t, nS, 6);
    };
};

// Robot Configuration Manifold
class Robot : public bodies::MultiBody {
public:
    Robot() : bodies::MultiBody("rsc/iiwa_bullet/model.urdf"), _frame("lbr_iiwa_link_7") {}

    static constexpr int dim() { return 7; }
    static constexpr int eDim() { return 7; }

    Eigen::VectorXd framePosition(const Eigen::VectorXd& q)
    {
        return static_cast<bodies::MultiBody*>(this)->framePosition(q, _frame);
    }

    Eigen::MatrixXd frameOrientation(const Eigen::VectorXd& q)
    {
        return static_cast<bodies::MultiBody*>(this)->frameOrientation(q, _frame);
    }

    Eigen::VectorXd frameVelocity(const Eigen::VectorXd& q, const Eigen::VectorXd& dq)
    {
        return this->jacobian(q) * dq;
    }

    Eigen::MatrixXd jacobian(const Eigen::VectorXd& q)
    {
        Eigen::Matrix3d R = this->frameOrientation(q);
        Eigen::Matrix<double, 6, 6> R6 = Eigen::Matrix<double, 6, 6>::Zero();
        R6.block(0, 0, 3, 3) = R;
        R6.block(3, 3, 3, 3) = R;
        return static_cast<bodies::MultiBody*>(this)->jacobian(q, _frame); // R6 *
    }

    Eigen::MatrixXd jacobianDerivative(const Eigen::VectorXd& q, const Eigen::VectorXd& dq)
    {
        Eigen::MatrixXd J = static_cast<bodies::MultiBody*>(this)->jacobian(q, _frame);
        Eigen::Vector3d omega = J.block(3, 0, 3, 7) * dq;
        Eigen::Matrix3d omega_x = (Eigen::Matrix3d() << 0, -omega(2), omega(1), omega(2), 0, -omega(0), -omega(1), omega(0), 0).finished(),
                        R = this->frameOrientation(q),
                        R_dot = R * omega_x;

        Eigen::Matrix<double, 6, 6> R6 = Eigen::Matrix<double, 6, 6>::Zero(), R_dot6 = Eigen::Matrix<double, 6, 6>::Zero();
        R6.block(0, 0, 3, 3) = R;
        R6.block(3, 3, 3, 3) = R;
        R_dot6.block(0, 0, 3, 3) = R_dot;
        R_dot6.block(3, 3, 3, 3) = R_dot;

        return static_cast<bodies::MultiBody*>(this)->jacobianDerivative(q, dq, _frame); // R_dot6 * J +R6 *
    }

    // protected:
    std::string _frame;
};

template <typename Target>
struct ControllerQP : public control::MultiBodyCtr {
    ControllerQP(Target& target)
        : control::MultiBodyCtr(ControlMode::CONFIGURATIONSPACE), _target(target)
    {
        // Robot state
        spatial::RN<7> state;
        state._pos = _target.manifoldShared()->state();
        state._vel = _target.manifoldShared()->velocity();
        state._acc = _target.manifoldShared()->acceleration();
        state._eff = _target.manifoldShared()->effort();

        // Task state
        Eigen::VectorXd u = _target.manifoldShared()->frameVelocity(state._pos, state._vel);
        Eigen::Vector3d x = _target.manifoldShared()->framePosition(state._pos),
                        v = u.head(3),
                        rot_v = u.tail(3);
        Eigen::Matrix3d rot = _target.manifoldShared()->frameOrientation(state._pos);

        // Set gravity compensation
        _eff = _target.manifoldShared()->nonLinearEffects(state._pos, state._vel);

        // Set feedback ds
        _center = Eigen::Vector3d(0.7, 0.0, 0.5);
        Eigen::Matrix3d K = 1 * Eigen::Matrix3d::Identity(),
                        D = 2 * Eigen::Matrix3d::Identity();
        spatial::SO3 oDes(geometric_control::tools::rotationAlign(Eigen::Vector3d(0, 0, 1), -(x - _center).normalized()));
        oDes._vel = Eigen::Vector3d::Zero();
        _feedback.setStiffness(K).setDamping(D).setReference(oDes);

        // Set config
        _target.update(state._pos, state._vel).solve();

        // Set task
        spatial::SO3 oCurr(rot);
        oCurr._vel = rot_v;
        _target.bundle(0).solve();
        _acc << _target.bundle(0)._ddx, _feedback.action(oCurr);

        // std::cout << _acc.transpose() << std::endl;

        // Set QP
        Eigen::MatrixXd Q = 10 * Eigen::MatrixXd::Identity(7, 7),
                        Qt = 0 * Eigen::MatrixXd::Identity(7, 7),
                        R = 0.1 * Eigen::MatrixXd::Identity(7, 7),
                        Rt = 0.05 * Eigen::MatrixXd::Identity(7, 7),
                        W = 1 * Eigen::MatrixXd::Identity(6, 6);

        std::cout << "qp" << std::endl;

        _qp.setModel(target.manifoldShared())
            .accelerationMinimization(Q)
            // .accelerationTracking(Qt, _target)
            .effortMinimization(R)
            .effortTracking(Rt, *this)
            .slackVariable(W)
            .modelDynamics(state)
            .inverseDynamics(state, *this)
            .positionLimits(state)
            .velocityLimits(state)
            .accelerationLimits()
            .effortLimits()
            .init();

        // QP Walid
        _qpWalid = std::make_unique<optimization::IDSolver>(7, 6, true);
        _qpWalid->setJointPositionLimits(target.manifoldShared()->positionLower(), target.manifoldShared()->positionUpper());
        _qpWalid->setJointVelocityLimits(target.manifoldShared()->velocityUpper());
        _qpWalid->setJointAccelerationLimits(50 * Eigen::VectorXd::Ones(7));
        _qpWalid->setJointTorqueLimits(target.manifoldShared()->effortUpper());
    }

    size_t dimension() const { return 7; };
    const Eigen::Matrix<double, 6, 1>& acceleration() const { return _acc; }
    const Eigen::Matrix<double, 7, 1>& effort() const { return _eff; }

    Eigen::VectorXd action(bodies::MultiBody& body) override
    {
        // robot state
        spatial::RN<7> state;
        state._pos = body.state();
        state._vel = body.velocity();
        state._acc = body.acceleration();
        state._eff = body.effort();

        // task state
        Eigen::VectorXd u = _target.manifoldShared()->frameVelocity(state._pos, state._vel);
        Eigen::Vector3d x = _target.manifoldShared()->framePosition(state._pos),
                        v = u.head(3),
                        rot_v = u.tail(3);
        Eigen::Matrix3d rot = _target.manifoldShared()->frameOrientation(state._pos);

        // update gravity tracker
        _eff = _target.manifoldShared()->nonLinearEffects(state._pos, state._vel);

        // update robot tracker
        _target.update(state._pos, state._vel).solve();

        // update task acceleration
        spatial::SO3 oDes(geometric_control::tools::rotationAlign(Eigen::Vector3d(0, 0, 1), -(x - _center).normalized())),
            oCurr(rot);
        oDes._vel = Eigen::Vector3d::Zero();
        oCurr._vel = rot_v;

        // update acceleration
        _target.bundle(0).solve();
        _acc << _target.bundle(0)._ddx, _feedback.setReference(oDes).action(oCurr);
        // _acc.tail(3) = Eigen::Vector3d(0, 0, 0);
        // auto tau = _qp.action(state);

        Eigen::VectorXd tau = Eigen::VectorXd::Zero(7);
        bool result = _qpWalid->step(tau, state._pos, state._vel, _acc,
            _target.manifoldShared()->jacobian(state._pos), _target.manifoldShared()->jacobianDerivative(state._pos, state._vel),
            _target.manifoldShared()->inertiaMatrix(state._pos), _target.manifoldShared()->nonLinearEffects(state._pos, state._vel), 0.001);

        return tau;
    }

protected:
    Target& _target;
    controllers::Feedback<Params, spatial::SO3> _feedback;
    controllers::QuadraticProgramming<Params, Robot> _qp;

    Eigen::Vector3d _center;
    Eigen::Matrix<double, 6, 1> _acc;
    Eigen::Matrix<double, 7, 1> _eff;

    std::unique_ptr<optimization::IDSolver> _qpWalid;
};

// Special Euclidean Group
using R3 = manifolds::Euclidean<3>;

// 2D Sphere
using S2 = manifolds::Sphere<2>;

// Define the nodes in the tree dynamics
using TreeManifoldsImpl = TreeManifolds<Robot, R3, S2>;

// Tree Mapping (non-specialize; this should be automatized)
template <typename ParentManifold>
class ManifoldsMapping : public TreeManifoldsImpl {
    Eigen::VectorXd map(const Eigen::VectorXd& x, Robot& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, Robot& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, Robot& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

    Eigen::VectorXd map(const Eigen::VectorXd& x, R3& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, R3& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, R3& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

    Eigen::VectorXd map(const Eigen::VectorXd& x, S2& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, S2& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, S2& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

public:
    std::shared_ptr<ParentManifold> _manifold;
    // ParentManifold* _manifold;
};

// Robot -> R3
template <>
Eigen::VectorXd ManifoldsMapping<Robot>::map(const Eigen::VectorXd& x, R3& manifold)
{
    return _manifold->framePosition(x);
}
template <>
Eigen::MatrixXd ManifoldsMapping<Robot>::jacobian(const Eigen::VectorXd& x, R3& manifold)
{
    return _manifold->jacobian(x).block(0, 0, 3, 7);
}
template <>
Eigen::MatrixXd ManifoldsMapping<Robot>::hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, R3& manifold)
{
    return _manifold->jacobianDerivative(x, v).block(0, 0, 3, 7);
}

// R3 -> S2
template <>
Eigen::VectorXd ManifoldsMapping<R3>::map(const Eigen::VectorXd& x, S2& manifold)
{
    Eigen::VectorXd xbar = x - Eigen::Vector3d(0.7, 0.0, 0.5);
    return 0.3 * xbar.normalized() + Eigen::Vector3d(0.7, 0.0, 0.5);
}
template <>
Eigen::MatrixXd ManifoldsMapping<R3>::jacobian(const Eigen::VectorXd& x, S2& manifold)
{
    Eigen::VectorXd xbar = x - Eigen::Vector3d(0.7, 0.0, 0.5);
    return 0.3 * (Eigen::MatrixXd::Identity(xbar.size(), xbar.size()) / xbar.norm() - xbar * xbar.transpose() / std::pow(xbar.norm(), 3));
}
template <>
Eigen::MatrixXd ManifoldsMapping<R3>::hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, S2& manifold)
{
    Eigen::VectorXd xbar = x - Eigen::Vector3d(0.7, 0.0, 0.5), vbar = v;
    return 0.3 * (3 * xbar.dot(vbar) / std::pow(xbar.norm(), 5) * (xbar * xbar.transpose()) - (vbar * xbar.transpose() + xbar * vbar.transpose()) / std::pow(xbar.norm(), 3) - xbar.dot(vbar) / std::pow(xbar.norm(), 3) * Eigen::MatrixXd::Identity(xbar.size(), xbar.size()));
}

int main(int argc, char** argv)
{
    /*=====================
    |
    |   Sphere 2D Space
    |
    ======================*/
    dynamics::BundleDynamics<S2, TreeManifoldsImpl, ManifoldsMapping> s2;
    double s2_radius = 0.3;
    Eigen::Vector3d s2_center(0.7, 0.0, 0.5);
    s2.manifold().setRadius(s2_radius).setCenter(s2_center);

    // Dissipative Energy
    s2.addTasks(std::make_unique<tasks::DissipativeEnergy<S2>>());
    static_cast<tasks::DissipativeEnergy<S2>&>(s2.task(0)).setDissipativeFactor(5 * Eigen::Matrix3d::Identity());

    // Potential Energy
    Eigen::Vector3d attractor = s2_radius * Eigen::Vector3d(-1, -1, 1).normalized() + s2_center;
    s2.addTasks(std::make_unique<tasks::PotentialEnergy<S2>>());
    static_cast<tasks::PotentialEnergy<S2>&>(s2.task(1)).setStiffness(Eigen::Matrix3d::Identity()).setAttractor(attractor);

    // Obstacle
    double obs_radius = 0.05;
    Eigen::Vector3d obs_center = s2_radius * Eigen::Vector3d(-1, 0, 1.1).normalized() + s2_center;
    Eigen::MatrixXd obs_center3d = obs_center.transpose();
    s2.addTasks(std::make_unique<tasks::ObstacleAvoidance<S2>>());
    static_cast<tasks::ObstacleAvoidance<S2>&>(s2.task(2))
        .setRadius(obs_radius)
        .setCenter2(obs_center)
        .setMetricParams(1, 3);

    // Initial state
    Eigen::Vector3d x = s2_radius * Eigen::Vector3d(-1, 1, 1).normalized() + s2_center,
                    v = Eigen::Vector3d::Zero();

    /*=====================
    |
    |   SE(3) Space
    |
    ======================*/
    dynamics::BundleDynamics<R3, TreeManifoldsImpl, ManifoldsMapping> r3;
    r3.addTasks(std::make_unique<tasks::DissipativeEnergy<R3>>());
    static_cast<tasks::DissipativeEnergy<R3>&>(r3.task(0)).setDissipativeFactor(1e-8 * Eigen::MatrixXd::Identity(r3.manifoldShared()->dim(), r3.manifoldShared()->dim()));

    // Initial State
    Eigen::MatrixXd R = geometric_control::tools::rotationAlign(Eigen::Vector3d(0, 0, 1), -(x - s2_center).normalized());

    /*=====================
    |
    |   Robot Space
    |
    ======================*/
    dynamics::BundleDynamics<Robot, TreeManifoldsImpl, ManifoldsMapping> robot;

    // Dissipative Energy
    robot.addTasks(std::make_unique<tasks::DissipativeEnergy<Robot>>());
    static_cast<tasks::DissipativeEnergy<Robot>&>(robot.task(0)).setDissipativeFactor(1e-8 * Eigen::MatrixXd::Identity(7, 7));

    // Initial State
    Eigen::VectorXd q_ref = (Eigen::Matrix<double, 7, 1>() << 0, 0, 0, -M_PI / 4, 0, M_PI / 4, 0).finished();
    robot.manifold().setState(q_ref);
    Eigen::VectorXd q = robot.manifold().inverseKinematics(x, R, "lbr_iiwa_link_7"),
                    dq = Eigen::VectorXd::Zero(7);
    robot.manifold().setState(q); // .activateGravity();

    /*=====================
    |
    |   Build Tree
    |
    ======================*/
    robot.addBundles(&r3);
    r3.addBundles(&s2);

    {
        Timer timer;
        std::cout << robot(q, dq).transpose() << std::endl;
    }

    /*=====================
    |
    |   Simulation
    |
    ======================*/
    Simulator simulator;
    simulator.addGround();

    bodies::SphereParams paramsSphere;
    paramsSphere.setRadius(s2_radius - 0.05).setMass(0.0).setFriction(0.5).setColor("grey");
    std::shared_ptr<bodies::RigidBody> sphere = std::make_shared<bodies::RigidBody>("sphere", paramsSphere);
    sphere->setPosition(s2_center(0), s2_center(1), s2_center(2));

    bodies::SphereParams paramsObs;
    paramsObs.setRadius(0.04).setMass(0.0).setFriction(0.5).setColor("red");
    std::shared_ptr<bodies::RigidBody> obs = std::make_shared<bodies::RigidBody>("sphere", paramsObs);

    Eigen::Vector3d obsCenter = (s2_radius - 0.05) * (obs_center - s2_center).normalized() + s2_center;
    obs->setPosition(obsCenter(0), obsCenter(1), obsCenter(2));

    bodies::MultiBodyPtr robotPtr = robot.manifoldShared();
    (*robotPtr).addControllers(std::make_unique<ControllerQP<decltype(robot)>>(robot));

    simulator.add(robotPtr, sphere, obs); // sphere, obs

    simulator.setGraphics(std::make_unique<graphics::MagnumGraphics>());

    simulator.initGraphics();
    // static_cast<graphics::MagnumGraphics&>(simulator.graphics())
    //     .app()
    //     .camera3D()
    //     .setPose(Vector3{-5., 3., 5.});
    simulator.run();

    // double time = 0, max_time = 100, dt = 0.001;
    // size_t num_steps = std::ceil(max_time / dt) + 1, index = 0, config_dim = 7, task_dim = 3;
    // Eigen::VectorXd time_log = Eigen::VectorXd::Zero(num_steps);
    // Eigen::MatrixXd config_log = Eigen::MatrixXd::Zero(num_steps, 2 * config_dim),
    //                 task_log = Eigen::MatrixXd::Zero(num_steps, 4 * task_dim);
    // time_log(0) = time;
    // config_log.row(0).head(config_dim) = q;
    // config_log.row(0).tail(config_dim) = dq;
    // task_log.row(0).head(task_dim) = robot.manifoldShared()->framePosition(q);
    // task_log.row(0).segment(task_dim, task_dim) = robot.manifoldShared()->frameVelocity(q, dq).head(task_dim);
    // task_log.row(0).segment(2 * task_dim, task_dim) = x;
    // task_log.row(0).tail(task_dim) = v;

    // while (time < max_time && index < num_steps - 1) {
    //     // Robot integration
    //     // simulator.step(index);

    //     // Configuration space integration
    //     dq = dq + dt * robot(q, dq);
    //     q = q + dt * dq;

    //     // Sphere space integration
    //     // v = v + dt * r3(robotPtr->framePosition(q), robotPtr->frameVelocity(q, dq).head(task_dim));
    //     // x = robotPtr->framePosition(q) + dt * v;
    //     v = v + dt * r3(x, v);
    //     x = x + dt * v;

    //     // Increase step & time
    //     time += dt;
    //     index++;

    //     // Record
    //     time_log(index) = time;
    //     config_log.row(index).head(config_dim) = q; // robotPtr->state();
    //     config_log.row(index).tail(config_dim) = dq; // robotPtr->velocity();
    //     task_log.row(index).head(task_dim) = robot.manifoldShared()->framePosition(q);
    //     task_log.row(index).segment(task_dim, task_dim) = robot.manifoldShared()->frameVelocity(q, dq).head(task_dim);
    //     task_log.row(index).segment(2 * task_dim, task_dim) = x;
    //     task_log.row(index).tail(task_dim) = v;
    // }

    // double box[] = {0, M_PI, 0, 2 * M_PI};
    // size_t resolution = 100, num_samples = resolution * resolution;
    // Eigen::MatrixXd gridX = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1),
    //                 gridY = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution),
    //                 X(num_samples, 2);
    // X << Eigen::Map<Eigen::VectorXd>(gridX.data(), gridX.size()), Eigen::Map<Eigen::VectorXd>(gridY.data(), gridY.size());
    // Eigen::VectorXd potential(num_samples);
    // Eigen::MatrixXd embedding(num_samples, 3);

    // for (size_t i = 0; i < num_samples; i++) {
    //     embedding.row(i) = s2.manifold().embedding(X.row(i));
    //     potential(i) = s2.task(1).map(embedding.row(i))[0];
    // }

    // Eigen::MatrixXd pos_limits(7, 2), vel_limits(7, 2);
    // pos_limits << robotPtr->positionLower(), robotPtr->positionUpper();
    // vel_limits << robotPtr->velocityLower(), robotPtr->velocityUpper();

    // FileManager io_manager;
    // io_manager.setFile("outputs/robot_bundle.csv")
    //     .write("TIME", time_log, "CONFIG", config_log, "TASK", task_log,
    //         "TARGET", attractor, "RADIUS", obs_radius, "CENTER", obs_center3d,
    //         "EMBEDDING", embedding, "POTENTIAL", potential,
    //         "POSLIMITS", pos_limits, "VELLIMITS", vel_limits);

    return 0;
}