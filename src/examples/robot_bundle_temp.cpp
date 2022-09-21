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
#include <control_lib/controllers/QuadraticProgramming.hpp>
#include <control_lib/spatial/RN.hpp>
#include <control_lib/spatial/SE3.hpp>

using namespace geometric_control;
using namespace beautiful_bullet;
using namespace utils_lib;
using namespace control_lib;

// Controller params
struct Params {
    struct controller : public defaults::controller {
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

struct GravityCompensation {
    GravityCompensation() = default;

    size_t dimension() const { return 7; };

    GravityCompensation& setModel(const bodies::MultiBodyPtr& model)
    {
        _model = model;
        return *this;
    }

    void update(const spatial::RN<7>& state)
    {
        _eff = _model->nonLinearEffects(state._pos, state._vel);
    }

    const Eigen::Matrix<double, 7, 1>& effort() const { return _eff; }

protected:
    bodies::MultiBodyPtr _model;
    Eigen::Matrix<double, 7, 1> _eff;
};

template <typename Target>
struct ControllerQP : public control::MultiBodyCtr {
    ControllerQP(const bodies::MultiBodyPtr& model, Target& target)
        : control::MultiBodyCtr(ControlMode::CONFIGURATIONSPACE), _target(target)
    {
        // Robot state
        spatial::RN<7> state;
        state._pos = model->state();
        state._vel = model->velocity();
        state._acc = model->acceleration();
        state._eff = model->effort();

        // Set gravity compensation
        _gravity
            .setModel(model)
            .update(state);

        // Set QP
        Eigen::MatrixXd Q = 50 * Eigen::MatrixXd::Identity(7, 7),
                        Qt = 10 * Eigen::MatrixXd::Identity(7, 7),
                        R = 1 * Eigen::MatrixXd::Identity(7, 7),
                        Rt = 0.5 * Eigen::MatrixXd::Identity(7, 7),
                        W = 0 * Eigen::MatrixXd::Identity(6, 6);

        _qp.setModel(model)
            .accelerationMinimization(Q)
            .accelerationTracking(Qt, _target)
            .effortMinimization(R)
            .effortTracking(Rt, _gravity)
            .slackVariable(W)
            .modelDynamics(state)
            .accelerationLimits()
            .init();
    }

    Eigen::VectorXd action(bodies::MultiBody& body) override
    {
        // robot state
        spatial::RN<7> state;
        state._pos = body.state();
        state._vel = body.velocity();
        state._acc = body.acceleration();
        state._eff = body.effort();

        // update gravity tracker
        _gravity.update(state);

        // update robot tracker
        _target.update(state._pos, state._vel).solve();

        return _qp.action(state);
    }

protected:
    Target& _target;
    GravityCompensation _gravity;
    controllers::QuadraticProgramming<Params, bodies::MultiBody> _qp;
};

// Robot Configuration Manifold
class Robot : public bodies::MultiBody {
public:
    Robot() : bodies::MultiBody("rsc/iiwa_bullet/model.urdf"), _frame("lbr_iiwa_link_7") {}

    static constexpr int dim() { return 7; }
    static constexpr int eDim() { return 7; }

    Eigen::VectorXd map(const Eigen::VectorXd& q) { return this->framePose(q, _frame); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& q) { return static_cast<bodies::MultiBody*>(this)->jacobian(q, _frame); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& q, const Eigen::VectorXd& v) { return this->jacobianDerivative(q, v, _frame); }

protected:
    std::string _frame;
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
    return _manifold->map(x).head(3);
}
template <>
Eigen::MatrixXd ManifoldsMapping<Robot>::jacobian(const Eigen::VectorXd& x, R3& manifold)
{
    // geometric_control::tools::expTransOperator(_manifold->frameOrientation(x)) *
    // _manifold->frameOrientation(x) *
    return _manifold->frameOrientation(x) * _manifold->jacobian(x).block(0, 0, 3, 7);
}
template <>
Eigen::MatrixXd ManifoldsMapping<Robot>::hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, R3& manifold)
{
    return _manifold->frameOrientation(x) * _manifold->hessian(x, v).block(0, 0, 3, 7);
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
    Eigen::Vector3d attractor = s2_radius * Eigen::Vector3d(0, -1, 1).normalized() + s2_center;
    s2.addTasks(std::make_unique<tasks::PotentialEnergy<S2>>());
    static_cast<tasks::PotentialEnergy<S2>&>(s2.task(1)).setStiffness(Eigen::Matrix3d::Identity()).setAttractor(attractor);

    // Obstacle
    double obs_radius = 0.05;
    Eigen::Vector2d obs_center = Eigen::Vector2d(1.9, 2.0);
    Eigen::MatrixXd obs_center3d = s2.manifold().embedding(obs_center).transpose();
    s2.addTasks(std::make_unique<tasks::ObstacleAvoidance<S2>>());
    static_cast<tasks::ObstacleAvoidance<S2>&>(s2.task(2))
        .setRadius(obs_radius)
        .setCenter(obs_center)
        .setMetricParams(1, 3);

    // Initial state
    Eigen::Vector3d x = s2_radius * Eigen::Vector3d(-1, 0, 1).normalized() + s2_center,
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

    bodies::MultiBodyPtr robotPtr = robot.manifoldShared();
    (*robotPtr).addControllers(std::make_unique<ControllerQP<decltype(robot)>>(robotPtr, robot));

    simulator.add(robotPtr, sphere);

    // simulator.setGraphics(std::make_unique<graphics::MagnumGraphics>());
    // simulator.initGraphics();
    // simulator.run();

    double time = 0, max_time = 100, dt = 0.001;
    size_t num_steps = std::ceil(max_time / dt) + 1, index = 0, config_dim = 7, task_dim = 3;
    Eigen::VectorXd time_log = Eigen::VectorXd::Zero(num_steps);
    Eigen::MatrixXd config_log = Eigen::MatrixXd::Zero(num_steps, 2 * config_dim),
                    task_log = Eigen::MatrixXd::Zero(num_steps, 4 * task_dim);
    time_log(0) = time;
    config_log.row(0).head(config_dim) = q;
    config_log.row(0).tail(config_dim) = dq;
    task_log.row(0).head(task_dim) = robotPtr->framePosition(q);
    task_log.row(0).segment(task_dim, task_dim) = robotPtr->frameVelocity(dq).head(task_dim);
    task_log.row(0).segment(2 * task_dim, task_dim) = x;
    task_log.row(0).tail(task_dim) = v;

    while (time < max_time && index < num_steps - 1) {
        // Robot integration
        // simulator.step(index);

        // Configuration space integration
        dq = dq + dt * robot(q, dq);
        q = q + dt * dq;

        // Sphere space integration
        v = v + dt * r3(x, v);
        x = x + dt * v;

        // Increase step & time
        time += dt;
        index++;

        // Record
        time_log(index) = time;
        config_log.row(index).head(config_dim) = q; // robotPtr->state();
        config_log.row(index).tail(config_dim) = dq; // robotPtr->velocity();
        task_log.row(index).head(task_dim) = robotPtr->framePosition(q);
        task_log.row(index).segment(task_dim, task_dim) = robotPtr->frameVelocity(dq).head(task_dim);
        task_log.row(index).segment(2 * task_dim, task_dim) = x;
        task_log.row(index).tail(task_dim) = v;
    }

    double box[] = {0, M_PI, 0, 2 * M_PI};
    size_t resolution = 100, num_samples = resolution * resolution;
    Eigen::MatrixXd gridX = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1),
                    gridY = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution),
                    X(num_samples, 2);
    X << Eigen::Map<Eigen::VectorXd>(gridX.data(), gridX.size()), Eigen::Map<Eigen::VectorXd>(gridY.data(), gridY.size());
    Eigen::VectorXd potential(num_samples);
    Eigen::MatrixXd embedding(num_samples, 3);

    for (size_t i = 0; i < num_samples; i++) {
        embedding.row(i) = s2.manifold().embedding(X.row(i));
        potential(i) = s2.task(1).map(embedding.row(i))[0];
    }

    Eigen::MatrixXd pos_limits(7, 2), vel_limits(7, 2);
    pos_limits << robotPtr->positionLower(), robotPtr->positionUpper();
    vel_limits << robotPtr->velocityLower(), robotPtr->velocityUpper();

    FileManager io_manager;
    io_manager.setFile("outputs/robot_bundle.csv")
        .write("TIME", time_log, "CONFIG", config_log, "TASK", task_log,
            "TARGET", attractor, "RADIUS", obs_radius, "CENTER", obs_center3d,
            "EMBEDDING", embedding, "POTENTIAL", potential,
            "POSLIMITS", pos_limits, "VELLIMITS", vel_limits);

    return 0;
}