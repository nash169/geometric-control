#include <beautiful_bullet/Simulator.hpp>
#include <geometric_control/optimization/IDSolver.hpp>

using namespace beautiful_bullet;
using namespace geometric_control;

int main(int argc, char const* argv[])
{
    // Robot
    bodies::MultiBody iiwa("rsc/iiwa/urdf/iiwa14.urdf");

    // QP Solver
    double dt = 0.001;
    optimization::IDSolver qp(7, 6, true);

    qp.setJointPositionLimits(iiwa.lowerLimits(), iiwa.upperLimits());
    qp.setJointVelocityLimits(iiwa.velocityLimits());
    qp.setJointAccelerationLimits(100 * Eigen::VectorXd::Ones(7));
    qp.setJointTorqueLimits(iiwa.torquesLimits());

    Eigen::VectorXd tau = iiwa.torques();

    bool result = qp.step(tau, iiwa.state(), iiwa.velocity(), Eigen::VectorXd::Random(6),
        iiwa.jacobian(), iiwa.hessian(), iiwa.inertiaMatrix(), iiwa.nonLinearEffects(),
        dt);

    return 0;
}
