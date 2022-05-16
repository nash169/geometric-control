#include <beautiful_bullet/Simulator.hpp>
#include <geometric_control/optimization/IDSolver.hpp>

using namespace beautiful_bullet;
using namespace geometric_control;

int main(int argc, char const* argv[])
{
    // Robot
    bodies::MultiBody iiwa("rsc/iiwa/urdf/iiwa14.urdf");

    // QP Solver
    optimization::IDSolver qp(7, 6, true);

    return 0;
}
