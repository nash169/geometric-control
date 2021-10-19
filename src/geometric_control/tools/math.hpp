#ifndef GEOMETRIC_CONTROL_TOOLS_MATH_HPP
#define GEOMETRIC_CONTROL_TOOLS_MATH_HPP

#include <unsupported/Eigen/CXX11/Tensor>

#include "geometric_control/tools/math.hpp"

namespace geometric_control {
    namespace tools {
        Eigen::Tensor<double, 3> leviCivitaConnection(const Eigen::Tensor<double, 2>& gInv, const Eigen::Tensor<double, 3>& gGrad)
        {
            Eigen::array<Eigen::IndexPair<int>, 1> c1 = {Eigen::IndexPair<int>(1, 0)},
                                                   c2 = {Eigen::IndexPair<int>(1, 2)};

            return 0.5 * (gInv.contract(gGrad + gGrad.shuffle(Eigen::array<int, 3>({0, 2, 1})), c1) - gInv.contract(gGrad, c2));
        }

        Eigen::MatrixXd pullback(const Eigen::MatrixXd& g, const Eigen::MatrixXd& df)
        {
            return df.transpose() * g * df;
        }
    } // namespace tools
} // namespace geometric_control
#endif // GEOMETRIC_CONTROL_TOOLS_MATH_HPP