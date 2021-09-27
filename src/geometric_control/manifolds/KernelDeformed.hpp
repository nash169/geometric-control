#ifndef GEOMETRIC_CONTROL_MANIFOLDS_KERNEL_DEFORMED_HPP
#define GEOMETRIC_CONTROL_MANIFOLDS_KERNEL_DEFORMED_HPP

#include <kernel_lib/utils/Expansion.hpp>

using namespace kernel_lib;

namespace geometric_control {
    namespace manifolds {
        class KernelDeformed {
        public:
            KernelDeformed()
            {
            }

            Eigen::VectorXd embedding(const Eigen::VectorXd& x) override
            {
            }

            /* Metric */
            Eigen::MatrixXd metric(const Eigen::VectorXd& x) override
            {
            }

            /* Christoffel symbols */
            Eigen::Tensor<double, 3> christoffel(const Eigen::VectorXd& x) override
            {
            }

        protected:
        };
    } // namespace manifolds
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_MANIFOLDS_KERNEL_DEFORMED_HPP