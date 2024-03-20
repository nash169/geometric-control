#ifndef GEOMETRIC_CONTROL_MANIFOLDS_KERNEL_DEFORMED_HPP
#define GEOMETRIC_CONTROL_MANIFOLDS_KERNEL_DEFORMED_HPP

#include <kernel_lib/kernels/SquaredExp.hpp>
#include <kernel_lib/utils/Expansion.hpp>

#include "geometric_control/manifolds/AbstractManifold.hpp"

using namespace kernel_lib;

namespace geometric_control {
    namespace manifolds {
        template <typename Params, typename Kernel = kernels::SquaredExp<Params>>
        class KernelDeformed : public AbstractManifold<Eigen::Dynamic, 1> {
        public:
            KernelDeformed() = default;

            // Set the space deformation points and weights
            KernelDeformed& setDeformations(const Eigen::MatrixXd& x, const Eigen::VectorXd w)
            {
                _f.setSamples(x).setWeights(w);
                return *this;
            }

            // Embedding
            Eigen::VectorXd embedding(const Eigen::VectorXd& x) const
            {
                Eigen::VectorXd y(x.rows() + 1);
                y.segment(0, x.rows()) = x;
                y(y.rows() - 1) = _f(x);

                return y;
            }

            // Jacobian
            Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const
            {
                // Dimension
                size_t d = x.rows();

                Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(d + 1, d);

                jac.block(0, 0, d, d).diagonal().array() += 1;

                jac.row(d) = _f.grad(x);

                return jac;
            }

            // Hessian
            Eigen::Tensor<double, 3> hessian(const Eigen::VectorXd& x) const
            {
                Eigen::Tensor<double, 3> hess(x.size() + 1, x.size(), x.size());
                return hess;
            }

            // Metric
            Eigen::MatrixXd metric(const Eigen::VectorXd& x) const
            {
                Eigen::MatrixXd g = _f.grad(x) * _f.grad(x).transpose();
                g.diagonal().array() += 1;

                return g;
            }

            // Christoffel symbols
            Eigen::Tensor<double, 3> christoffel(const Eigen::VectorXd& x) const
            {
                return tools::leviCivitaConnection(tools::TensorCast(metric(x).inverse()), metricGrad(x));
            }

        protected:
            utils::Expansion<Params, Kernel> _f;

            Eigen::Tensor<double, 3> metricGrad(const Eigen::VectorXd& x) const
            {
                // Dimension
                u_int d = x.rows();

                Eigen::Tensor<double, 3> grad(d, d, d);
                grad.setZero();

                Eigen::VectorXd value = 2 * _f.hess(x).diagonal().array() * _f.grad(x).array();

                for (u_int i = 0; i < d; i++)
                    grad(i, i, i) = value(i);

                return grad;
            }
        };
    } // namespace manifolds
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_MANIFOLDS_KERNEL_DEFORMED_HPP