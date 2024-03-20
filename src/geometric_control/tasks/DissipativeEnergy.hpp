#ifndef GEOMETRIC_CONTROL_TASKS_DISSIPATIVE_ENERGY_HPP
#define GEOMETRIC_CONTROL_TASKS_DISSIPATIVE_ENERGY_HPP

#include "geometric_control/tasks/AbstractTask.hpp"

namespace geometric_control {
    namespace tasks {
        template <typename Manifold>
        class DissipativeEnergy : public AbstractTask<Manifold> {
        public:
            DissipativeEnergy() : AbstractTask<Manifold>() {}

            // Get dissipative factor
            Eigen::MatrixXd dissipativeFactor() { return _D; }

            // Set dissipative factor
            DissipativeEnergy& setDissipativeFactor(const Eigen::MatrixXd& D)
            {
                _D = D;
                return *this;
            }

            // Space dimension
            constexpr int dim() const override
            {
                return _M->eDim();
            }

            // Optimization weight
            virtual Eigen::MatrixXd weight(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
            {
                return Eigen::MatrixXd::Identity(x.rows(), x.rows());
            }

            // Map between configuration and task manifolds
            Eigen::VectorXd map(const Eigen::VectorXd& x) const override
            {
                return x;
            }

            // Jacobian
            Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const override
            {
                return Eigen::MatrixXd::Identity(x.rows(), x.rows());
            }

            // Hessian
            Eigen::Tensor<double, 3> hessian(const Eigen::VectorXd& x) const override
            {
                Eigen::Tensor<double, 3> hess(x.rows(), x.rows(), x.rows());
                hess.setZero();
                return hess;
            }

            Eigen::MatrixXd hessianDir(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
            {
                return Eigen::MatrixXd::Zero(x.rows(), x.rows());
            }

            // Task manifold metric
            Eigen::MatrixXd metric(const Eigen::VectorXd& x) const override
            {
                return Eigen::MatrixXd::Identity(x.rows(), x.rows());
            }

            Eigen::MatrixXd metricInv(const Eigen::VectorXd& x) const override
            {
                return Eigen::MatrixXd::Identity(x.rows(), x.rows());
            }

            // Connection functions
            Eigen::Tensor<double, 3> christoffel(const Eigen::VectorXd& x) const override
            {
                Eigen::Tensor<double, 3> gamma(x.rows(), x.rows(), x.rows());
                gamma.setZero();
                return gamma;
            }

            Eigen::MatrixXd christoffelDir(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
            {
                return Eigen::MatrixXd::Identity(x.rows(), x.rows());
            }

            // Energy scalar field -> (0,0) tensor field
            double energy(const Eigen::VectorXd& x) const override
            {
                return 0;
            }

            // Energy gradient field -> (1,0) tensor field when sharped
            Eigen::VectorXd energyGrad(const Eigen::VectorXd& x) const override
            {
                return Eigen::VectorXd::Zero(x.rows());
            }

            // (Co-vector) field -> (0,1) tensor field
            Eigen::VectorXd field(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
            {
                return _D * v; //_D * (jacobian(x) * v);
            }

        protected:
            // Manifold
            using AbstractTask<Manifold>::_M;

            // Dissipative factor
            Eigen::MatrixXd _D;
        };
    } // namespace tasks
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_TASKS_DISSIPATIVE_ENERGY_HPP