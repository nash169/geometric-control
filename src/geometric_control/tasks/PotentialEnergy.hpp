#ifndef GEOMETRIC_CONTROL_TASKS_POTENTIAL_ENERGY_HPP
#define GEOMETRIC_CONTROL_TASKS_POTENTIAL_ENERGY_HPP

#include "geometric_control/tasks/AbstractTask.hpp"

namespace geometric_control {
    namespace tasks {
        template <typename Manifold>
        class PotentialEnergy : public AbstractTask<Manifold> {
        public:
            PotentialEnergy() : AbstractTask<Manifold>() {}

            // Get attractor & stiffness
            Eigen::VectorXd attractor() const { return _a; }
            Eigen::VectorXd stiffness() const { return _K; }

            // Set attractor & stiffness
            PotentialEnergy& setAttractor(const Eigen::VectorXd& x)
            {
                _a = x;
                return *this;
            }

            PotentialEnergy& setStiffness(const Eigen::MatrixXd& K)
            {
                _K = K;
                return *this;
            }

            // Optimization weight
            Eigen::MatrixXd weight(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
            {
                return tools::makeMatrix(1);
            }

            // Map between configuration and task manifolds
            Eigen::VectorXd map(const Eigen::VectorXd& x) const override
            {
                return tools::makeVector(_M.distEE(_a, x));
            }

            // Jacobian
            Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const override
            {
                return _M.distEEGrad(_a, x).transpose();
            }

            // Hessian
            Eigen::Tensor<double, 3> hessian(const Eigen::VectorXd& x) const override
            {
                return tools::TensorCast(_M.distEEHess(_a, x), 1, x.rows(), x.rows());
            }

            // Task manifold metric
            Eigen::MatrixXd metric(const Eigen::VectorXd& x) const override
            {
                return tools::makeMatrix(1.0);
            }

            // Connection functions
            Eigen::Tensor<double, 3> christoffel(const Eigen::VectorXd& x) const override
            {
                Eigen::Tensor<double, 3> gamma(1, 1, 1);
                gamma.setZero();
                return gamma;
            }

            // Energy scalar field -> (0,0) tensor field
            double energy(const Eigen::VectorXd& x) const override
            {
                double y = map(x)[0];
                return 0.5 * y * y;
            }

            // Energy gradient field -> (1,0) tensor field when sharped
            Eigen::VectorXd energyGrad(const Eigen::VectorXd& x) const override
            {
                return map(x);
            }

            // (Co-vector) field -> (0,1) tensor field
            Eigen::VectorXd field(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
            {
                return tools::makeVector(0);
            }

        protected:
            // Manifold
            using AbstractTask<Manifold>::_M;

            // Attractor
            Eigen::VectorXd _a;

            // Stiffness factor
            Eigen::MatrixXd _K;
        };
    } // namespace tasks
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_TASKS_POTENTIAL_ENERGY_HPP