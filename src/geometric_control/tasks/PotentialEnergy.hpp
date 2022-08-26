#ifndef GEOMETRIC_CONTROL_TASKS_POTENTIAL_ENERGY_HPP
#define GEOMETRIC_CONTROL_TASKS_POTENTIAL_ENERGY_HPP

#include "geometric_control/tasks/AbstractTask.hpp"

namespace geometric_control {
    namespace tasks {
        template <typename Manifold>
        class PotentialEnergy : public AbstractTask<Manifold> {
        public:
            PotentialEnergy() : AbstractTask<Manifold>() {}

            // Get attractor
            Eigen::VectorXd attractor() const { return _a; }

            // Get stiffness
            Eigen::VectorXd stiffness() const { return _K; }

            // Set attractor
            PotentialEnergy& setAttractor(const Eigen::VectorXd& x)
            {
                _a = x;
                return *this;
            }

            // Set stiffness
            PotentialEnergy& setStiffness(const Eigen::MatrixXd& K)
            {
                _K = K;
                return *this;
            }

            // Space dimension
            constexpr int dim() const override
            {
                return 1;
            }

            // Optimization weight
            virtual Eigen::MatrixXd weight(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
            {
                return tools::makeMatrix(1);
            }

            // Map between configuration and task manifolds
            // x is a point on the manifold but expressed in embedding Euclidean space coordinates
            // this map is actually a multi scalar valued function on the manifold
            Eigen::VectorXd map(const Eigen::VectorXd& x) const override
            {
                return tools::makeVector(_M->dist(_a, x));
            }

            // Jacobian
            // Gradient of the scalar function. It needs to be projected onto the tangent space
            Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const override
            {
                return _M->riemannGrad(x, _M->distGrad(_a, x).transpose());
            }

            // Hessian
            Eigen::Tensor<double, 3> hessian(const Eigen::VectorXd& x) const override
            {
                return tools::TensorCast(_M->distHess(_a, x), 1, x.rows(), x.rows());
            }

            Eigen::MatrixXd hessianDir(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
            {
                // find a way not to calculate multiple times gradient
                // maybe the projection might take place in the bundle DS
                return _M->riemannHess(x, v, _M->distGrad(_a, x).transpose(), (_M->distHess(_a, x) * v).transpose());
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