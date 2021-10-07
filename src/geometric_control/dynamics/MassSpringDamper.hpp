#ifndef GEOMETRIC_CONTROL_DYNAMICS_MASS_SPRING_DAMPER_HPP
#define GEOMETRIC_CONTROL_DYNAMICS_MASS_SPRING_DAMPER_HPP

#include "geometric_control/dynamics/AbstractDynamics.hpp"

namespace geometric_control {
    namespace dynamics {
        template <typename Manifold>
        class MassSpringDamper : public AbstractDynamics<Manifold> {
        public:
            MassSpringDamper(const size_t& dimension) : AbstractDynamics<Manifold>(dimension)
            {
                _K = Eigen::MatrixXd::Identity(dimension, dimension);
                _D = Eigen::MatrixXd::Identity(dimension, dimension);
            }

            ~MassSpringDamper() {}

            Eigen::VectorXd operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& v)
            {
                Eigen::array<Eigen::IndexPair<int>, 1> c = {Eigen::IndexPair<int>(1, 0)};

                return -_M.metric(x).inverse() * (dissipativeForces(v) + potentialGrad(x))
                    - tools::VectorCast(_M.christoffel(x).contract(tools::TensorCast(v), c).contract(tools::TensorCast(v), c));
            }

            double potentialEnergy(const Eigen::VectorXd& x)
            {
                return 0.5 * pow(_M.distance(_a, x), 2);
            }

            Eigen::VectorXd potentialGrad(const Eigen::VectorXd& x)
            {
                return _M.distance(_a, x) * _K * _M.distanceGrad(_a, x);
            }

            Eigen::VectorXd dissipativeForces(const Eigen::VectorXd& v)
            {
                return _D * v;
            }

            MassSpringDamper& setPotentialFactor(const Eigen::MatrixXd& K)
            {
                _K = K;

                return *this;
            }

            MassSpringDamper& setDissipativeFactor(const Eigen::MatrixXd& D)
            {
                _D = D;

                return *this;
            }

        protected:
            // Manifold
            using AbstractDynamics<Manifold>::_M;

            // Mass
            using AbstractDynamics<Manifold>::_m;

            // Attractor
            using AbstractDynamics<Manifold>::_a;

            // Damping and Stiffness matrices
            Eigen::MatrixXd _D, _K;

            // Eigen::VectorXd potentialGrad(const Eigen::VectorXd& x)
            // {
            //     return _m.distance(_a, x) * _m.distanceGrad(_a, x);
            // }
        };
    } // namespace dynamics
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_DYNAMICS_MASS_SPRING_DAMPER_HPP