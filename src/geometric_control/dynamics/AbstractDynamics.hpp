#ifndef GEOMETRIC_CONTROL_DYNAMICS_ABSTRACT_DYNAMICS_HPP
#define GEOMETRIC_CONTROL_DYNAMICS_ABSTRACT_DYNAMICS_HPP

#include <unsupported/Eigen/CXX11/Tensor>

#include "geometric_control/tools/helper.hpp"
#include "geometric_control/tools/math.hpp"

namespace geometric_control {
    namespace dynamics {
        template <typename Manifold>
        class AbstractDynamics {
        public:
            AbstractDynamics(const size_t& dim) : _M(dim), _m(1), _a(Eigen::VectorXd::Zero(dim)) {}

            ~AbstractDynamics() {}

            Manifold& manifold() { return _M; }

            AbstractDynamics& setAttractor(const Eigen::VectorXd& x)
            {
                _a = x;

                return *this;
            }

        protected:
            // Manifold
            Manifold _M;

            // Mass
            double _m;

            // Equilibrium point (chart components)
            Eigen::VectorXd _a;
        };
    } // namespace dynamics
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_DYNAMICS_ABSTRACT_DYNAMICS_HPP