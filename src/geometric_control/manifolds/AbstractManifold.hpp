#ifndef GEOMETRIC_CONTROL_MANIFOLDS_ABSTRACTMANIFOLD_HPP
#define GEOMETRIC_CONTROL_MANIFOLDS_ABSTRACTMANIFOLD_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "geometric_control/tools/helper.hpp"
#include "geometric_control/tools/math.hpp"

namespace geometric_control {
    namespace manifolds {
        template <int N, int M>
        class AbstractManifold {
        public:
            AbstractManifold() = default;
        };
    } // namespace manifolds
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_MANIFOLDS_ABSTRACTMANIFOLD_HPP