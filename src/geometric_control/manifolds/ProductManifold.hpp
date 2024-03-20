#ifndef GEOMETRIC_CONTROL_MANIFOLDS_PRODUCT_MANIFOLD_HPP
#define GEOMETRIC_CONTROL_MANIFOLDS_PRODUCT_MANIFOLD_HPP

#include "geometric_control/manifolds/AbstractManifold.hpp"

namespace geometric_control {
    namespace manifolds {
        class ProductManifold : public AbstractManifold<Eigen::Dynamic, Eigen::Dynamic> {
        public:
            ProductManifold() = default;

        protected:
        };
    } // namespace manifolds
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_MANIFOLDS_PRODUCT_MANIFOLD_HPP