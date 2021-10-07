#ifndef GEOMETRIC_CONTROL_MANIFOLDS_ABSTRACT_MANIFOLD_HPP
#define GEOMETRIC_CONTROL_MANIFOLDS_ABSTRACT_MANIFOLD_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "geometric_control/tools/helper.hpp"
#include "geometric_control/tools/math.hpp"

namespace geometric_control {
    namespace manifolds {
        class AbstractManifold {
        public:
            AbstractManifold(const u_int& dim) : _d(dim) {}

            ~AbstractManifold() {}

            /* Euclidean Embedding */
            virtual Eigen::VectorXd embedding(const Eigen::VectorXd& x) = 0;

            /* Metric */
            virtual Eigen::MatrixXd metric(const Eigen::VectorXd& x) = 0;

            /* Christoffel symbols */
            virtual Eigen::Tensor<double, 3> christoffel(const Eigen::VectorXd& x) = 0;

        protected:
            u_int _d;
        };
    } // namespace manifolds
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_MANIFOLDS_ABSTRACT_MANIFOLD_HPP