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

            /* Get manifold dimension */
            u_int dimension() { return _d; }

            /* Euclidean Embedding */
            virtual Eigen::VectorXd embedding(const Eigen::VectorXd& x) const = 0;

            /* Jacobian of the Euclidean Embedding */
            virtual Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const = 0;

            /* Hessian of the Euclidean Embedding */
            virtual Eigen::Tensor<double, 3> hessian(const Eigen::VectorXd& x) const = 0;

            /* Metric (pulled back from the Euclidean Embedding) */
            virtual Eigen::MatrixXd metric(const Eigen::VectorXd& x) const = 0;

            /* Christoffel symbols (based on the pulled back metric) */
            virtual Eigen::Tensor<double, 3> christoffel(const Eigen::VectorXd& x) const = 0;

        protected:
            u_int _d;
        };
    } // namespace manifolds
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_MANIFOLDS_ABSTRACT_MANIFOLD_HPP