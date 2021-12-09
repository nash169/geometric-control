#ifndef GEOMETRIC_CONTROL_MANIFOLDS_MANIFOLD_HPP
#define GEOMETRIC_CONTROL_MANIFOLDS_MANIFOLD_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "geometric_control/tools/helper.hpp"
#include "geometric_control/tools/math.hpp"

namespace geometric_control {
    namespace manifolds {
        template <int N>
        class Manifold {
        public:
            Manifold() {}

            /* Euclidean Embedding */
            virtual Eigen::Matrix<double, (N == -1) ? N : N + 1, 1> embedding(const Eigen::Matrix<double, N, 1>& x) const = 0;

            virtual ~Manifold() {}

            // /* Jacobian of the Euclidean Embedding */
            // virtual Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const = 0;

            // /* Hessian of the Euclidean Embedding */
            // virtual Eigen::Tensor<double, 3> hessian(const Eigen::VectorXd& x) const = 0;

            // /* Metric (pulled back from the Euclidean Embedding) */
            // virtual Eigen::MatrixXd metric(const Eigen::VectorXd& x) const = 0;

            // /* Christoffel symbols (based on the pulled back metric) */
            // virtual Eigen::Tensor<double, 3> christoffel(const Eigen::VectorXd& x) const = 0;

            static constexpr int dimension() { return N; }
        };
    } // namespace manifolds
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_MANIFOLDS_MANIFOLD_HPP