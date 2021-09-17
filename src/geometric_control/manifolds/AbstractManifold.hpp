#ifndef GEOMETRIC_CONTROL_MANIFOLDS_ABSTRACT_MANIFOLD_HPP
#define GEOMETRIC_CONTROL_MANIFOLDS_ABSTRACT_MANIFOLD_HPP

#include <Eigen/Core>
#include <vector>

namespace geometric_control {
    namespace manifolds {
        class AbstractManifold {
        public:
            AbstractManifold(const u_int& dimension) : _d(dimension) {}

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