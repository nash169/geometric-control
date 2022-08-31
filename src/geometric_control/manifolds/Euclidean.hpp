#ifndef GEOMETRICCONTROL_MANIFOLDS_EUCLIDEAN_HPP
#define GEOMETRICCONTROL_MANIFOLDS_EUCLIDEAN_HPP

#include "geometric_control/manifolds/AbstractManifold.hpp"

namespace geometric_control {
    namespace manifolds {
        template <int N, int M = 1>
        class Euclidean : public AbstractManifold<N, M> {
        public:
            Euclidean()
            {
                if constexpr (N != Eigen::Dynamic) {
                    _c.setZero(N + 1);
                    _Y.setZero();
                    _Y.block(0, 0, N, N) = Eigen::MatrixXd::Identity(N, N);
                }
            }

            // Manifold dimension
            static constexpr int dim()
            {
                return N;
            }

            // Euclidean embedding space dimension
            static constexpr int eDim()
            {
                if constexpr (N != Eigen::Dynamic)
                    return N + 1;
                else
                    return N;
            }

            // Get center
            const Eigen::Matrix<double, eDim(), 1>& center() { return _c; }

            // Get frame
            const Eigen::Matrix<double, eDim(), dim()>& frame() { return _Y; }

            // Set center
            Euclidean& setCenter(const Eigen::Matrix<double, eDim(), 1>& center)
            {
                _c = center;
                return *this;
            }

            // Set frame
            Euclidean& setFrame(const Eigen::Matrix<double, eDim(), dim()>& frame)
            {
                _Y = frame;
                return *this;
            }

            /*
            |
            |   EMBEDDED GEOMETRY
            |
            */

            // Euclidean metric
            virtual Eigen::Matrix<double, eDim(), eDim()> metric(const Eigen::Matrix<double, eDim(), 1>& x) const
            {
                if constexpr (N != Eigen::Dynamic)
                    return Eigen::MatrixXd::Identity(eDim(), eDim());
                else
                    return Eigen::MatrixXd::Identity(x.size(), x.size());
            }

            // Dot (inner) product in the Euclidean embedding space
            double inner(const Eigen::Matrix<double, eDim(), 1>& x, const Eigen::Matrix<double, eDim(), 1>& u, const Eigen::Matrix<double, eDim(), 1>& v) const
            {
                return u.transpose() * metric(x) * v;
            }

            // Norm in the Euclidean embedding space
            double norm(const Eigen::Matrix<double, eDim(), 1>& x, const Eigen::Matrix<double, eDim(), 1>& u) const
            {
                return std::sqrt(inner(x, u, u));
            }

            // Projector (project over the tangent space in the Euclidean embedding)
            Eigen::Matrix<double, eDim(), 1> project(const Eigen::Matrix<double, eDim(), 1>& x, const Eigen::Matrix<double, eDim(), 1>& u) const
            {
                return (_Y * _Y.transpose()) * u;
            }

            // Retraction
            Eigen::Matrix<double, eDim(), 1> retract(const Eigen::Matrix<double, eDim(), 1>& x, const Eigen::Matrix<double, eDim(), 1>& u, const double& t = 1) const
            {
                return project(x, x + t * u) + _c;
            }

            // Distance in the Euclidean embedding
            double dist(const Eigen::Matrix<double, dim(), 1>& x, const Eigen::Matrix<double, dim(), 1>& y) const
            {
                return (x - y).norm();
            }

            // Distance gradient in the Euclidean embedding (here differential components and the gradient coincides due to the linearity of the space)
            Eigen::Matrix<double, dim(), 1> distGrad(const Eigen::Matrix<double, dim(), 1>& x, const Eigen::Matrix<double, dim(), 1>& y) const
            {
                return (y - x) / (x - y).norm();
            }

            // Distance hessian in the Euclidean space (same as for the gradient)
            Eigen::Matrix<double, dim(), dim()> distHess(const Eigen::Matrix<double, dim(), 1>& x, const Eigen::Matrix<double, dim(), 1>& y) const
            {
                Eigen::Matrix<double, dim(), 1> d = x - y;

                return (Eigen::MatrixXd::Identity(dim(), dim()) - d * d.transpose() / d.squaredNorm()) / d.norm();
            }

            Eigen::MatrixXd riemannGrad(const Eigen::Matrix<double, dim(), 1>& x, const Eigen::MatrixXd& grad) const
            {
                // Eigen::MatrixXd rGrad(grad.rows(), grad.cols());

                // for (size_t i = 0; i < grad.rows(); i++)
                //     rGrad.row(i) = project(x, grad.row(i));

                // return rGrad;
                return grad;
            }

            Eigen::MatrixXd riemannHess(const Eigen::Matrix<double, dim(), 1>& x, const Eigen::Matrix<double, dim(), 1>& v, const Eigen::MatrixXd& grad, const Eigen::MatrixXd& hess) const
            {
                // Eigen::MatrixXd rHess(hess.rows(), hess.cols());

                // for (size_t i = 0; i < hess.rows(); i++)
                //     rHess.row(i) = project(x, hess.row(i));

                // return rHess;
                return hess;
            }

            /*
            |
            |   CHART DEPENDENT GEOMETRY (char geometry might be enterely implemented through visitor pattern)
            |
            */

            // Embedding
            Eigen::Matrix<double, eDim(), 1> embedding(const Eigen::Matrix<double, dim(), 1>& x) const
            {
                return _Y * x + _c;
            }

        protected:
            // Stiefel manifold subspace representation
            Eigen::Matrix<double, eDim(), dim()> _Y;

            // Subspace center
            Eigen::Matrix<double, eDim(), 1> _c;
        };
    } // namespace manifolds
} // namespace geometric_control

#endif // GEOMETRICCONTROL_MANIFOLDS_EUCLIDEAN_HPP