#ifndef GEOMETRIC_CONTROL_MANIFOLDS_SPHEREN_HPP
#define GEOMETRIC_CONTROL_MANIFOLDS_SPHEREN_HPP

#include "geometric_control/manifolds/Manifold.hpp"
#include <complex>

namespace geometric_control {
    namespace manifolds {
        template <int N = Eigen::Dynamic>
        class SphereN : public Manifold<N> {
        public:
            SphereN() : _r(1.0)
            {
                // Center coordinate in N+1 Euclidean space
                _c.setZero(N + 1);
            }

            // Get radius
            const double& radius() { return _r; }

            // Get center
            const Eigen::Matrix<double, N, 1>& center() { return _c; }

            // Set radius
            SphereN& setRadius(const double& radius)
            {
                _r = radius;
                return *this;
            }

            // Set center
            SphereN& setCenter(const Eigen::Matrix<double, N + 1, 1>& center)
            {
                _c = center;
                return *this;
            }

            // Embedding
            Eigen::Matrix<double, (N == -1) ? N : N + 1, 1> embedding(const Eigen::Matrix<double, N, 1>& x) const override
            {
                // std::cout << "General template" << std::endl;
                decltype(embedding(std::declval<const Eigen::Matrix<double, N, 1>&>())) y = Eigen::VectorXd::Ones(x.rows() + 1);

                for (size_t i = 0; i < x.rows() + 1; i++)
                    if (i != x.rows())
                        for (size_t j = 0; j < i + 1; j++)
                            y(i) *= (j == i) ? cos(x(j)) : sin(x(j));
                    else
                        for (size_t j = 0; j < i; j++)
                            y(i) *= sin(x(j));

                return y;
            }

            // Jacobian
            Eigen::Matrix<double, N + 1, N> jacobian(const Eigen::Matrix<double, N, 1>& x) const
            {
                // std::cout << "General template" << std::endl;
                Eigen::Matrix<double, N + 1, N> jac = Eigen::MatrixXd::Zero(N + 1, N);

                // Terrible way of calculating this jacobian...
                for (size_t i = 0; i < N + 1; i++)
                    if (i != N)
                        for (size_t j = 0; j < i + 1; j++) {
                            jac(i, j) = 1;
                            for (size_t k = 0; k < i + 1; k++)
                                if (k == j)
                                    jac(i, j) *= (k == i) ? -sin(x(k)) : cos(x(k));
                                else
                                    jac(i, j) *= (k == i) ? cos(x(k)) : sin(x(k));
                        }
                    else
                        for (size_t j = 0; j < i; j++) {
                            jac(i, j) = 1;
                            for (size_t k = 0; k < i; k++)
                                jac(i, j) *= (k == j) ? cos(x(k)) : sin(x(k));
                        }

                return jac;
            }

            // Hessian
            Eigen::TensorFixedSize<double, Eigen::Sizes<N + 1, N, N>> hessian(const Eigen::Matrix<double, N, 1>& x) const
            {
                // std::cout << "General template" << std::endl;
                Eigen::TensorFixedSize<double, Eigen::Sizes<N + 1, N, N>> hess;

                hess.setZero();

                return hess;
            }

            // Metric
            Eigen::Matrix<double, N, N> metric(const Eigen::Matrix<double, N, 1>& x) const
            {
                // std::cout << "General template" << std::endl;
                Eigen::Matrix<double, N, N> g;

                g(0, 0) = 1;

                for (size_t i = 1; i < N; i++)
                    g(i, i) = x.segment(0, i).array().sin().square().prod();

                return g;
            }

            // Metric Gradient
            Eigen::TensorFixedSize<double, Eigen::Sizes<N, N, N>> metricGrad(const Eigen::Matrix<double, N, 1>& x) const
            {
                // std::cout << "General template" << std::endl;
                Eigen::TensorFixedSize<double, Eigen::Sizes<N, N, N>> grad;
                grad.setZero();

                for (u_int i = 1; i < N; i++)
                    for (u_int j = 0; j < i; j++) {
                        grad(i, i, j) = 1;

                        for (u_int k = 0; k < i; k++)
                            grad(i, i, j) *= (k == j) ? 2 * sin(x(k)) * cos(x(k)) : sin(x(k)) * sin(x(k));
                    }

                return grad;
            }

            // Christoffel symbols
            Eigen::TensorFixedSize<double, Eigen::Sizes<2, 2, 2>> christoffel(const Eigen::Matrix<double, N, 1>& x) const
            {
                // std::cout << "General template" << std::endl;
                return tools::leviCivitaConnection(tools::TensorCast(metric(x).inverse()), metricGrad(x));
            }

            // Distance
            double distance(const Eigen::Matrix<double, N, 1>& x, const Eigen::Matrix<double, N, 1>& y)
            {
                std::complex<double> d = (embedding(x) - embedding(y)).norm();

                return 2 * asin(0.5 * d).real();
            }

            // Distance gradient (this actually the components of pullback of the differential of the distance function with
            // respect to the second entry). Sharping the operator we should get the real gradient of the distance function
            Eigen::Matrix<double, N, 1> distanceGrad(const Eigen::Matrix<double, N, 1>& x, const Eigen::Matrix<double, N, 1>& y)
            {
                double p = embedding(x).transpose() * embedding(y);

                return -1 / sqrt(1 - p * p) * embedding(x).transpose() * jacobian(y);
            }

            // Distance hessian (the same as above)
            Eigen::Matrix<double, N, N> distanceHess(const Eigen::Matrix<double, N, 1>& x, const Eigen::Matrix<double, N, 1>& y)
            {
                double p = embedding(x).transpose() * embedding(y), c = -1 / sqrt(1 - p * p);

                // row-row contraction
                Eigen::array<Eigen::IndexPair<int>, 1> contraction = {Eigen::IndexPair<int>(0, 0)};

                return pow(c, 3) * p * jacobian(y).transpose() * embedding(x) * embedding(x).transpose() * jacobian(y)
                    + c * tools::MatrixCast(hessian(y).contract(tools::TensorCast(embedding(x)), contraction), N, N);
            }

            /* Euclidean Embedding */

            // Projector (project over the tangent space in the Euclidean embedding)
            Eigen::Matrix<double, (N == -1) ? N : N + 1, 1> projector(const Eigen::Matrix<double, (N == -1) ? N : N + 1, 1>& x, const Eigen::Matrix<double, (N == -1) ? N : N + 1, 1>& u) const
            {
                return u - (x.transpose() * u) * x;
            }

            Eigen::Matrix<double, (N == -1) ? N : N + 1, 1> retraction(const Eigen::Matrix<double, (N == -1) ? N : N + 1, 1>& x, const Eigen::Matrix<double, (N == -1) ? N : N + 1, 1>& u, const double& t = 1) const
            {
                return (x + t * u) / (x + t * u).norm();
            }

            // Distance in the Euclidean embedding
            double distEE(const Eigen::Matrix<double, (N == -1) ? N : N + 1, 1>& x, const Eigen::Matrix<double, (N == -1) ? N : N + 1, 1>& y) const
            {
                std::complex<double> d = (x - y).norm();

                return 2 * asin(0.5 * d).real();
            }

            // Distance gradient in the Euclidean embedding (here differential components and the gradient coincides to the linearity of the space)
            Eigen::Matrix<double, (N == -1) ? N : N + 1, 1> distEEGrad(const Eigen::Matrix<double, (N == -1) ? N : N + 1, 1>& x, const Eigen::Matrix<double, (N == -1) ? N : N + 1, 1>& y) const
            {
                double p = x.transpose() * y;

                return -1 / sqrt(1 - p * p) * x;
            }

            // Distance hessian in the Euclidean space (same as for the gradient)
            Eigen::Matrix<double, (N == -1) ? N : N + 1, (N == -1) ? N : N + 1> distEEHess(const Eigen::Matrix<double, (N == -1) ? N : N + 1, 1>& x, const Eigen::Matrix<double, (N == -1) ? N : N + 1, 1>& y) const
            {
                double p = x.transpose() * y, c = -1 / sqrt(1 - p * p);

                return pow(c, 3) * p * x * x.transpose();
            }

        protected:
            // Sphere radius
            double _r;

            // Sphere center (embedding space)
            Eigen::Matrix<double, N + 1, 1> _c;
        };

        /* S2 sphere specialization */
        template <>
        Eigen::Vector3d SphereN<2>::embedding(const Eigen::Vector2d& x) const
        {
            // std::cout << "Special template" << std::endl;
            return _c + _r * Eigen::Vector3d(cos(x(0)), sin(x(0)) * cos(x(1)), sin(x(0)) * sin(x(1)));
        }

        template <>
        Eigen::Matrix<double, 3, 2> SphereN<2>::jacobian(const Eigen::Vector2d& x) const
        {
            // std::cout << "Special template" << std::endl;
            Eigen::Matrix<double, 3, 2> grad;

            grad << -sin(x(0)), 0,
                cos(x(0)) * cos(x(1)), -sin(x(0)) * sin(x(1)),
                cos(x(0)) * sin(x(1)), sin(x(0)) * cos(x(1));

            return _r * grad;
        }

        template <>
        Eigen::TensorFixedSize<double, Eigen::Sizes<3, 2, 2>> SphereN<2>::hessian(const Eigen::Vector2d& x) const
        {
            // std::cout << "Special template" << std::endl;
            Eigen::TensorFixedSize<double, Eigen::Sizes<3, 2, 2>> hess;

            // hess.setValues({{{-cos(x(0)), 0},
            //                     {0, 0},
            //                     {-sin(x(0)) * cos(x(1)), -cos(x(0)) * sin(x(1))}},
            //     {{-cos(x(0)) * sin(x(1)), -sin(x(0)) * cos(x(1))},
            //         {-sin(x(0)) * sin(x(1)), cos(x(0)) * cos(x(1))},
            //         {-cos(x(0)) * cos(x(1)), -sin(x(0)) * sin(x(1))}}});

            hess.setValues({{{-cos(x(0)), 0}, {0, 0}},
                {{-sin(x(0)) * cos(x(1)), -cos(x(0)) * sin(x(1))}, {-cos(x(0)) * sin(x(1)), -sin(x(0)) * cos(x(1))}},
                {{-sin(x(0)) * sin(x(1)), cos(x(0)) * cos(x(1))}, {cos(x(0)) * cos(x(1)), -sin(x(0)) * sin(x(1))}}});

            return _r * hess;
        }

        template <>
        Eigen::Matrix<double, 2, 2> SphereN<2>::metric(const Eigen::Vector2d& x) const
        {
            // std::cout << "Special template" << std::endl;
            Eigen::Matrix<double, 2, 2> g;

            g << 1, 0,
                0, sin(x(0)) * sin(x(0));

            return _r * _r * g;
        }

        template <>
        Eigen::TensorFixedSize<double, Eigen::Sizes<2, 2, 2>> SphereN<2>::metricGrad(const Eigen::Vector2d& x) const
        {
            // std::cout << "Special template" << std::endl;
            Eigen::TensorFixedSize<double, Eigen::Sizes<2, 2, 2>> dg;

            dg.setValues({{{0, 0},
                              {0, 0}},
                {{0, 0},
                    {2 * sin(x(0)) * cos(x(0)), 0}}});

            return _r * _r * dg;
        }

        template <>
        Eigen::TensorFixedSize<double, Eigen::Sizes<2, 2, 2>> SphereN<2>::christoffel(const Eigen::Vector2d& x) const
        {
            // std::cout << "Special template" << std::endl;
            Eigen::TensorFixedSize<double, Eigen::Sizes<2, 2, 2>> c;

            c(1, 1, 0) = cos(x(0)) / sinf(x(0));
            c(1, 0, 1) = c(1, 1, 0);
            c(0, 1, 1) = -sin(x(0)) * cos(x(0));

            return c;
        }
    } // namespace manifolds
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_MANIFOLDS_SPHEREN_HPP