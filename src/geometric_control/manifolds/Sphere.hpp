#ifndef GEOMETRIC_CONTROL_MANIFOLDS_SPHERE_HPP
#define GEOMETRIC_CONTROL_MANIFOLDS_SPHERE_HPP

#include "geometric_control/manifolds/AbstractManifold.hpp"
#include <complex>

namespace geometric_control {
    namespace manifolds {
        enum class SphereChart : unsigned int {
            POLAR = 1 << 0,
            STEREOGRAPHIC = 1 << 1
        };

        template <int N = Eigen::Dynamic>
        class Sphere : public AbstractManifold<N, 1> {
        public:
            Sphere() : _r(1.0)
            {
                if constexpr (N != Eigen::Dynamic)
                    _c.setZero(N + 1);
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

            // Get radius
            const double& radius() { return _r; }

            // Get center
            const Eigen::Matrix<double, N, 1>& center() { return _c; }

            // Set radius
            Sphere& setRadius(const double& radius)
            {
                _r = radius;
                return *this;
            }

            // Set center
            Sphere& setCenter(const Eigen::Matrix<double, eDim(), 1>& center)
            {
                _c = center;
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
                return u - ((x - _c).normalized().transpose() * u) * (x - _c).normalized();
            }

            // Retraction
            Eigen::Matrix<double, eDim(), 1> retract(const Eigen::Matrix<double, eDim(), 1>& x, const Eigen::Matrix<double, eDim(), 1>& u, const double& t = 1) const
            {
                auto p = (x - _c).normalized();
                return _r * (p + t * u) / (p + t * u).norm() + _c;
            }

            // Distance in the Euclidean embedding
            double dist(const Eigen::Matrix<double, eDim(), 1>& x, const Eigen::Matrix<double, eDim(), 1>& y) const
            {
                std::complex<double> d = ((x - _c).normalized() - (y - _c).normalized()).norm();

                return 2 * asin(0.5 * d).real() * _r;
            }

            // Distance gradient in the Euclidean embedding (here differential components and the gradient coincides due to the linearity of the space)
            Eigen::Matrix<double, eDim(), 1> distGrad(const Eigen::Matrix<double, eDim(), 1>& x, const Eigen::Matrix<double, eDim(), 1>& y) const
            {
                double p = (x - _c).normalized().transpose() * (y - _c).normalized();
                return -_r / sqrt(1 - p * p) * (x - _c).normalized();

                // double d = (x.normalized() - y.normalized()).norm();
                // return (y - x) / (2 * d * (1 - std::pow(d, 2))) * _r;
            }

            // Distance hessian in the Euclidean space (same as for the gradient)
            Eigen::Matrix<double, eDim(), eDim()> distHess(const Eigen::Matrix<double, eDim(), 1>& x, const Eigen::Matrix<double, eDim(), 1>& y) const
            {
                double p = (x - _c).normalized().transpose() * (y - _c).normalized(), c = -1 / sqrt(1 - p * p);

                return pow(c, 3) * p * (x - _c).normalized() * (x - _c).normalized().transpose() * _r;
            }

            Eigen::MatrixXd riemannGrad(const Eigen::Matrix<double, eDim(), 1>& x, const Eigen::MatrixXd& grad) const
            {
                Eigen::MatrixXd rGrad(grad.rows(), grad.cols());

                for (size_t i = 0; i < grad.rows(); i++)
                    rGrad.row(i) = project(x, grad.row(i));

                return rGrad;
            }

            Eigen::MatrixXd riemannHess(const Eigen::Matrix<double, eDim(), 1>& x, const Eigen::Matrix<double, eDim(), 1>& v, const Eigen::MatrixXd& grad, const Eigen::MatrixXd& hess) const
            {
                Eigen::MatrixXd rHess(hess.rows(), hess.cols());

                for (size_t i = 0; i < hess.rows(); i++)
                    rHess.row(i) = project(x, hess.row(i)) - (grad.row(i) * x) * v;

                return rHess;
            }

            /*
            |
            |   CHART DEPENDENT GEOMETRY (char geometry might be entirely implemented through visitor pattern)
            |
            */

            // Embedding
            Eigen::Matrix<double, eDim(), 1> embedding(const Eigen::Matrix<double, dim(), 1>& x) const
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
            Eigen::Matrix<double, eDim(), dim()> embeddingJacobian(const Eigen::Matrix<double, dim(), 1>& x) const
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
            Eigen::TensorFixedSize<double, Eigen::Sizes<eDim(), dim(), dim()>> embeddingHessian(const Eigen::Matrix<double, N, 1>& x) const
            {
                // std::cout << "General template" << std::endl;
                Eigen::TensorFixedSize<double, Eigen::Sizes<N + 1, N, N>> hess;

                hess.setZero();

                return hess;
            }

            // Metric
            Eigen::Matrix<double, dim(), dim()> pullMetric(const Eigen::Matrix<double, N, 1>& x) const
            {
                // std::cout << "General template" << std::endl;
                Eigen::Matrix<double, N, N> g;

                g(0, 0) = 1;

                for (size_t i = 1; i < N; i++)
                    g(i, i) = x.segment(0, i).array().sin().square().prod();

                return g;
            }

            // Metric Gradient
            Eigen::TensorFixedSize<double, Eigen::Sizes<dim(), dim(), dim()>> pullMetricGrad(const Eigen::Matrix<double, N, 1>& x) const
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
            Eigen::TensorFixedSize<double, Eigen::Sizes<dim(), dim(), dim()>> leviCivitaChristoffel(const Eigen::Matrix<double, N, 1>& x) const
            {
                // std::cout << "General template" << std::endl;
                return tools::leviCivitaConnection(tools::TensorCast(pullMetric(x).inverse()), pullMetricGrad(x));
            }

            // Distance
            double distChart(const Eigen::Matrix<double, dim(), 1>& x, const Eigen::Matrix<double, dim(), 1>& y)
            {
                std::complex<double> d = (embedding(x) - embedding(y)).norm();

                return 2 * asin(0.5 * d).real();
            }

            // Distance gradient (this actually the components of pullback of the differential of the distance function with
            // respect to the second entry). Sharping the operator we should get the real gradient of the distance function
            Eigen::Matrix<double, N, 1> distChartGrad(const Eigen::Matrix<double, N, 1>& x, const Eigen::Matrix<double, N, 1>& y)
            {
                double p = embedding(x).transpose() * embedding(y);

                return -1 / sqrt(1 - p * p) * embedding(x).transpose() * jacobian(y);
            }

            // Distance hessian (the same as above)
            Eigen::Matrix<double, N, N> distChartHess(const Eigen::Matrix<double, N, 1>& x, const Eigen::Matrix<double, N, 1>& y)
            {
                double p = embedding(x).transpose() * embedding(y), c = -1 / sqrt(1 - p * p);

                // row-row contraction
                Eigen::array<Eigen::IndexPair<int>, 1> contraction = {Eigen::IndexPair<int>(0, 0)};

                return pow(c, 3) * p * jacobian(y).transpose() * embedding(x) * embedding(x).transpose() * jacobian(y)
                    + c * tools::MatrixCast(hessian(y).contract(tools::TensorCast(embedding(x)), contraction), N, N);
            }

        protected:
            // Sphere radius
            double _r;

            // Sphere center (embedding space)
            Eigen::Matrix<double, eDim(), 1> _c;
        };

        /*
        |
        |   TEMPLATE SPECIALIZATION
        |
        */

        /* S2 sphere specialization */
        template <>
        Eigen::Vector3d Sphere<2>::embedding(const Eigen::Vector2d& x) const
        {
            // std::cout << "Special template" << std::endl;
            return _c + _r * Eigen::Vector3d(cos(x(0)), sin(x(0)) * cos(x(1)), sin(x(0)) * sin(x(1)));
        }

        template <>
        Eigen::Matrix<double, 3, 2> Sphere<2>::embeddingJacobian(const Eigen::Vector2d& x) const
        {
            // std::cout << "Special template" << std::endl;
            Eigen::Matrix<double, 3, 2> grad;

            grad << -sin(x(0)), 0,
                cos(x(0)) * cos(x(1)), -sin(x(0)) * sin(x(1)),
                cos(x(0)) * sin(x(1)), sin(x(0)) * cos(x(1));

            return _r * grad;
        }

        template <>
        Eigen::TensorFixedSize<double, Eigen::Sizes<3, 2, 2>> Sphere<2>::embeddingHessian(const Eigen::Vector2d& x) const
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
        Eigen::Matrix<double, 2, 2> Sphere<2>::pullMetric(const Eigen::Vector2d& x) const
        {
            // std::cout << "Special template" << std::endl;
            Eigen::Matrix<double, 2, 2> g;

            g << 1, 0,
                0, sin(x(0)) * sin(x(0));

            return _r * _r * g;
        }

        template <>
        Eigen::TensorFixedSize<double, Eigen::Sizes<2, 2, 2>> Sphere<2>::pullMetricGrad(const Eigen::Vector2d& x) const
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
        Eigen::TensorFixedSize<double, Eigen::Sizes<2, 2, 2>> Sphere<2>::leviCivitaChristoffel(const Eigen::Vector2d& x) const
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

#endif // GEOMETRIC_CONTROL_MANIFOLDS_SPHERE_HPP