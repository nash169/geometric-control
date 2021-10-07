#ifndef GEOMETRIC_CONTROL_MANIFOLDS_SPHERE_HPP
#define GEOMETRIC_CONTROL_MANIFOLDS_SPHERE_HPP

#include <complex>

#include "geometric_control/manifolds/AbstractManifold.hpp"

namespace geometric_control {
    namespace manifolds {
        class Sphere : public AbstractManifold {
        public:
            Sphere(const size_t& dim) : AbstractManifold(dim), _radius(1.0)
            {
                _center.setZero(dim + 1);
            }

            ~Sphere() {}

            const double& radius() { return _radius; }

            const Eigen::VectorXd& center() { return _center; }

            Sphere& setRadius(const double& radius)
            {
                _radius = radius;
                return *this;
            }

            Sphere& setCenter(const Eigen::VectorXd& center)
            {
                _center = center;
                return *this;
            }

            /* Euclidean Embedding */
            Eigen::VectorXd embedding(const Eigen::VectorXd& x) override
            {
                // Dimension
                size_t d = x.rows();

                Eigen::VectorXd y = Eigen::VectorXd::Ones(d + 1);

                for (size_t i = 0; i < d + 1; i++)
                    if (i != d)
                        for (size_t j = 0; j < i + 1; j++)
                            y(i) *= (j == i) ? cos(x(j)) : sin(x(j));
                    else
                        for (size_t j = 0; j < i; j++)
                            y(i) *= sin(x(j));

                return y;
            }

            Eigen::MatrixXd jacobian(const Eigen::VectorXd& x)
            {
                // Dimension
                size_t d = x.rows();

                Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(d + 1, d);

                // Terrible way of calculating this jacobian...
                for (size_t i = 0; i < d + 1; i++)
                    if (i != d)
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

            Eigen::Tensor<double, 3> hessian(const Eigen::VectorXd& x)
            {
                Eigen::Tensor<double, 3> hess;

                return hess;
            }

            Eigen::MatrixXd metric(const Eigen::VectorXd& x) override
            {
                // Dimension
                size_t d = x.rows();

                Eigen::MatrixXd g(d, d);

                g(0, 0) = 1;

                for (size_t i = 1; i < d; i++)
                    g(i, i) = x.segment(0, i).array().sin().square().prod();

                return g;
            }

            Eigen::Tensor<double, 3> christoffel(const Eigen::VectorXd& x) override
            {
                return tools::leviCivitaConnection(tools::TensorCast(metric(x).inverse()), metricGrad(x));
            }

            Eigen::Tensor<double, 3> metricGrad(const Eigen::VectorXd& x)
            {
                // Dimension
                u_int d = x.rows();

                Eigen::Tensor<double, 3> grad(d, d, d);
                grad.setZero();

                for (u_int i = 1; i < d; i++)
                    for (u_int j = 0; j < i; j++) {
                        grad(i, i, j) = 1;

                        for (u_int k = 0; k < i; k++)
                            grad(i, i, j) *= (k == j) ? 2 * sin(x(k)) * cos(x(k)) : sin(x(k)) * sin(x(k));
                    }

                return grad;
            }

            double distance(const Eigen::VectorXd& x, const Eigen::VectorXd& y)
            {
                std::complex<double> d = (embedding(x) - embedding(y)).norm();

                return 2 * asin(0.5 * d).real();
            }

            Eigen::VectorXd distanceGrad(const Eigen::VectorXd& x, const Eigen::VectorXd& y)
            {
                double p = embedding(x).transpose() * embedding(y);

                return -1 / sqrt(1 - p * p) * embedding(x).transpose() * jacobian(y);

                // double d = (embedding(x) - embedding(y)).norm();

                // return -cos(d) / pow(sin(d), 2) / d * (embedding(x) - embedding(y)).transpose() * jacobian(x);
            }

            Eigen::VectorXd projector(const Eigen::VectorXd& x, const Eigen::VectorXd& u)
            {
                return u - (x.transpose() * u) * x;
            }

        protected:
            double _radius;
            Eigen::VectorXd _center;

            // // with respect to y
            // Eigen::VectorXd distanceGrad(const Eigen::VectorXd& x, const Eigen::VectorXd& y)
            // {
            //     double p = embedding(x).transpose() * embedding(y);

            //     return projector(embedding(y), pow(acos(p), 2) * sin(p) * embedding(x));
            // }
        };
    } // namespace manifolds
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_MANIFOLDS_SPHERE_HPP