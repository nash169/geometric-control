#ifndef GEOMETRIC_CONTROL_MANIFOLDS_SPHERE_HPP
#define GEOMETRIC_CONTROL_MANIFOLDS_SPHERE_HPP

#include <unsupported/Eigen/CXX11/Tensor>

#include "geometric_control/manifolds/AbstractManifold.hpp"
#include "geometric_control/tools/helper.hpp"

namespace geometric_control {
    namespace manifolds {
        class Sphere : public AbstractManifold {
        public:
            Sphere(const size_t& dimension) : AbstractManifold(dimension), _radius(1.0)
            {
                _center.setZero(dimension + 1);
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
                Eigen::MatrixXd g = metric(x);
                g.diagonal() = g.diagonal().array().inverse();

                Eigen::array<Eigen::IndexPair<int>, 1> c1 = {Eigen::IndexPair<int>(1, 0)};

                Eigen::Tensor<double, 3> grad = metricGrad(x),
                                         T = 0.5 * tools::TensorCast(g).contract(grad.shuffle(Eigen::array<int, 3>({1, 0, 2})) + grad.shuffle(Eigen::array<int, 3>({1, 0, 2})).shuffle(Eigen::array<int, 3>({0, 2, 1})) - grad, c1);

                return T;
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

                return grad.shuffle(Eigen::array<int, 3>({2, 0, 1}));
            }

        protected:
            double _radius;
            Eigen::VectorXd _center;
        };
    } // namespace manifolds
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_MANIFOLDS_SPHERE_HPP