#ifndef GEOMETRIC_CONTROL_TASKS_OBSTACLE_AVOIDANCE_HPP
#define GEOMETRIC_CONTROL_TASKS_OBSTACLE_AVOIDANCE_HPP

#include "geometric_control/tasks/AbstractTask.hpp"

namespace geometric_control {
    namespace tasks {
        template <typename Manifold>
        class ObstacleAvoidance : public AbstractTask<Manifold> {
        public:
            ObstacleAvoidance() : AbstractTask<Manifold>()
            {
                // Init obstacle center to radius distance
                _r = 1;

                // Init obstacle center (in the embedding space)
                _c = _M.embedding(Eigen::VectorXd::Zero(Manifold::dim()));

                // Init metric params
                _a = 1;
                _b = 2;
            }

            // Set radius
            ObstacleAvoidance& setRadius(const double& r)
            {
                _r = r;
                return *this;
            }

            // Set center
            // In order to avoid issues the center location is always passed in chart coordinate
            ObstacleAvoidance& setCenter(const Eigen::VectorXd& c)
            {
                _c = _M.embedding(c);
                return *this;
            }

            // Set metric parameters
            ObstacleAvoidance& setMetricParams(const double& a, const double& b)
            {
                _a = a;
                _b = b;
                return *this;
            }

            // Space dimension
            constexpr int dim() const override
            {
                return 1;
            }

            // Optimization weight
            Eigen::MatrixXd weight(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
            {
                return tools::makeMatrix(1);
            }

            // Map between configuration and task manifolds
            Eigen::VectorXd map(const Eigen::VectorXd& x) const override
            {
                return tools::makeVector(_M.dist(_c, x) - _r);
            }

            // Jacobian
            Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const override
            {
                return _M.riemannGrad(x, _M.distGrad(_c, x).transpose()); // return _M.distGrad(_c, x).transpose();
            }

            // Hessian
            Eigen::Tensor<double, 3> hessian(const Eigen::VectorXd& x) const override
            {
                return tools::TensorCast(_M.distHess(_c, x), 1, x.rows(), x.rows());
            }

            Eigen::MatrixXd hessianDir(const Eigen::VectorXd& x, const Eigen::VectorXd& v) override
            {
                return _M.riemannHess(x, v, _M.distGrad(_c, x).transpose(), (_M.distHess(_c, x) * v).transpose());
            }

            // Task manifold metric
            Eigen::MatrixXd metric(const Eigen::VectorXd& x) const override
            {
                double y = map(x)[0];
                return tools::makeMatrix(std::exp(_a / (_b * std::pow(y, _b))));
            }

            // Connection functions
            Eigen::Tensor<double, 3> christoffel(const Eigen::VectorXd& x) const override
            {
                double y = map(x)[0];

                Eigen::Tensor<double, 3> gamma(1, 1, 1);
                gamma(0, 0, 0) = -0.5 * _a / std::pow(y, _b + 1);

                return gamma;
            }

            // Energy scalar field -> (0,0) tensor field
            double energy(const Eigen::VectorXd& x) const override
            {
                return 0;
            }

            // Energy gradient field -> (1,0) tensor field when sharped
            Eigen::VectorXd energyGrad(const Eigen::VectorXd& x) const override
            {
                return tools::makeVector(0);
            }

            // (Co-vector) field -> (0,1) tensor field
            Eigen::VectorXd field(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
            {
                return tools::makeVector(0);
            }

        protected:
            // Manifold
            using AbstractTask<Manifold>::_M;

            // Obstacle radius and center position (for the moment 2D/3D sphere)
            double _r;
            Eigen::VectorXd _c; // this can be Eigen::Matrix<double, Manifold::dimension(), 1>

            // Metric parmeters
            double _a, _b;
        };
    } // namespace tasks
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_TASKS_OBSTACLE_AVOIDANCE_HPP