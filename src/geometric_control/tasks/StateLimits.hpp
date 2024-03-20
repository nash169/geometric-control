#ifndef GEOMETRICCONTROL_TASKS_STATELIMITS_HPP
#define GEOMETRICCONTROL_TASKS_STATELIMITS_HPP

#include "geometric_control/tasks/AbstractTask.hpp"

namespace geometric_control {
    namespace tasks {
        template <typename Manifold>
        class StateLimits : public AbstractTask<Manifold> {
        public:
            StateLimits() : AbstractTask<Manifold>()
            {
                // Init state index
                _index = 0;

                // Init limits
                _low = -1;
                _up = 1;

                // Init metric params
                _a = 1e-6;
                _b = 1e6;
            }

            // Set state index
            StateLimits& setIndex(const size_t& index)
            {
                _index = index;
                return *this;
            }

            // Set lower and upper limits
            StateLimits& setLimits(const double& low, const double& up)
            {
                _low = low;
                _up = up;
                return *this;
            }

            // Set metric parameters
            StateLimits& setMetricParams(const double& a, const double& b)
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
                return ((jacobian(x) * v)[0] < 0) ? tools::makeMatrix(1) : tools::makeMatrix(0);
            }

            // Map between configuration and task manifolds
            Eigen::VectorXd map(const Eigen::VectorXd& x) const override
            {
                return tools::makeVector(x[_index]);
            }

            // Jacobian
            Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const override
            {
                Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, x.size());
                J(0, _index) = 1;
                return J;
            }

            // Hessian
            Eigen::Tensor<double, 3> hessian(const Eigen::VectorXd& x) const override
            {
                Eigen::Tensor<double, 3> H(1, x.size(), x.size());
                H.setZero();
                return H;
            }

            Eigen::MatrixXd hessianDir(const Eigen::VectorXd& x, const Eigen::VectorXd& v) const override
            {
                return Eigen::MatrixXd::Zero(1, x.size());
            }

            // Task manifold metric
            Eigen::MatrixXd metric(const Eigen::VectorXd& x) const override
            {
                double y = map(x)[0];
                return tools::makeMatrix(std::exp(_a / (std::pow(y + _low, _b) * std::pow(y - _up, _b))));
            }

            // Connection functions
            Eigen::Tensor<double, 3> christoffel(const Eigen::VectorXd& x) const override
            {
                double y = map(x)[0];

                Eigen::Tensor<double, 3> gamma(1, 1, 1);
                gamma(0, 0, 0) = -0.5 * _a * std::pow(y, -_b - 1);

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

            // State index
            size_t _index;

            // Upper/Low Limits and Metric Parameters
            double _up, _low, _a, _b;
        };
    } // namespace tasks
} // namespace geometric_control

#endif // GEOMETRICCONTROL_TASKS_STATELIMITS_HPP