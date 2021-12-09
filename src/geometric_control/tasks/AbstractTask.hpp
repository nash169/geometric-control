#ifndef GEOMETRIC_CONTROL_TASKS_ABSTRACT_TASK_HPP
#define GEOMETRIC_CONTROL_TASKS_ABSTRACT_TASK_HPP

#include "geometric_control/tools/helper.hpp"

namespace geometric_control {
    namespace tasks {
        template <typename Manifold>
        class AbstractTask {
        public:
            AbstractTask() : _M() {}

            virtual ~AbstractTask() {}

            // Optimization weight
            virtual Eigen::MatrixXd weight(const Eigen::VectorXd&, const Eigen::VectorXd&) const = 0;

            // Map between configuration and task manifolds
            virtual Eigen::VectorXd map(const Eigen::VectorXd&) const = 0;

            // Jacobian
            virtual Eigen::MatrixXd jacobian(const Eigen::VectorXd&) const = 0;

            // Hessian
            virtual Eigen::Tensor<double, 3> hessian(const Eigen::VectorXd&) const = 0;

            // Task manifold metric
            virtual Eigen::MatrixXd metric(const Eigen::VectorXd&) const = 0;

            // Connection functions
            virtual Eigen::Tensor<double, 3> christoffel(const Eigen::VectorXd&) const = 0;

            // Energy scalar field -> (0,0) tensor field
            virtual double energy(const Eigen::VectorXd&) const = 0;

            // Energy gradient field -> (1,0) tensor field when sharped
            virtual Eigen::VectorXd energyGrad(const Eigen::VectorXd&) const = 0;

            // (Covector) field -> (0,1) tensor field
            virtual Eigen::VectorXd field(const Eigen::VectorXd&, const Eigen::VectorXd&) const = 0;

        protected:
            // Base manifold
            Manifold _M;
        };
    } // namespace tasks
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_TASKS_ABSTRACT_TASK_HPP