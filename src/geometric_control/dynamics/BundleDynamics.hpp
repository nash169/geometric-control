#ifndef GEOMETRIC_CONTROL_DYNAMICS_BUNDLE_DYNAMICS_HPP
#define GEOMETRIC_CONTROL_DYNAMICS_BUNDLE_DYNAMICS_HPP

#include <Eigen/Dense>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>

#include "geometric_control/tasks/AbstractTask.hpp"
#include "geometric_control/tools/helper.hpp"
#include "geometric_control/tools/math.hpp"

namespace geometric_control {
    namespace dynamics {
        template <typename Manifold>
        class BundleDynamics {
        public:
            BundleDynamics() : _M(), _m(1.0) {}

            Manifold& manifold() { return _M; }

            Eigen::VectorXd operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& v)
            {
                Eigen::MatrixXd A = Eigen::MatrixXd::Zero(x.rows(), x.rows());
                Eigen::VectorXd b = Eigen::VectorXd::Zero(x.rows());

                Eigen::array<Eigen::IndexPair<int>, 1> c1 = {Eigen::IndexPair<int>(1, 0)},
                                                       c2 = {Eigen::IndexPair<int>(2, 0)};

                for (auto& t : _t) {
                    A += t->jacobian(x).transpose() * t->weight(x, v) * t->jacobian(x);
                    b -= t->jacobian(x).transpose() * t->weight(x, v)
                        * (tools::VectorCast(t->hessian(x).contract(tools::TensorCast(v), c2).contract(tools::TensorCast(v), c1))
                            + tools::VectorCast(t->christoffel(x).contract(tools::TensorCast(t->jacobian(x) * v), c2).contract(tools::TensorCast(t->jacobian(x) * v), c1))
                            + t->metric(x).inverse() * (t->energyGrad(x) + t->field(x, v)) / _m);
                }

                return A.selfadjointView<Eigen::Upper>().llt().solve(b);
            }

            tasks::AbstractTask<Manifold>& task(const size_t& i) { return *_t[i].get(); }

            template <typename... Args>
            BundleDynamics& addTasks(std::unique_ptr<tasks::AbstractTask<Manifold>> task, Args... args)
            {
                _t.push_back(std::move(task));

                if constexpr (sizeof...(args) > 0)
                    addTasks(std::move(args)...);

                return *this;
            }

        protected:
            // Manifold
            Manifold _M;

            // Mass
            double _m;

            // Tasks
            std::vector<std::unique_ptr<tasks::AbstractTask<Manifold>>> _t;
        };
    } // namespace dynamics
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_DYNAMICS_BUNDLE_DYNAMICS_HPP