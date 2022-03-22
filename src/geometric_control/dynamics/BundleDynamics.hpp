#ifndef GEOMETRIC_CONTROL_DYNAMICS_BUNDLE_DYNAMICS_HPP
#define GEOMETRIC_CONTROL_DYNAMICS_BUNDLE_DYNAMICS_HPP

#include <Eigen/Dense>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>

#include "geometric_control/tasks/AbstractTask.hpp"
#include "geometric_control/tools/helper.hpp"
#include "geometric_control/tools/math.hpp"

namespace geometric_control {
    // Visitable
    template <typename T1, typename... Types>
    class Visitable : public Visitable<T1>, public Visitable<Types...> {
    public:
        using Visitable<T1>::mapFrom;
        using Visitable<Types...>::mapFrom;

        using Visitable<T1>::jacobianFrom;
        using Visitable<Types...>::jacobianFrom;

        using Visitable<T1>::hessianFrom;
        using Visitable<Types...>::hessianFrom;
    };

    template <typename T1>
    class Visitable<T1> {
    public:
        virtual Eigen::VectorXd mapFrom(const Eigen::VectorXd& x, T1&) = 0; // accept

        virtual Eigen::MatrixXd jacobianFrom(const Eigen::VectorXd& x, T1&) = 0;

        virtual Eigen::Tensor<double, 3> hessianFrom(const Eigen::VectorXd& x, T1&) = 0;
    };

    // Visitor
    template <typename T1, typename... Types>
    class Visitor : public Visitor<T1>, public Visitor<Types...> {
    public:
        using Visitor<T1>::map;
        using Visitor<Types...>::map;

        using Visitor<T1>::jacobian;
        using Visitor<Types...>::jacobian;

        using Visitor<T1>::hessian;
        using Visitor<Types...>::hessian;
    };

    template <typename T1>
    class Visitor<T1> {
    public:
        virtual Eigen::VectorXd map(const Eigen::VectorXd& x, T1&) = 0; // visit

        virtual Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, T1&) = 0;

        virtual Eigen::Tensor<double, 3> hessian(const Eigen::VectorXd& x, T1&) = 0;
    };

    // Tree of manifolds
    template <typename... Args>
    class TreeManifolds : public Visitor<Args...> {
    };

    namespace dynamics {
        // Bundle Dynamics interface
        // Each Bundle Dynamics is an entity define by a manifold with some task attached to it and
        // potentially attached to other Bundle Dynamics
        template <typename TreeManifoldsImpl>
        class BundleDynamicsInterface : public Visitable<TreeManifoldsImpl> {
        public:
        };

        template <typename Manifold, typename TreeManifoldsImpl, template <typename> typename Mapping>
        class BundleDynamics : public BundleDynamicsInterface<TreeManifoldsImpl> {
        public:
            BundleDynamics() : _manifold(), _m(1.0) {}

            // Dynamical System
            Eigen::VectorXd operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& v)
            {
                Eigen::MatrixXd A = Eigen::MatrixXd::Zero(x.rows(), x.rows());
                Eigen::VectorXd b = Eigen::VectorXd::Zero(x.rows());

                Eigen::array<Eigen::IndexPair<int>, 1> c1 = {Eigen::IndexPair<int>(1, 0)},
                                                       c2 = {Eigen::IndexPair<int>(2, 0)};

                for (auto& task : _tasks) {
                    A += task->jacobian(x).transpose() * task->weight(x, v) * task->jacobian(x);
                    b -= task->jacobian(x).transpose() * task->weight(x, v)
                        * (tools::VectorCast(task->hessian(x).contract(tools::TensorCast(v), c2).contract(tools::TensorCast(v), c1))
                            + tools::VectorCast(task->christoffel(x).contract(tools::TensorCast(task->jacobian(x) * v), c2).contract(tools::TensorCast(task->jacobian(x) * v), c1))
                            + task->metric(x).inverse() * (task->energyGrad(x) + task->field(x, v)) / _m);
                }

                return A.selfadjointView<Eigen::Upper>().llt().solve(b);
            }

            // Get Bundle Dynamics
            BundleDynamicsInterface<TreeManifoldsImpl>& bundle(const size_t& i) { return *_bundles[i]; }

            // Get Task
            tasks::AbstractTask<Manifold>& task(const size_t& i) { return *_tasks[i].get(); }

            // Add Bundle Dynamics
            template <typename... Args>
            BundleDynamics& addBundles(BundleDynamicsInterface<TreeManifoldsImpl>* bundle, Args... args)
            {
                _bundles.push_back(std::move(bundle));

                if constexpr (sizeof...(args) > 0)
                    addBundles(std::move(args)...);

                return *this;
            }

            // Add Tasks
            template <typename... Args>
            BundleDynamics& addTasks(std::unique_ptr<tasks::AbstractTask<Manifold>> task, Args... args)
            {
                _tasks.push_back(std::move(task));

                if constexpr (sizeof...(args) > 0)
                    addTasks(std::move(args)...);

                return *this;
            }

            // Recursive map computation
            Eigen::VectorXd mapFrom(const Eigen::VectorXd& x, TreeManifoldsImpl& node) override
            {
                _f.setZero(x.size());

                for (auto& bundle : _bundles)
                    _f += bundle->mapFrom(x, _mapping);

                for (auto& task : _tasks)
                    _f += task->map(x);

                return _f + node.map(x, _manifold);
            }

            // Recursive jacobian computation
            Eigen::MatrixXd jacobianFrom(const Eigen::VectorXd& x, TreeManifoldsImpl& node) override
            {
                _j.setZero(x.size(), x.size());
                return _j;
            }

            // Recursive hessian computation
            Eigen::Tensor<double, 3> hessianFrom(const Eigen::VectorXd& x, TreeManifoldsImpl& node) override
            {
                _h = Eigen::Tensor<double, 3>(x.size(), x.size(), x.size());
                _h.setZero();
                return _h;
            }

        protected:
            // Mass
            double _m;

            // Manifold
            Manifold _manifold;

            // Map, Jacobian, Hessian
            Eigen::VectorXd _f;
            Eigen::MatrixXd _j;
            Eigen::Tensor<double, 3> _h;

            // Tree mapping
            Mapping<Manifold> _mapping;

            // Bundles
            std::vector<BundleDynamicsInterface<TreeManifoldsImpl>*> _bundles;

            // Tasks
            std::vector<std::unique_ptr<tasks::AbstractTask<Manifold>>> _tasks;
        };
    } // namespace dynamics
} // namespace geometric_control

#endif // GEOMETRIC_CONTROL_DYNAMICS_BUNDLE_DYNAMICS_HPP