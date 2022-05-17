#ifndef GEOMETRIC_CONTROL_DYNAMICS_BUNDLE_DYNAMICS_HPP
#define GEOMETRIC_CONTROL_DYNAMICS_BUNDLE_DYNAMICS_HPP

#include <Eigen/Dense>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>

#include "geometric_control/tasks/AbstractTask.hpp"
#include "geometric_control/tools/helper.hpp"
#include "geometric_control/tools/math.hpp"

#include <utils_lib/Timer.hpp>

using namespace utils_lib;

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
        // accept
        virtual Eigen::VectorXd mapFrom(const Eigen::VectorXd& x, T1&) = 0;

        virtual Eigen::MatrixXd jacobianFrom(const Eigen::VectorXd& x, T1&) = 0;

        virtual Eigen::MatrixXd hessianFrom(const Eigen::VectorXd& x, const Eigen::VectorXd& v, T1&) = 0;
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
        // visit
        virtual Eigen::VectorXd map(const Eigen::VectorXd& x, T1&) = 0;

        virtual Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, T1&) = 0;

        virtual Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, T1&) = 0;
    };

    // Tree of manifolds
    template <typename... Args>
    class TreeManifolds : public Visitor<Args...> {
    };

    // Introduce parent manifol template at this level

    // template <typename ParnetManifold, typename... Args>
    // class TreeManifolds : public Visitor<Args...> {
    // public:
    //     TreeManifolds(ParentManifold& manifold) : _manifold(manifold) {}
    //     ParentManifold& _manifold;
    // }

    // The create the tree manifold implementation as

    // template<typename ParentManifold>
    // using TreeManifoldsImpl = TreeManifolds<ParentManifold, Manifold1, Manifold2, ...>;

    namespace dynamics {
        // Bundle Dynamics interface
        // Each Bundle Dynamics is an entity define by a manifold with some task attached to it and
        // potentially attached to other Bundle Dynamics
        template <typename TreeManifoldsImpl>
        class BundleDynamicsInterface : public Visitable<TreeManifoldsImpl> {
        public:
            virtual BundleDynamicsInterface& update(const Eigen::VectorXd& x, const Eigen::VectorXd& v) = 0;
            virtual BundleDynamicsInterface& solve() = 0;

            // Pseudo, hessian and christofell matrices
            Eigen::MatrixXd _A, _H, _G;
            // Forces vector
            Eigen::VectorXd _F;
            // State
            Eigen::VectorXd _x, _dx, _ddx;
        };

        template <typename Manifold, typename TreeManifoldsImpl, template <typename> typename Mapping>
        class BundleDynamics : public BundleDynamicsInterface<TreeManifoldsImpl> {
        public:
            BundleDynamics() : _manifold(), _m(1.0)
            {
                _mapping._manifold = &_manifold;
            }

            // Dynamical System
            Eigen::VectorXd operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& v)
            {
                return update(x, v).solve()._ddx;
            }

            BundleDynamics& update(const Eigen::VectorXd& x, const Eigen::VectorXd& v) override
            {
                // store position and velocity
                _x = x;
                _dx = v;

                // init matrices
                _A.setZero(x.size(), x.size());
                _H.setZero(x.size(), x.size());
                _G.setZero(x.size(), x.size());
                _F.setZero(x.size());

                // Update bundles recursively
                for (auto& bundle : _bundles) {
                    // bundle jacobian and hessian
                    Eigen::MatrixXd J = bundle->jacobianFrom(x, _mapping),
                                    H = bundle->hessianFrom(x, v, _mapping);
                    // recursive call
                    bundle->update(bundle->mapFrom(x, _mapping), J * v);
                    // update pseudo matrix
                    _A += J.transpose() * bundle->_A * J;
                    // update hessian matrix
                    _H += J.transpose() * (bundle->_H * J + bundle->_A * H);
                    // update christoffel matrix
                    _G += J.transpose() * bundle->_G * J;
                    // update forces matrix
                    _F += J.transpose() * bundle->_F;
                }

                // Update tasks
                for (auto& task : _tasks) {
                    // task jacobian and weight
                    Eigen::MatrixXd J = task->jacobian(x),
                                    W = task->weight(x, v);
                    // Update matrices
                    _A += J.transpose() * W * J;
                    _H -= J.transpose() * W * task->hessianDir(x, v);
                    _G -= J.transpose() * W * task->christoffelDir(x, v) * J;
                    _F -= J.transpose() * W * task->metricInv(x) * (task->energyGrad(x) + task->field(x, v));
                }

                return *this;
            }

            BundleDynamics& solve() override
            {
                // store acceleration of the DS
                _ddx = Eigen::MatrixXd(_A).selfadjointView<Eigen::Upper>().llt().solve(_F / _m + (_H + _G) * _dx);

                return *this;
            }

            // Get acceleration
            const Eigen::VectorXd& acceleration()
            {
                return _ddx;
            }

            // Get manifold (for now not const reference but it should be)
            Manifold& manifold()
            {
                return _manifold;
            }

            // Get child Bundle Dynamics
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
                return node.map(x, _manifold);
            }

            // Recursive jacobian computation
            Eigen::MatrixXd jacobianFrom(const Eigen::VectorXd& x, TreeManifoldsImpl& node) override
            {
                return node.jacobian(x, _manifold);
            }

            // Recursive hessian computation
            Eigen::MatrixXd hessianFrom(const Eigen::VectorXd& x, const Eigen::VectorXd& v, TreeManifoldsImpl& node) override
            {
                return node.hessian(x, v, _manifold);
            }

        protected:
            // Lift variables
            using BundleDynamicsInterface<TreeManifoldsImpl>::_A;
            using BundleDynamicsInterface<TreeManifoldsImpl>::_H;
            using BundleDynamicsInterface<TreeManifoldsImpl>::_G;
            using BundleDynamicsInterface<TreeManifoldsImpl>::_F;

            using BundleDynamicsInterface<TreeManifoldsImpl>::_x;
            using BundleDynamicsInterface<TreeManifoldsImpl>::_dx;
            using BundleDynamicsInterface<TreeManifoldsImpl>::_ddx;

            // Mass
            double _m;

            // Manifold
            Manifold _manifold;

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

// Eigen::MatrixXd A = Eigen::MatrixXd::Zero(x.rows(), x.rows());
// Eigen::VectorXd b = Eigen::VectorXd::Zero(x.rows());

// Eigen::array<Eigen::IndexPair<int>, 1> c1 = {Eigen::IndexPair<int>(1, 0)},
//                                        c2 = {Eigen::IndexPair<int>(2, 0)};

// for (auto& task : _tasks) {
//     A += task->jacobian(x).transpose() * task->weight(x, v) * task->jacobian(x);
//     b -= task->jacobian(x).transpose() * task->weight(x, v)
//         * (tools::VectorCast(task->hessian(x).contract(tools::TensorCast(v), c2).contract(tools::TensorCast(v), c1))
//             + tools::VectorCast(task->christoffel(x).contract(tools::TensorCast(task->jacobian(x) * v), c2).contract(tools::TensorCast(task->jacobian(x) * v), c1))
//             + task->metric(x).inverse() * (task->energyGrad(x) + task->field(x, v)) / _m);
// }

// return A.selfadjointView<Eigen::Upper>().llt().solve(b);

// Eigen::MatrixXd A2 = Eigen::MatrixXd::Zero(x.rows(), x.rows());
// Eigen::VectorXd b2 = Eigen::VectorXd::Zero(x.rows());

// for (auto& task : _tasks) {
//     Eigen::MatrixXd J = task->jacobian(x), w = task->weight(x, v);
//     A2 += J.transpose() * w * J;
//     b2 -= J.transpose() * w
//         * (task->metric(x).inverse() * (task->energyGrad(x) + task->field(x, v)) / _m
//             + (task->hessianDir(x, v) + task->christoffelDir(x, v) * J) * v);
// }

// return A2.selfadjointView<Eigen::Upper>().llt().solve(b2);