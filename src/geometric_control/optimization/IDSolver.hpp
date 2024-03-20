#ifndef GEOMETRICCONTROL_OPTIMIZATION_IDSOLVER_HPP
#define GEOMETRICCONTROL_OPTIMIZATION_IDSOLVER_HPP

#include <geometric_control/optimization/QPSolver.hpp>
#include <iostream>

namespace geometric_control {
    namespace optimization {
        class IDSolver {
        public:
            // Constructor
            IDSolver(unsigned int nb_joints, unsigned int task_dim, bool use_slack = false);

            // Destructor
            ~IDSolver();

            void setJointPositionLimits(const Eigen::VectorXd& q_min,
                const Eigen::VectorXd& q_max);

            void setJointVelocityLimits(const Eigen::VectorXd& dq_max);

            void setJointVelocityLimits(const Eigen::VectorXd& dq_min,
                const Eigen::VectorXd& dq_max);

            void setJointAccelerationLimits(const Eigen::VectorXd& ddq_max);

            void setJointAccelerationLimits(const Eigen::VectorXd& ddq_min,
                const Eigen::VectorXd& ddq_max);

            void setJointTorqueLimits(const Eigen::VectorXd& tau_max);

            void setJointTorqueLimits(const Eigen::VectorXd& tau_min,
                const Eigen::VectorXd& tau_max);

            // Step function
            bool step(Eigen::VectorXd& tau, const Eigen::VectorXd& q,
                const Eigen::VectorXd& dq,
                const Eigen::VectorXd& error,
                const Eigen::MatrixXd& J,
                const Eigen::MatrixXd& dJ,
                const Eigen::MatrixXd& M,
                const Eigen::VectorXd& Cg,
                double dt);

        private:
            unsigned int nb_joints_;
            unsigned int task_dim_;
            unsigned int nb_slacks_;
            QPSolver qp_solver_;
            double dt_;

            Eigen::VectorXd q_min_;
            Eigen::VectorXd q_max_;
            Eigen::VectorXd dq_min_;
            Eigen::VectorXd dq_max_;
            Eigen::VectorXd ddq_min_;
            Eigen::VectorXd ddq_max_;
            Eigen::VectorXd tau_min_;
            Eigen::VectorXd tau_max_;
            Eigen::VectorXd slack_min_;
            Eigen::VectorXd slack_max_;
            Eigen::VectorXd error_int_;
            Eigen::MatrixXd J_prev_;
        };
    } // namespace optimization
} // namespace geometric_control

#endif // GEOMETRICCONTROL_OPTIMIZATION_IDSOLVER_HPP