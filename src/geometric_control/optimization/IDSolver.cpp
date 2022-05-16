#include "geometric_control/optimization/IDSolver.hpp"

namespace geometric_control {
    namespace optimization {
        IdSolver::IdSolver(unsigned int nb_joints, unsigned int task_dim, bool use_slack) : nb_joints_(nb_joints), task_dim_(task_dim)
        {

            q_min_.resize(nb_joints_);
            q_max_.resize(nb_joints_);
            dq_min_.resize(nb_joints_);
            dq_max_.resize(nb_joints_);
            ddq_min_.resize(nb_joints_);
            ddq_max_.resize(nb_joints_);
            tau_min_.resize(nb_joints_);
            tau_max_.resize(nb_joints_);

            if (use_slack) {
                nb_slacks_ = task_dim_;
            }
            else {
                nb_slacks_ = 0;
            }

            error_int_.resize(task_dim_);
            error_int_.setZero();

            slack_min_.resize(nb_slacks_);
            slack_max_.resize(nb_slacks_);
            slack_min_.setConstant(-0.1);
            slack_max_.setConstant(0.1);
            // slack_min_.head(task_dim_).setConstant(-5.0);
            // slack_max_.head(task_dim_).setConstant(5.0);

            qp_solver_.init(2 * nb_joints_ + nb_slacks_, task_dim_ + 3 * nb_joints_);
            // qp_solver_.init(nb_joints_+nb_slacks_,task_dim_+3*nb_joints_);
            J_prev_.resize(task_dim_, nb_joints_);
            J_prev_.setZero();
        }

        IdSolver::~IdSolver()
        {
        }

        void IdSolver::setJointPositionLimits(const Eigen::VectorXd& q_min,
            const Eigen::VectorXd& q_max)
        {
            q_min_ = q_min;
            q_max_ = q_max;
        }

        void IdSolver::setJointVelocityLimits(const Eigen::VectorXd& dq_max)
        {
            setJointVelocityLimits(-dq_max, dq_max);
        }

        void IdSolver::setJointVelocityLimits(const Eigen::VectorXd& dq_min,
            const Eigen::VectorXd& dq_max)
        {
            dq_min_ = dq_min;
            dq_max_ = dq_max;
        }

        void IdSolver::setJointAccelerationLimits(const Eigen::VectorXd& ddq_max)
        {
            setJointAccelerationLimits(-ddq_max, ddq_max);
        }

        void IdSolver::setJointAccelerationLimits(const Eigen::VectorXd& ddq_min,
            const Eigen::VectorXd& ddq_max)
        {
            ddq_min_ = ddq_min;
            ddq_max_ = ddq_max;
        }

        void IdSolver::setJointTorqueLimits(const Eigen::VectorXd& tau_max)
        {
            setJointTorqueLimits(-tau_max, tau_max);
        }

        void IdSolver::setJointTorqueLimits(const Eigen::VectorXd& tau_min,
            const Eigen::VectorXd& tau_max)
        {
            tau_min_ = tau_min;
            tau_max_ = tau_max;
        }

        bool IdSolver::step(Eigen::VectorXd& tau, const Eigen::VectorXd& q,
            const Eigen::VectorXd& dq,
            const Eigen::VectorXd& error,
            const Eigen::MatrixXd& J,
            const Eigen::MatrixXd& dJ,
            const Eigen::MatrixXd& M,
            const Eigen::VectorXd& Cg,
            double dt)
        {
            assert(q.size() == nb_joints_);
            assert(dq.size() == nb_joints_);
            assert(error.size() == task_dim_);
            assert(J.rows() == task_dim_ && J.cols() == nb_joints_);
            assert(M.rows() == nb_joints_ && M.cols() == nb_joints_);
            assert(Cg.size() == nb_joints_);

            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2 * nb_joints_ + nb_slacks_,
                2 * nb_joints_ + nb_slacks_);

            // std::cout <<"1" << std::endl;
            H.block(0, 0, 2 * nb_joints_, 2 * nb_joints_) = Eigen::MatrixXd::Identity(2 * nb_joints_, 2 * nb_joints_);
            H.block(nb_joints_, nb_joints_, nb_joints_, nb_joints_) = 0.1 * Eigen::MatrixXd::Identity(nb_joints_, nb_joints_);
            H.block(2 * nb_joints_, 2 * nb_joints_, nb_slacks_, nb_slacks_) = 1e6 * Eigen::MatrixXd::Identity(nb_slacks_, nb_slacks_);

            // std::cout <<"2" << std::endl;

            // H.block(0,0,nb_joints_,nb_joints_) = 1000*J.transpose()*J+
            // 	0.00001*Eigen::MatrixXd::Identity(nb_joints_, nb_joints_);
            // H.block(nb_joints_,nb_joints_,nb_joints_,nb_joints_) =
            // 		0.001*Eigen::MatrixXd::Identity(nb_joints_, nb_joints_);
            // H.block(2*nb_joints_,2*nb_joints_,nb_slacks_,nb_slacks_) =
            // 		10000.0*Eigen::MatrixXd::Identity(nb_slacks_, nb_slacks_);

            // error_int_ += dt*error;
            Eigen::VectorXd g = Eigen::VectorXd::Zero(2 * nb_joints_ + nb_slacks_);
            // g.segment(0,nb_joints_) = -1000*J.transpose()*(5*error-(J-J_prev_)*dq/dt);
            // g.segment(0,nb_joints_) = -J.transpose()*(error-dJ*dq);

            g.segment(nb_joints_, nb_joints_) = -0.1 * Cg;

            // std::cout <<"3" << std::endl;

            // Eigen::MatrixXd H = Eigen::MatrixXd::Zero(nb_joints_+nb_slacks_,
            //                                           nb_joints_+nb_slacks_);

            // H.block(0,0,nb_joints_,nb_joints_) = Eigen::MatrixXd::Identity(nb_joints_, nb_joints_);
            // H.block(nb_joints_,nb_joints_,nb_slacks_,nb_slacks_) =
            //         1e6*Eigen::MatrixXd::Identity(nb_slacks_, nb_slacks_);
            // Eigen::VectorXd g = Eigen::VectorXd::Zero(nb_joints_+nb_slacks_);

            /////////////////
            // Constraints //
            /////////////////
            Eigen::MatrixXd A(task_dim_ + 3 * nb_joints_, 2 * nb_joints_ + nb_slacks_);
            A.setZero();
            // Task
            A.block(0, 0, task_dim_, nb_joints_) = J;
            A.block(0, 2 * nb_joints_, nb_slacks_, nb_slacks_) = -Eigen::MatrixXd::Identity(nb_slacks_, nb_slacks_);
            // Dynamics
            A.block(task_dim_, 0, nb_joints_, nb_joints_) = M;
            A.block(task_dim_, nb_joints_, nb_joints_, nb_joints_) = -Eigen::MatrixXd::Identity(nb_joints_, nb_joints_);
            // A.block(task_dim_,2*nb_joints_+task_dim_,nb_joints_,nb_joints_) =
            // 		-Eigen::MatrixXd::Identity(nb_joints_, nb_joints_);
            // Joint velocity
            A.block(task_dim_ + nb_joints_, 0, nb_joints_, nb_joints_) = dt * Eigen::MatrixXd::Identity(nb_joints_, nb_joints_);
            // Joint position
            A.block(task_dim_ + 2 * nb_joints_, 0, nb_joints_, nb_joints_) = 0.5 * dt * dt * Eigen::MatrixXd::Identity(nb_joints_, nb_joints_);

            // std::cout <<"4" << std::endl;

            //  Eigen::MatrixXd A(3*nb_joints_,2*nb_joints_+nb_slacks_);
            //  A.setZero();
            //  // Dynamics
            //  A.block(0,0,nb_joints_,nb_joints_) = M;
            //  A.block(0,nb_joints_,nb_joints_,nb_joints_) =
            //  		-Eigen::MatrixXd::Identity(nb_joints_, nb_joints_);
            // // Joint velocity
            //  A.block(nb_joints_,0,nb_joints_,nb_joints_) =
            //  		dt*Eigen::MatrixXd::Identity(nb_joints_, nb_joints_);
            // // Joint position
            //  A.block(2*nb_joints_,0,nb_joints_,nb_joints_) =
            //  		0.5*dt*dt*Eigen::MatrixXd::Identity(nb_joints_, nb_joints_);

            // Eigen::MatrixXd A(task_dim_+3*nb_joints_,nb_joints_+nb_slacks_);
            // A.setZero();
            // // Task
            // A.block(0,0,task_dim_,nb_joints_) = J;
            // A.block(0,nb_joints_,nb_slacks_,nb_slacks_) =
            //   -Eigen::MatrixXd::Identity(nb_slacks_, nb_slacks_);
            // // Dynamics
            // A.block(task_dim_,0,nb_joints_,nb_joints_) = M;
            // // Joint velocity
            // A.block(task_dim_+nb_joints_,0,nb_joints_,nb_joints_) =
            //     dt*Eigen::MatrixXd::Identity(nb_joints_, nb_joints_);
            // // Joint position
            // A.block(task_dim_+2*nb_joints_,0,nb_joints_,nb_joints_) =
            //     0.5*dt*dt*Eigen::MatrixXd::Identity(nb_joints_, nb_joints_);

            ////////////////////////
            // Constraints bounds //
            ////////////////////////
            Eigen::VectorXd lbA(task_dim_ + 3 * nb_joints_), ubA(task_dim_ + 3 * nb_joints_);
            // Task
            lbA.segment(0, task_dim_) = (error - dJ * dq);
            ubA.segment(0, task_dim_) = (error - dJ * dq);
            // Dynamics
            lbA.segment(task_dim_, nb_joints_) = -Cg;
            ubA.segment(task_dim_, nb_joints_) = -Cg;
            // Joint velocity
            lbA.segment(task_dim_ + nb_joints_, nb_joints_) = dq_min_ - dq;
            ubA.segment(task_dim_ + nb_joints_, nb_joints_) = dq_max_ - dq;
            // Joint position
            lbA.segment(task_dim_ + 2 * nb_joints_, nb_joints_) = q_min_ - dt * dq - q;
            ubA.segment(task_dim_ + 2 * nb_joints_, nb_joints_) = q_max_ - dt * dq - q;
            // Eigen::VectorXd lbA(3*nb_joints_), ubA(3*nb_joints_);
            // // Dynamics
            // lbA.segment(0,nb_joints_) = -Cg;
            // ubA.segment(0,nb_joints_) = -Cg;
            // // Joint velocity
            // lbA.segment(0+nb_joints_,nb_joints_) = dq_min_-dq;
            // ubA.segment(0+nb_joints_,nb_joints_) = dq_max_-dq;
            // // Joint position
            // lbA.segment(0+2*nb_joints_,nb_joints_) = q_min_-dt*dq-q;
            // ubA.segment(0+2*nb_joints_,nb_joints_) = q_max_-dt*dq-q;

            // Eigen::VectorXd lbA(task_dim_+3*nb_joints_), ubA(task_dim_+3*nb_joints_);
            // // Task
            // lbA.segment(0,task_dim_) = (error-dJ*dq);
            // ubA.segment(0,task_dim_) = (error-dJ*dq);
            // // Dynamics
            // lbA.segment(task_dim_,nb_joints_) = tau_min_-Cg;
            // ubA.segment(task_dim_,nb_joints_) = tau_max_-Cg;
            // // Joint velocity
            // lbA.segment(task_dim_+nb_joints_,nb_joints_) = dq_min_-dq;
            // ubA.segment(task_dim_+nb_joints_,nb_joints_) = dq_max_-dq;
            // // Joint position
            // lbA.segment(task_dim_+2*nb_joints_,nb_joints_) = q_min_-dt*dq-q;
            // ubA.segment(task_dim_+2*nb_joints_,nb_joints_) = q_max_-dt*dq-q;

            ////////////////////
            // Variables bounds //
            //////////////////////
            Eigen::VectorXd lb(2 * nb_joints_ + nb_slacks_), ub(2 * nb_joints_ + nb_slacks_);
            lb.segment(0, nb_joints_) = ddq_min_;
            ub.segment(0, nb_joints_) = ddq_max_;
            lb.segment(nb_joints_, nb_joints_) = tau_min_;
            ub.segment(nb_joints_, nb_joints_) = tau_max_;
            lb.segment(2 * nb_joints_, nb_slacks_) = slack_min_;
            ub.segment(2 * nb_joints_, nb_slacks_) = slack_max_;

            // Eigen::VectorXd lb(nb_joints_+nb_slacks_), ub(nb_joints_+nb_slacks_);
            // lb.segment(0,nb_joints_) = ddq_min_;
            // ub.segment(0,nb_joints_) = ddq_max_;
            // lb.segment(nb_joints_,nb_slacks_) = slack_min_;
            // ub.segment(nb_joints_,nb_slacks_) = slack_max_;

            Eigen::VectorXd x(2 * nb_joints_ + nb_slacks_);
            // Eigen::VectorXd x(nb_joints_+nb_slacks_);
            bool result = qp_solver_.step(x, H, A, g, lbA, ubA, lb, ub);
            tau = x.segment(nb_joints_, nb_joints_);
            // tau = M*x.segment(0,nb_joints_)+Cg;

            J_prev_ = J;
            std::cout << "tau:" << tau.transpose() << std::endl;
            // std::cout << "ddq:" << x.segment(0,nb_joints_).transpose() << std::endl;
            // std::cout << "M:" << std::endl << M << std::endl;
            // std::cout << "slack:" << x.segment(2*nb_joints_,nb_slacks_).transpose() << std::endl;
            // std::cout << "tau_max:" << tau_max_.transpose() << std::endl;
            return result;
        }
    } // namespace optimization
} // namespace geometric_control