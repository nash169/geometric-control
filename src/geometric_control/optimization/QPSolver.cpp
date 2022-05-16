#include "geometric_control/optimization/QPSolver.hpp"

namespace geometric_control {
    namespace optimization {
        QPSolver::QPSolver()
        {
            first_ = false;
        }

        QPSolver::~QPSolver()
        {
            delete H_qp_;
            delete A_qp_;
            delete g_qp_;
            delete lb_qp_;
            delete ub_qp_;
            delete lbA_qp_;
            delete ubA_qp_;
            delete sqp_;
        }

        void QPSolver::init(unsigned int nV, unsigned int nC)
        {
            nV_ = nV;
            nC_ = nC;

            H_.resize(nV_, nV_);
            A_.resize(nC_, nV_);
            g_.resize(nV_);
            lb_.resize(nV_);
            ub_.resize(nV_);
            lbA_.resize(nC_);
            ubA_.resize(nC_);

            H_.setZero();
            A_.setZero();
            g_.setZero();
            lb_.setZero();
            ub_.setZero();
            lbA_.setZero();
            ubA_.setZero();

            H_qp_ = new qpOASES::real_t[nV_ * nV_];
            A_qp_ = new qpOASES::real_t[nC_ * nV_];
            g_qp_ = new qpOASES::real_t[nV_];
            lb_qp_ = new qpOASES::real_t[nV_];
            ub_qp_ = new qpOASES::real_t[nV_];
            lbA_qp_ = new qpOASES::real_t[nC_];
            ubA_qp_ = new qpOASES::real_t[nC_];

            sqp_ = new qpOASES::SQProblem(nV_, nC_);

            auto options = sqp_->getOptions();

            options.printLevel = qpOASES::PL_LOW;
            options.enableRamping = qpOASES::BT_FALSE;
            options.enableNZCTests = qpOASES::BT_FALSE;
            options.enableDriftCorrection = 0;
            options.terminationTolerance = 1e-6;
            options.boundTolerance = 1e-4;
            options.epsIterRef = 1e-6;
            sqp_->setOptions(options);

            max_iters_ = 10000;
        }

        bool QPSolver::step(Eigen::VectorXd& x, const Eigen::MatrixXd& H,
            const Eigen::MatrixXd& A, const Eigen::VectorXd& g,
            const Eigen::VectorXd& lbA, const Eigen::VectorXd& ubA,
            const Eigen::VectorXd& lb, const Eigen::VectorXd& ub)
        {

            assert(H.rows() == nV_ && H.cols() == nV_);
            H_ = H;
            assert(A.rows() == nC_ && A.cols() == nV_);
            A_ = A;
            assert(g.size() == nV_);
            g_ = g;
            assert(lbA.size() == nC_ && ubA.size() == nC_);
            lbA_ = lbA;
            ubA_ = ubA;
            assert(lb.size() == nV_ && ub.size() == nV_);
            lb_ = lb;
            ub_ = ub;

            copyQpOasesVariables();

            qpOASES::SymSparseMat H_mat(nV_, nV_, nV_, H_qp_);
            H_mat.createDiagInfo();
            qpOASES::SparseMatrix A_mat(nC_, nV_, nV_, A_qp_);

            qpOASES::returnValue ret = qpOASES::TERMINAL_LIST_ELEMENT;

            int max_iters = 1000;

            // if (!first_) {
            ret = sqp_->init(&H_mat, g_qp_, &A_mat, lb_qp_, ub_qp_,
                lbA_qp_, ubA_qp_, max_iters);
            first_ = true;
            // } else {
            //  	ret = sqp_->hotstart(&H_mat, g_qp_, &A_mat, lb_qp_, ub_qp_,
            //  		                   lbA_qp_, ubA_qp_, max_iters_);
            // }

            if (ret == qpOASES::SUCCESSFUL_RETURN) {
                qpOASES::real_t x_qp[nV_];
                sqp_->getPrimalSolution(x_qp);
                memcpy(x.data(), x_qp, nV_ * sizeof(double));
                return true;
            }
            else {
                return false;
            }
        }

        void QPSolver::copyQpOasesVariables()
        {
            for (size_t i = 0; i < nV_; i++) {
                for (size_t j = 0; j < nV_; j++) {
                    H_qp_[i * nV_ + j] = H_(i, j);
                }
            }

            for (size_t i = 0; i < nC_; i++) {
                for (size_t j = 0; j < nV_; j++) {
                    A_qp_[i * nV_ + j] = A_(i, j);
                }
            }

            for (size_t i = 0; i < nV_; i++) {
                g_qp_[i] = g_(i);
                lb_qp_[i] = lb_(i);
                ub_qp_[i] = ub_(i);
            }

            for (size_t i = 0; i < nC_; i++) {
                lbA_qp_[i] = lbA_(i);
                ubA_qp_[i] = ubA_(i);
            }
        }
    } // namespace optimization
} // namespace geometric_control