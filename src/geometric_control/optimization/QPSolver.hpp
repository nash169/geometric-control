#ifndef GEOMETRICCONTROL_OPTIMIZATION_QPSOLVER_HPP
#define GEOMETRICCONTROL_OPTIMIZATION_QPSOLVER_HPP

#include <Eigen/Core>
#include <qpOASES.hpp>

namespace geometric_control {
    namespace optimization {
        class QPSolver {
        public:
            // Constructor
            QPSolver();
            // Destructor
            virtual ~QPSolver();

            void init(unsigned int nV, unsigned int nC);

            // Step function
            bool step(Eigen::VectorXd& x, const Eigen::MatrixXd& H,
                const Eigen::MatrixXd& A, const Eigen::VectorXd& g,
                const Eigen::VectorXd& lbA, const Eigen::VectorXd& ubA,
                const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);

        private:
            void copyQpOasesVariables();

            // Number of variables
            unsigned int nV_;

            // Number of constraints
            unsigned int nC_;

            // Hessian matrix
            Eigen::MatrixXd H_;
            // Constraint matrix
            Eigen::MatrixXd A_;
            // Gradient vector
            Eigen::VectorXd g_;
            // Lower/upper constraints bounds
            Eigen::VectorXd lbA_;
            Eigen::VectorXd ubA_;
            // Lower/upper variables bounds
            Eigen::VectorXd lb_;
            Eigen::VectorXd ub_;

            // QpOases variables
            qpOASES::real_t* H_qp_;
            qpOASES::real_t* A_qp_;
            qpOASES::real_t* g_qp_;
            qpOASES::real_t* lb_qp_;
            qpOASES::real_t* ub_qp_;
            qpOASES::real_t* lbA_qp_;
            qpOASES::real_t* ubA_qp_;

            // Create QP problem with varying matrices
            qpOASES::SQProblem* sqp_;

            // Booleans
            bool first_;

            int max_iters_;
        };
    } // namespace optimization
} // namespace geometric_control

#endif // GEOMETRICCONTROL_OPTIMIZATION_QPSOLVER_HPP