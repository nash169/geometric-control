#ifndef GEOMETRIC_CONTROL_TOOLS_MATH_HPP
#define GEOMETRIC_CONTROL_TOOLS_MATH_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

namespace geometric_control {
    namespace tools {
        Eigen::Tensor<double, 3> leviCivitaConnection(const Eigen::Tensor<double, 2>& gInv, const Eigen::Tensor<double, 3>& gGrad)
        {
            Eigen::array<Eigen::IndexPair<int>, 1> c1 = {Eigen::IndexPair<int>(1, 0)},
                                                   c2 = {Eigen::IndexPair<int>(1, 2)};

            return 0.5 * (gInv.contract(gGrad + gGrad.shuffle(Eigen::array<int, 3>({0, 2, 1})), c1) - gInv.contract(gGrad, c2));
        }

        Eigen::MatrixXd pullback(const Eigen::MatrixXd& g, const Eigen::MatrixXd& df)
        {
            return df.transpose() * g * df;
        }

        Eigen::MatrixXd gramSchmidt(const Eigen::MatrixXd& V)
        {
            size_t n_points = V.rows(), n_features = V.cols();

            Eigen::MatrixXd M(n_points * n_features, n_features);

            for (size_t i = 0; i < n_points; i++) {
                Eigen::MatrixXd mat(n_features, n_features);

                for (size_t j = 0; j < n_features; j++)
                    mat.col(j) = V.row(i).transpose();

                Eigen::HouseholderQR<Eigen::MatrixXd> qr(mat);

                M.block(i * n_features, 0, n_features, n_features) = qr.householderQ();
            }

            return -M;
        }

        Eigen::MatrixXd frameMatrix(const Eigen::VectorXd& u)
        {
            Eigen::Matrix3d oTemp = gramSchmidt(u.transpose());
            Eigen::Matrix3d oInit;
            oInit.col(0) = oTemp.col(1);
            oInit.col(1) = oTemp.col(2);
            oInit.col(2) = oTemp.col(0);

            return oInit;
        }
    } // namespace tools
} // namespace geometric_control
#endif // GEOMETRIC_CONTROL_TOOLS_MATH_HPP