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

        Eigen::Matrix3d frameMatrix(const Eigen::VectorXd& u)
        {
            Eigen::Matrix3d oTemp = gramSchmidt(u.transpose());
            Eigen::Matrix3d oInit;
            oInit.col(0) = oTemp.col(1);
            oInit.col(1) = oTemp.col(2);
            oInit.col(2) = oTemp.col(0);

            return oInit;
        }

        Eigen::Vector3d rotationError(const Eigen::Matrix3d& R_current, const Eigen::Matrix3d& R_desired)
        {
            Eigen::AngleAxisd aa = Eigen::AngleAxisd(R_current.transpose() * R_desired);

            return aa.axis() * aa.angle();
        }

        Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v)
        {
            return (Eigen::Matrix3d() << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0).finished();
        }

        Eigen::Matrix3d rotationAlign(const Eigen::Vector3d& u, const Eigen::Vector3d& v)
        {
            Eigen::Vector3d k = u.cross(v);
            Eigen::Matrix3d K = skewSymmetric(k);
            double c = u.dot(v), s = k.norm();

            return Eigen::Matrix3d::Identity() + K + K * K / (1 + c);
        }

        Eigen::Matrix3d expTransOperator(const Eigen::Matrix3d& R) // u->t se3->SE3 (V)
        {
            Eigen::AngleAxisd aa(R);
            Eigen::Vector3d omega = aa.angle() * aa.axis();
            Eigen::Matrix3d omega_x = (Eigen::Matrix3d() << 0, -omega(2), omega(1), omega(2), 0, -omega(0), -omega(1), omega(0), 0).finished();

            double theta = omega.norm(),
                   A = std::sin(theta) / theta,
                   B = (1 - std::cos(theta)) / std::pow(theta, 2),
                   C = (1 - A) / std::pow(theta, 2);

            return Eigen::Matrix3d::Identity() + B * omega_x + C * omega_x * omega_x;
        }

        Eigen::Matrix3d logTransOperator(const Eigen::Matrix3d& R) // t->u SE3->se3 (V^-1)
        {
            Eigen::AngleAxisd aa(R);
            Eigen::Vector3d omega = aa.angle() * aa.axis();
            Eigen::Matrix3d omega_x = (Eigen::Matrix3d() << 0, -omega(2), omega(1), omega(2), 0, -omega(0), -omega(1), omega(0), 0).finished();

            double theta = omega.norm(),
                   A = std::sin(theta) / theta,
                   B = (1 - std::cos(theta)) / std::pow(theta, 2);

            return Eigen::Matrix3d::Identity() - 0.5 * omega_x + (1 - A / 2 / B) / std::pow(theta, theta) * omega_x * omega_x;
        }

    } // namespace tools
} // namespace geometric_control
#endif // GEOMETRIC_CONTROL_TOOLS_MATH_HPP