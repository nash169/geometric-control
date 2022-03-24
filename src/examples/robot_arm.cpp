#include <iostream>

#include <beautiful_bullet/Simulator.hpp>
#include <beautiful_bullet/graphics/MagnumGraphics.hpp>
#include <utils_lib/FileManager.hpp>

// Bundle DS
#include <geometric_control/dynamics/BundleDynamics.hpp>

// Manifolds
#include <geometric_control/manifolds/Euclidean.hpp>
#include <geometric_control/manifolds/SpecialEuclidean.hpp>
#include <geometric_control/manifolds/Sphere.hpp>

// Tasks
#include <geometric_control/tasks/DissipativeEnergy.hpp>
#include <geometric_control/tasks/ObstacleAvoidance.hpp>
#include <geometric_control/tasks/PotentialEnergy.hpp>

using namespace geometric_control;
using namespace beautiful_bullet;
using namespace utils_lib;

// create the manifold multibody
class FrankaRobot : public bodies::MultiBody {
public:
    FrankaRobot() : bodies::MultiBody("rsc/iiwa/urdf/iiwa14.urdf"), _frame("lbr_iiwa_link_7") {}

    Eigen::VectorXd map(const Eigen::VectorXd& q)
    {
        return this->setState(q).framePose(_frame);
    }

    Eigen::MatrixXd jacobian(const Eigen::VectorXd& q)
    {
        return this->setState(q).jacobian(_frame);
    }

    Eigen::MatrixXd hessian(const Eigen::VectorXd& q, const Eigen::VectorXd& v)
    {
        return this->setState(q).setVelocity(v).hessian(_frame);
    }

protected:
    std::string _frame;
};

// Define the nodes in the tree dynamics
using TreeManifoldsImpl = TreeManifolds<FrankaRobot, manifolds::SpecialEuclidean<3>, manifolds::Sphere<2>>;

// Tree Mapping (non-specialize; this should be automatized)
template <typename ParentManifold>
class ManifoldsMapping : public TreeManifoldsImpl {
    Eigen::VectorXd map(const Eigen::VectorXd& x, FrankaRobot& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, FrankaRobot& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, FrankaRobot& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

    Eigen::VectorXd map(const Eigen::VectorXd& x, manifolds::SpecialEuclidean<3>& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, manifolds::SpecialEuclidean<3>& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, manifolds::SpecialEuclidean<3>& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

    Eigen::VectorXd map(const Eigen::VectorXd& x, manifolds::Sphere<2>& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, manifolds::Sphere<2>& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, manifolds::Sphere<2>& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
};

// Robot -> SE3
template <>
Eigen::VectorXd ManifoldsMapping<FrankaRobot>::map(const Eigen::VectorXd& x, manifolds::SpecialEuclidean<3>& manifold)
{
    return FrankaRobot().map(x);
}
template <>
Eigen::MatrixXd ManifoldsMapping<FrankaRobot>::jacobian(const Eigen::VectorXd& x, manifolds::SpecialEuclidean<3>& manifold)
{
    return FrankaRobot().jacobian(x);
}
template <>
Eigen::MatrixXd ManifoldsMapping<FrankaRobot>::hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, manifolds::SpecialEuclidean<3>& manifold)
{
    return FrankaRobot().hessian(x, v);
}

// SE3 -> S2
template <>
Eigen::VectorXd ManifoldsMapping<manifolds::SpecialEuclidean<3>>::map(const Eigen::VectorXd& x, manifolds::Sphere<2>& manifold)
{
    return x.head(3);
}
template <>
Eigen::MatrixXd ManifoldsMapping<manifolds::SpecialEuclidean<3>>::jacobian(const Eigen::VectorXd& x, manifolds::Sphere<2>& manifold)
{
    Eigen::MatrixXd j = Eigen::MatrixXd::Zero(3, x.size());
    j.block(0, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
    return j;
}
template <>
Eigen::MatrixXd ManifoldsMapping<manifolds::SpecialEuclidean<3>>::hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, manifolds::Sphere<2>& manifold)
{
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(3, x.size());
    h.block(0, 0, 3, 3).diagonal() = v.head(3);
    return h;
}

int main(int argc, char** argv)
{
    // Data
    double box[] = {0, M_PI, 0, 2 * M_PI};
    constexpr size_t dim = 2;
    size_t resolution = 100, num_samples = resolution * resolution;

    Eigen::MatrixXd gridX = Eigen::RowVectorXd::LinSpaced(resolution, box[0], box[1]).replicate(resolution, 1),
                    gridY = Eigen::VectorXd::LinSpaced(resolution, box[2], box[3]).replicate(1, resolution),
                    X(num_samples, dim);
    // Test points
    X << Eigen::Map<Eigen::VectorXd>(gridX.data(), gridX.size()), Eigen::Map<Eigen::VectorXd>(gridY.data(), gridY.size());

    // Create Bundle DS for each manifold
    dynamics::BundleDynamics<FrankaRobot, TreeManifoldsImpl, ManifoldsMapping> robot;
    dynamics::BundleDynamics<manifolds::SpecialEuclidean<3>, TreeManifoldsImpl, ManifoldsMapping> se3;
    dynamics::BundleDynamics<manifolds::Sphere<2>, TreeManifoldsImpl, ManifoldsMapping> s2;

    robot.addBundles(&se3);
    se3.addBundles(&s2);

    Eigen::VectorXd x1(7),
        v1 = Eigen::VectorXd::Random(7);
    x1 << 0., 0.7, 0.4, 0.6, 0.3, 0.5, 0.1;

    std::cout << "ROBOT to SE3" << std::endl;
    ManifoldsMapping<FrankaRobot> robotTose3;
    std::cout << "map" << std::endl;
    Eigen::VectorXd x2 = se3.mapFrom(x1, robotTose3);
    std::cout << x2.transpose() << std::endl;
    std::cout << "jacobian" << std::endl;
    Eigen::VectorXd v2 = se3.jacobianFrom(x1, robotTose3) * v1;
    std::cout << v2.transpose() << std::endl;
    std::cout << "hessian" << std::endl;
    std::cout << se3.hessianFrom(x1, v1, robotTose3) << std::endl;

    std::cout << "SE3 to S2" << std::endl;
    ManifoldsMapping<manifolds::SpecialEuclidean<3>> se3Tos2;
    std::cout << "map" << std::endl;
    std::cout << s2.mapFrom(x2, se3Tos2).transpose() << std::endl;
    std::cout << "jacobian" << std::endl;
    std::cout << (s2.jacobianFrom(x2, se3Tos2) * v2).transpose() << std::endl;
    std::cout << "hessian" << std::endl;
    std::cout << s2.hessianFrom(x2, v2, se3Tos2) << std::endl;

    std::cout << "COMPOSITION" << std::endl;
    std::cout << "map" << std::endl;
    std::cout << "jacobian" << std::endl;
    std::cout << "hessian" << std::endl;

    // Add tasks to the Bundle Dynamics over S2
    s2.addTasks(
        std::make_unique<tasks::PotentialEnergy<manifolds::Sphere<2>>>(),
        std::make_unique<tasks::DissipativeEnergy<manifolds::Sphere<2>>>(),
        std::make_unique<tasks::ObstacleAvoidance<manifolds::Sphere<2>>>());

    // Attractor
    Eigen::Vector3d a = manifolds::Sphere<2>().embedding(Eigen::Vector2d(1.5, 3));

    // Set tasks' properties
    static_cast<tasks::PotentialEnergy<manifolds::Sphere<2>>&>(s2.task(0)).setStiffness(Eigen::Matrix3d::Identity()).setAttractor(a);
    static_cast<tasks::DissipativeEnergy<manifolds::Sphere<2>>&>(s2.task(1)).setDissipativeFactor(10 * Eigen::Matrix3d::Identity());
    static_cast<tasks::ObstacleAvoidance<manifolds::Sphere<2>>&>(s2.task(2)).setRadius(0.4).setCenter(Eigen::Vector2d(1.2, 3.5)).setMetricParams(1, 1);

    // Embedding
    Eigen::VectorXd potential(num_samples);
    Eigen::MatrixXd embedding(num_samples, dim + 1);

    for (size_t i = 0; i < num_samples; i++) {
        embedding.row(i) = manifolds::Sphere<2>().embedding(X.row(i));
        potential(i) = s2.task(0).map(embedding.row(i))[0];
    }

    // Dynamics
    double time = 0, max_time = 30, dt = 0.001;
    size_t num_steps = std::ceil(max_time / dt) + 1, index = 0;
    Eigen::Vector3d x = manifolds::Sphere<2>().embedding(Eigen::Vector2d(0.7, 5)),
                    v = manifolds::Sphere<2>().project(x, (manifolds::Sphere<2>().embedding(Eigen::Vector2d(1.5, 3)) - x) * 0.01);
    v = manifolds::Sphere<2>().embeddingJacobian(Eigen::Vector2d(1, 4)) * Eigen::Vector2d(-1, 1);

    // Record
    Eigen::MatrixXd record = Eigen::MatrixXd::Zero(num_steps, 1 + 2 * (dim + 1));
    record.row(0)(0) = time;
    record.row(0).segment(1, dim + 1) = x;
    record.row(0).segment(dim + 2, dim + 1) = v;

    std::cout << "DYNAMICS ACCELERATION" << std::endl;
    std::cout << s2(x, v).transpose() << std::endl;

    // while (time < max_time && index < num_steps - 1) {
    //     // Velocity
    //     v = v + dt * Manifold().projector(x, ds(x, v));

    //     // Position
    //     x = Manifold().retraction(x, v, dt);

    //     // Step forward
    //     time += dt;
    //     index++;

    //     // Record
    //     record.row(index)(0) = time;
    //     record.row(index).segment(1, dim + 1) = x;
    //     record.row(index).segment(dim + 2, dim + 1) = v;
    // }

    // FileManager io_manager;
    // io_manager.setFile("rsc/sphere_bundle.csv");
    // io_manager.write("RECORD", record, "EMBEDDING", embedding, "POTENTIAL", potential);

    return 0;
}