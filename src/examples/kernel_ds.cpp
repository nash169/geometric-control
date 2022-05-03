#include <iostream>
#include <random>

// File Manager
#include <utils_lib/FileManager.hpp>
#include <utils_lib/Timer.hpp>

// Kernel
#include <kernel_lib/kernels/RiemannMatern.hpp>
#include <kernel_lib/kernels/RiemannSqExp.hpp>
#include <kernel_lib/kernels/SquaredExp.hpp>
#include <kernel_lib/utils/Expansion.hpp>

// Bundle DS
#include <geometric_control/dynamics/BundleDynamics.hpp>

// Base manifold
#include <geometric_control/manifolds/Euclidean.hpp>
#include <geometric_control/manifolds/Sphere.hpp>

// Tasks
#include <geometric_control/tasks/DissipativeEnergy.hpp>
#include <geometric_control/tasks/KernelEnergy.hpp>
#include <geometric_control/tasks/ObstacleAvoidance.hpp>

using namespace geometric_control;
using namespace utils_lib;
using namespace kernel_lib;

// Define Kernel
struct ParamsEigenfunction {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0);
        PARAM_SCALAR(double, sn, -5);
    };

    struct exp_sq : public defaults::exp_sq {
        PARAM_SCALAR(double, l, -0.6931); // -4.6052 -2.99573 -2.30259 -0.6931 (0.01 0.05 0.1 0.5)
    };
};

struct ParamsKernel {
    struct kernel : public defaults::kernel {
        PARAM_SCALAR(double, sf, 0);
        PARAM_SCALAR(double, sn, -5);
    };

    struct riemann_exp_sq : public defaults::riemann_exp_sq {
        PARAM_SCALAR(double, l, 0);
    };

    struct riemann_matern : public defaults::riemann_matern {
        PARAM_SCALAR(double, l, -0.6931);

        PARAM_SCALAR(double, d, 2);

        PARAM_SCALAR(double, nu, 1.5);
    };
};

using Kernel_t = kernels::SquaredExp<ParamsEigenfunction>;
using Expansion_t = utils::Expansion<ParamsEigenfunction, Kernel_t>;
using RiemannExp_t = kernels::RiemannSqExp<ParamsKernel, Expansion_t>;
using RiemannMatern_t = kernels::RiemannMatern<ParamsKernel, Expansion_t>;

// Define base manifold
using Manifold = manifolds::Sphere<2>;

// Define the nodes in the tree dynamics
using TreeManifoldsImpl = TreeManifolds<Manifold>;

// Tree Mapping
template <typename ParentManifold>
class ManifoldsMapping : public TreeManifoldsImpl {
    Eigen::VectorXd map(const Eigen::VectorXd& x, Manifold& manifold) override { return Eigen::VectorXd::Zero(x.size()); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& x, Manifold& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }
    Eigen::MatrixXd hessian(const Eigen::VectorXd& x, const Eigen::VectorXd& v, Manifold& manifold) override { return Eigen::MatrixXd::Zero(x.size(), x.size()); }

public:
    ParentManifold* _manifold;
};

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

    // Create DS on a specific manifold
    dynamics::BundleDynamics<Manifold, TreeManifoldsImpl, ManifoldsMapping> ds;

    // Dissipative task
    ds.addTasks(std::make_unique<tasks::DissipativeEnergy<Manifold>>());
    static_cast<tasks::DissipativeEnergy<Manifold>&>(ds.task(0)).setDissipativeFactor(5 * Eigen::Matrix3d::Identity());

    // Potential task
    ds.addTasks(std::make_unique<tasks::KernelEnergy<Manifold, RiemannMatern_t>>());

    std::string manifold = "sphere";
    int num_modes = 100;

    FileManager mn;

    // Samples on manifold
    Eigen::MatrixXd N = mn.setFile("rsc/" + manifold + "_nodes.csv").read<Eigen::MatrixXd>(),
                    I = mn.setFile("rsc/" + manifold + "_elems.csv").read<Eigen::MatrixXd>();

    // Eigenvalues and eigenvectors
    Eigen::VectorXd D = mn.setFile("rsc/" + manifold + "_eigval.csv").read<Eigen::MatrixXd>();
    Eigen::MatrixXd U = mn.setFile("rsc/" + manifold + "_eigvec.csv").read<Eigen::MatrixXd>().transpose();

    for (size_t i = 0; i < num_modes; i++) {
        Expansion_t f; // Create eigenfunction
        f.setSamples(N).setWeights(U.col(i)); // Set manifold sampled points and weights
        static_cast<tasks::KernelEnergy<Manifold, RiemannMatern_t>&>(ds.task(1)).kernel().addPair(D(i), f); // Add eigen-pair to Riemann kernel
    }

    Eigen::Vector3d a(-0.901173, 0.426548, -0.0770997);
    static_cast<tasks::KernelEnergy<Manifold, RiemannMatern_t>&>(ds.task(1)).setStiffness(Eigen::Matrix3d::Identity()).setAttractor(a.normalized());

    // Obstacles tasks
    size_t num_obstacles = 50;
    double radius_obstacles = 0.1;
    Eigen::MatrixXd center_obstacles(num_obstacles, ds.manifold().eDim());

    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr_x1(0, M_PI), distr_x2(0, 2 * M_PI);

    for (size_t i = 0; i < num_obstacles; i++) {
        ds.addTasks(std::make_unique<tasks::ObstacleAvoidance<Manifold>>());
        Eigen::Vector2d oCenter(distr_x1(eng), distr_x2(eng));
        static_cast<tasks::ObstacleAvoidance<Manifold>&>(ds.task(i + 2))
            .setRadius(radius_obstacles)
            .setCenter(oCenter)
            .setMetricParams(1, 3);
        center_obstacles.row(i) = ds.manifold().embedding(oCenter);
    }

    // Embedding
    Eigen::VectorXd potential(num_samples);
    Eigen::MatrixXd embedding(num_samples, dim + 1);

#pragma omp parallel for
    for (size_t i = 0; i < num_samples; i++) {
        embedding.row(i) = ds.manifold().embedding(X.row(i));
        potential(i) = ds.task(1).map(embedding.row(i))[0];
    }

    // Dynamics
    double time = 0, max_time = 5, dt = 0.01;
    size_t num_steps = std::ceil(max_time / dt) + 1, index = 0;
    Eigen::Vector3d x = ds.manifold().embedding(Eigen::Vector2d(0.7, 6)),
                    v = ds.manifold().project(x, (a - x) * 0.005);

    {
        Timer t;
        std::cout << v.transpose() << std::endl;
        std::cout << ds(x, v).transpose() << std::endl;
    }

    // Record
    Eigen::MatrixXd record = Eigen::MatrixXd::Zero(num_steps, 1 + 2 * (dim + 1));
    record.row(0)(0) = time;
    record.row(0).segment(1, dim + 1) = x;
    record.row(0).segment(dim + 2, dim + 1) = v;

    while (time < max_time && index < num_steps - 1) {
        // Integration
        v = v + dt * ds(x, v);
        x = ds.manifold().retract(x, v, dt);

        // Step forward
        time += dt;
        index++;

        // Record
        record.row(index)(0) = time;
        record.row(index).segment(1, dim + 1) = x;
        record.row(index).segment(dim + 2, dim + 1) = v;
    }

    FileManager io_manager;
    io_manager.setFile("outputs/kernel_bundle.csv");
    io_manager.write("RECORD", record, "EMBEDDING", embedding, "POTENTIAL", potential, "TARGET", a, "RADIUS", radius_obstacles, "CENTER", center_obstacles);

    return 0;
}