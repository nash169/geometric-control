#include <geometric_control/BulletPhysics.hpp>
#include <iostream>

using namespace geometric_control;

int main(int argc, char** argv)
{
    BulletPhysics app({argc, argv});
    return app.exec();
}