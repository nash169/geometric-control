#ifndef GEOMETRICCONTROL_RIGIDBODY_HPP
#define GEOMETRICCONTROL_RIGIDBODY_HPP

#include <Magnum/BulletIntegration/DebugDraw.h>
#include <Magnum/BulletIntegration/Integration.h>
#include <Magnum/BulletIntegration/MotionState.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/Shaders/Phong.h>
#include <Magnum/Trade/PhongMaterialData.h>
#include <btBulletDynamicsCommon.h>

namespace geometric_control {
    using namespace Magnum;
    using namespace Math::Literals;

    typedef SceneGraph::Object<SceneGraph::MatrixTransformation3D> Object3D;

    class RigidBody : public Object3D {
    public:
        RigidBody(Object3D* parent, Float mass, btCollisionShape* bShape, btDynamicsWorld& bWorld) : Object3D{parent}, _bWorld(bWorld)
        {
            /* Calculate inertia so the object reacts as it should with
               rotation and everything */
            btVector3 bInertia(0.0f, 0.0f, 0.0f);
            if (!Math::TypeTraits<Float>::equals(mass, 0.0f))
                bShape->calculateLocalInertia(mass, bInertia);

            /* Bullet rigid body setup */
            auto* motionState = new BulletIntegration::MotionState{*this};
            _bRigidBody.emplace(btRigidBody::btRigidBodyConstructionInfo{
                mass, &motionState->btMotionState(), bShape, bInertia});
            _bRigidBody->forceActivationState(DISABLE_DEACTIVATION);
            bWorld.addRigidBody(_bRigidBody.get());
        }

        ~RigidBody()
        {
            _bWorld.removeRigidBody(_bRigidBody.get());
        }

        btRigidBody& rigidBody() { return *_bRigidBody; }

        /* needed after changing the pose from Magnum side */
        void syncPose()
        {
            _bRigidBody->setWorldTransform(btTransform(transformationMatrix()));
        }

    private:
        btDynamicsWorld& _bWorld;
        Containers::Pointer<btRigidBody> _bRigidBody;
    };
} // namespace geometric_control

#endif // GEOMETRICCONTROL_RIGIDBODY_HPP