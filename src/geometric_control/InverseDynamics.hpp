#ifndef GEOMETRIC_CONTROL_INVERSE_DYNAMICS_HPP
#define GEOMETRIC_CONTROL_INVERSE_DYNAMICS_HPP

#include <iostream>
#include <vector>

#include <btBulletDynamicsCommon.h>

#include <Bullet3Common/b3FileUtils.h>
#include <Bullet3Common/b3Logging.h>

#include <BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h>
#include <BulletDynamics/Featherstone/btMultiBodyJointMotor.h>
#include <BulletDynamics/Featherstone/btMultiBodyLinkCollider.h>

#include <BulletInverseDynamics/IDConfig.hpp>

#include "geometric_control/interfaces/CommonMultiBodyBase.h"
#include "geometric_control/interfaces/CommonParameterInterface.h"

#include <InverseDynamics/btMultiBodyTreeCreator.hpp>

#include "geometric_control/rendering/TimeSeriesCanvas.h"

#include "geometric_control/importers/ImportURDFDemo/BulletUrdfImporter.h"
#include "geometric_control/importers/ImportURDFDemo/MyMultiBodyCreator.h"
#include "geometric_control/importers/ImportURDFDemo/URDF2Bullet.h"

enum btInverseDynamicsOptions {
    BT_ID_LOAD_URDF = 0,
    BT_ID_PROGRAMMATICALLY = 1
};

class CommonExampleInterface* InverseDynamicsCreateFunc(struct CommonExampleOptions& options);

// the UI interface makes it easier to use static variables & free functions
// as parameters and callbacks
static btScalar kp = 10 * 10;
static btScalar kd = 2 * 10;
static bool useInverseModel = true;
static std::vector<btScalar> qd;
static std::vector<std::string> qd_name;
static std::vector<std::string> q_name;

static btVector4 sJointCurveColors[8] = {
    btVector4(1, 0.3, 0.3, 1),
    btVector4(0.3, 1, 0.3, 1),
    btVector4(0.3, 0.3, 1, 1),
    btVector4(0.3, 1, 1, 1),
    btVector4(1, 0.3, 1, 1),
    btVector4(1, 1, 0.3, 1),
    btVector4(1, 0.7, 0.7, 1),
    btVector4(0.7, 1, 1, 1),

};

class InverseDynamics : public CommonMultiBodyBase {
    btInverseDynamicsOptions m_option;
    btMultiBody* m_multiBody;
    btInverseDynamics::MultiBodyTree* m_inverseModel;
    TimeSeriesCanvas* m_timeSeriesCanvas;

public:
    InverseDynamics(struct GUIHelperInterface* helper, btInverseDynamicsOptions option)
        : CommonMultiBodyBase(helper),
          m_option(option),
          m_multiBody(0),
          m_inverseModel(0),
          m_timeSeriesCanvas(0)
    {
    }

    virtual ~InverseDynamics()
    {
        delete m_inverseModel;
        delete m_timeSeriesCanvas;
    }

    virtual void initPhysics()
    {
        //roboticists like Z up
        int upAxis = 2;
        m_guiHelper->setUpAxis(upAxis);

        createEmptyDynamicsWorld();
        btVector3 gravity(0, 0, 0);
        // gravity[upAxis]=-9.8;
        m_dynamicsWorld->setGravity(gravity);

        m_guiHelper->createPhysicsDebugDrawer(m_dynamicsWorld);

        {
            SliderParams slider("Kp", &kp);
            slider.m_minVal = 0;
            slider.m_maxVal = 2000;
            if (m_guiHelper->getParameterInterface())
                m_guiHelper->getParameterInterface()->registerSliderFloatParameter(slider);
        }
        {
            SliderParams slider("Kd", &kd);
            slider.m_minVal = 0;
            slider.m_maxVal = 50;
            if (m_guiHelper->getParameterInterface())
                m_guiHelper->getParameterInterface()->registerSliderFloatParameter(slider);
        }

        BulletURDFImporter u2b(m_guiHelper, 0, 0, 1, 0);
        bool loadOk = u2b.loadURDF("rsc/kuka_iiwa/model.urdf"); // lwr / kuka.urdf");
        if (loadOk) {
            int rootLinkIndex = u2b.getRootLinkIndex();
            b3Printf("urdf root link index = %d\n", rootLinkIndex);
            MyMultiBodyCreator creation(m_guiHelper);
            btTransform identityTrans;
            identityTrans.setIdentity();
            ConvertURDF2Bullet(u2b, creation, identityTrans, m_dynamicsWorld, true, u2b.getPathPrefix());
            for (int i = 0; i < u2b.getNumAllocatedCollisionShapes(); i++) {
                m_collisionShapes.push_back(u2b.getAllocatedCollisionShape(i));
            }
            m_multiBody = creation.getBulletMultiBody();
            if (m_multiBody) {
                //kuka without joint control/constraints will gain energy explode soon due to timestep/integrator
                //temporarily set some extreme damping factors until we have some joint control or constraints
                m_multiBody->setAngularDamping(0 * 0.99);
                m_multiBody->setLinearDamping(0 * 0.99);
                b3Printf("Root link name = %s", u2b.getLinkName(u2b.getRootLinkIndex()).c_str());
            }
        }

        if (m_multiBody) {
            {
                std::cout << std::endl
                          << "hello: " << m_guiHelper->getParameterInterface() << std::endl;
                if (m_guiHelper->getAppInterface() && m_guiHelper->getParameterInterface()) {
                    m_timeSeriesCanvas = new TimeSeriesCanvas(m_guiHelper->getAppInterface()->m_2dCanvasInterface, 512, 230, "Joint Space Trajectory");
                    m_timeSeriesCanvas->setupTimeSeries(3, 100, 0);
                }
            }

            // construct inverse model
            btInverseDynamics::btMultiBodyTreeCreator id_creator;
            if (-1 == id_creator.createFromBtMultiBody(m_multiBody, false)) {
                b3Error("error creating tree\n");
            }
            else {
                m_inverseModel = btInverseDynamics::CreateMultiBodyTree(id_creator);
            }
            // add joint target controls
            qd.resize(m_multiBody->getNumDofs());

            qd_name.resize(m_multiBody->getNumDofs());
            q_name.resize(m_multiBody->getNumDofs());

            if (m_timeSeriesCanvas && m_guiHelper->getParameterInterface()) {
                for (std::size_t dof = 0; dof < qd.size(); dof++) {
                    qd[dof] = 0;
                    char tmp[25];
                    sprintf(tmp, "q_desired[%lu]", dof);
                    qd_name[dof] = tmp;
                    SliderParams slider(qd_name[dof].c_str(), &qd[dof]);
                    slider.m_minVal = -3.14;
                    slider.m_maxVal = 3.14;

                    sprintf(tmp, "q[%lu]", dof);
                    q_name[dof] = tmp;
                    m_guiHelper->getParameterInterface()->registerSliderFloatParameter(slider);
                    btVector4 color = sJointCurveColors[dof & 7];
                    m_timeSeriesCanvas->addDataSource(q_name[dof].c_str(), color[0] * 255, color[1] * 255, color[2] * 255);
                }
            }
        }

        m_guiHelper->autogenerateGraphicsObjects(m_dynamicsWorld);
    }

    virtual void stepSimulation(float deltaTime)
    {
        if (m_multiBody) {
            const int num_dofs = m_multiBody->getNumDofs();
            btInverseDynamics::vecx nu(num_dofs), qdot(num_dofs), q(num_dofs), joint_force(num_dofs);
            btInverseDynamics::vecx pd_control(num_dofs);

            // compute joint forces from one of two control laws:
            // 1) "computed torque" control, which gives perfect, decoupled,
            //    linear second order error dynamics per dof in case of a
            //    perfect model and (and negligible time discretization effects)
            // 2) decoupled PD control per joint, without a model
            for (int dof = 0; dof < num_dofs; dof++) {
                q(dof) = m_multiBody->getJointPos(dof);
                qdot(dof) = m_multiBody->getJointVel(dof);

                const btScalar qd_dot = 0;
                const btScalar qd_ddot = 0;
                if (m_timeSeriesCanvas)
                    m_timeSeriesCanvas->insertDataAtCurrentTime(q[dof], dof, true);

                // pd_control is either desired joint torque for pd control,
                // or the feedback contribution to nu
                pd_control(dof) = kd * (qd_dot - qdot(dof)) + kp * (qd[dof] - q(dof));
                // nu is the desired joint acceleration for computed torque control
                nu(dof) = qd_ddot + pd_control(dof);
            }
            if (useInverseModel) {
                // calculate joint forces corresponding to desired accelerations nu
                if (m_multiBody->hasFixedBase()) {
                    if (-1 != m_inverseModel->calculateInverseDynamics(q, qdot, nu, &joint_force)) {
                        //joint_force(dof) += damping*dot_q(dof);
                        // use inverse model: apply joint force corresponding to
                        // desired acceleration nu

                        for (int dof = 0; dof < num_dofs; dof++) {
                            m_multiBody->addJointTorque(dof, joint_force(dof));
                        }
                    }
                }
                else {
                    //the inverse dynamics model represents the 6 DOFs of the base, unlike btMultiBody.
                    //append some dummy values to represent the 6 DOFs of the base
                    btInverseDynamics::vecx nu6(num_dofs + 6), qdot6(num_dofs + 6), q6(num_dofs + 6), joint_force6(num_dofs + 6);
                    for (int i = 0; i < num_dofs; i++) {
                        nu6[6 + i] = nu[i];
                        qdot6[6 + i] = qdot[i];
                        q6[6 + i] = q[i];
                        joint_force6[6 + i] = joint_force[i];
                    }
                    if (-1 != m_inverseModel->calculateInverseDynamics(q6, qdot6, nu6, &joint_force6)) {
                        //joint_force(dof) += damping*dot_q(dof);
                        // use inverse model: apply joint force corresponding to
                        // desired acceleration nu

                        for (int dof = 0; dof < num_dofs; dof++) {
                            m_multiBody->addJointTorque(dof, joint_force6(dof + 6));
                        }
                    }
                }
            }
            else {
                for (int dof = 0; dof < num_dofs; dof++) {
                    // no model: just apply PD control law
                    m_multiBody->addJointTorque(dof, pd_control(dof));
                }
            }
        }

        if (m_timeSeriesCanvas)
            m_timeSeriesCanvas->nextTick();

        //todo: joint damping for btMultiBody, tune parameters

        // step the simulation
        if (m_dynamicsWorld) {
            // todo(thomas) check that this is correct:
            // want to advance by 10ms, with 1ms timesteps
            m_dynamicsWorld->stepSimulation(1e-3, 0); //,1e-3);
            btAlignedObjectArray<btQuaternion> scratch_q;
            btAlignedObjectArray<btVector3> scratch_m;
            m_multiBody->forwardKinematics(scratch_q, scratch_m);
#if 0
		for (int i = 0; i < m_multiBody->getNumLinks(); i++)
		{
			//btVector3 pos = m_multiBody->getLink(i).m_cachedWorldTransform.getOrigin();
			btTransform tr = m_multiBody->getLink(i).m_cachedWorldTransform;
			btVector3 pos = tr.getOrigin() - quatRotate(tr.getRotation(), m_multiBody->getLink(i).m_dVector);
			btVector3 localAxis = m_multiBody->getLink(i).m_axes[0].m_topVec;
			//printf("link %d: %f,%f,%f, local axis:%f,%f,%f\n", i, pos.x(), pos.y(), pos.z(), localAxis.x(), localAxis.y(), localAxis.z());
		}
#endif
        }
    }

    void setFileName(const char* urdfFileName);

    virtual void resetCamera()
    {
        float dist = 1.5;
        float pitch = -10;
        float yaw = -80;
        float targetPos[3] = {0, 0, 0};
        m_guiHelper->resetCamera(dist, yaw, pitch, targetPos[0], targetPos[1], targetPos[2]);
    }
};

CommonExampleInterface* InverseDynamicsCreateFunc(CommonExampleOptions& options)
{
    return new InverseDynamics(options.m_guiHelper, btInverseDynamicsOptions(options.m_option));
}

// B3_STANDALONE_EXAMPLE(InverseDynamicsCreateFunc)

#endif // GEOMETRIC_CONTROL_INVERSE_DYNAMICS_HPP
