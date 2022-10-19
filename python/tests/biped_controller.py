#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 13:44:21 2022

@author: nvilla
"""


from sobec_pywrap import WBC, HorizonManager, ModelMaker, Support
import pinocchio as pin
import numpy as np


class BipedControl:
    def __init__(self, conf, design):
        # Vector of Formulations
        MM_conf = dict(
            timeStep=conf.DT,
            gravity=conf.gravity,
            mu=conf.mu,
            coneBox=conf.cone_box,
            minNforce=conf.minNforce,
            maxNforce=conf.maxNforce,
            comHeight=conf.normal_height,
            omega=conf.omega,
            footSize=conf.footSize,
            wFootPlacement=conf.wFootPlacement,
            wStateReg=conf.wStateReg,
            wControlReg=conf.wControlReg,
            wLimit=conf.wLimit,
            wWrenchCone=conf.wWrenchCone,
            wForceTask=conf.wForceTask,
            wCoP=conf.wCoP,
            wDCM=conf.wDCM,
            stateWeights=conf.stateWeights,
            controlWeights=conf.controlWeight,
            forceWeights=conf.forceWeights,
            lowKinematicLimits=conf.lowKinematicLimits,
            highKinematicLimits=conf.highKinematicLimits,
            th_grad=conf.th_grad,
            th_stop=conf.th_stop,
        )

        formuler = ModelMaker()
        formuler.initialize(MM_conf, design)
        all_models = formuler.formulateHorizon(length=conf.T)
        ter_model = formuler.formulateTerminalStepTracker(Support.DOUBLE)

        # Horizon
        H_conf = dict(leftFootName=conf.lf_frame_name, rightFootName=conf.rf_frame_name)
        horizon = HorizonManager()
        horizon.initialize(H_conf, design.get_x0(), all_models, ter_model)

        # MPC
        wbc_conf = dict(
            totalSteps=conf.total_steps,
            T=conf.T,
            TdoubleSupport=conf.TdoubleSupport,
            TsingleSupport=conf.TsingleSupport,
            Tstep=conf.Tstep,
            ddpIteration=conf.ddpIteration,
            Dt=conf.DT,
            simu_step=conf.simu_period,
            Nc=conf.Nc,
        )

        mpc = WBC()
        mpc.initialize(
            wbc_conf,
            design,
            horizon,
            design.get_q0Complete(),
            design.get_v0Complete(),
            "actuationTask",
        )
        mpc.generateWalkingCycle(formuler)
        mpc.generateStandingCycle(formuler)

        self.conf = conf
        self.design = design
        self.mpc = mpc

    def get_timming(self):
        landing_LF = (
            self.mpc.land_LF()[0]
            if self.mpc.land_LF()
            else (
                self.mpc.takeoff_LF()[0] + self.conf.TsingleSupport
                if self.mpc.takeoff_LF()
                else 2 * self.mpc.horizon.size()
            )
        )
        landing_RF = (
            self.mpc.land_RF()[0]
            if self.mpc.land_RF()
            else (
                self.mpc.takeoff_RF()[0] + self.conf.TsingleSupport
                if self.mpc.takeoff_RF()
                else 2 * self.mpc.horizon.size()
            )
        )

        return {"landing_LF": landing_LF, "landing_RF": landing_RF}

    def set_references(self, ref_motion=None):

        LF_refs = ref_motion["LF_refs"]
        RF_refs = ref_motion["RF_refs"]

        self.mpc.ref_LF_poses[:] = [pin.SE3(np.eye(3), xyz) for xyz in LF_refs]
        self.mpc.ref_RF_poses[:] = [pin.SE3(np.eye(3), xyz) for xyz in RF_refs]

    def iterate(self, s, q_current, v_current):

        self.mpc.iterate(s, q_current, v_current)
        torques = self.mpc.horizon.currentTorques(self.mpc.x0)

        return {"tau": torques}
