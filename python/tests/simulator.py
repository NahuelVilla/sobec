#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 19:56:37 2022

@author: nvilla
"""

from bullet_Talos import BulletTalos
from cricket.virtual_talos import VirtualPhysics
from flex_joints import Flex
import numpy as np


class SimuEstimator:
    def __init__(self, conf, design):

        if conf.simulator == "bullet":
            device = BulletTalos(conf, design.get_rModelComplete())
            device.initializeJoints(design.get_q0Complete().copy())
            device.showTargetToTrack(design.get_LF_frame(), design.get_RF_frame())

        elif conf.simulator == "pinocchio":
            design.rmodelComplete = design.get_rModelComplete()
            design.rmodelComplete.q0 = design.get_q0Complete()
            design.rmodelComplete.v0 = design.get_v0Complete()

            device = VirtualPhysics(conf, view=True, block_joints=conf.blocked_joints)
            device.initialize(design.rmodelComplete)

        self.device = device
        self.conf = conf
        self.design = design

        if conf.model_name == "talos_flex":
            flex = Flex()
            flex.initialize(
                dict(
                    left_stiffness=np.array(conf.H_stiff[:2]),
                    right_stiffness=np.array(conf.H_stiff[2:]),
                    left_damping=np.array(conf.H_damp[:2]),
                    right_damping=np.array(conf.H_damp[2:]),
                    flexToJoint=conf.flexToJoint,
                    dt=conf.simu_period,
                    MA_duration=0.01,
                    left_hip_indices=np.array([0, 1, 2]),
                    right_hip_indices=np.array([6, 7, 8]),
                    filtered=True,
                )
            )
            self.flex = flex

    def measure_state(self):

        if self.conf.simulator == "bullet":
            q_current, v_current = self.device.measureState()

        elif self.conf.simulator == "pinocchio":
            state = self.device.measure_state(
                self.device.Cq0, self.device.Cv0, self.device.Ca0
            )

            q_current = state["q"]
            v_current = state["dq"]

        return q_current, v_current

    def iterate(self, command, contacts=None, iter_count=None, data=None):

        if self.conf.simulator == "bullet":
            torques = command["tau"]
            self.device.execute(torques)
            q_current, v_current = self.device.measureState()

        elif self.conf.simulator == "pinocchio":

            correct_contacts = contacts
            real_state, real_motion = self.device.execute(
                command, correct_contacts, iter_count
            )

            if self.conf.model_name == "talos":

                q_current = real_state["q"]
                v_current = real_state["dq"]

            elif self.conf.model_name == "talos_flex":
                Lforce, Rforce = self.estimate_hip_force(correct_contacts, data)
                qc, dqc = self.flex.correctEstimatedDeflections(
                    torques, real_state["q"][7:], real_state["dq"][6:], Lforce, Rforce
                )

                q_current = np.hstack([real_state["q"][:7], qc])
                v_current = np.hstack([real_state["dq"][:6], dqc])

        return q_current, v_current, real_motion

    def updateMarkers(self):
        pass

    # self.device.moveMarkers(mpc.ref_LF_poses[0], mpc.ref_RF_poses[0])

    def estimate_hip_force(self, correct_contacts, data=None):
        LW = data.f[2].linear
        RW = data.f[8].linear
        TW = data.f[1].linear

        if not all(correct_contacts.values()):
            Lforce = TW - LW if correct_contacts["leg_left_sole_fix_joint"] else -LW
            Rforce = TW - RW if correct_contacts["leg_right_sole_fix_joint"] else -RW
        else:
            Lforce = TW / 2 - LW
            Rforce = TW / 2 - RW

        return Lforce, Rforce

    def close(self):
        if self.conf.simulator == "bullet":
            self.device.close()
