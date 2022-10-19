#!/usr/bin/env python3
"""
Created on Sat Jun 11 17:42:39 2022
@author: nvilla
"""
import configuration as conf

from simulator import SimuEstimator
from guide import Guide
from biped_controller import BipedControl
from cricket.loggin import Loggin

# import pinocchio as pin
from sobec_pywrap import RobotDesigner  # , WBC, HorizonManager, ModelMaker, Support

# import numpy as np

# from time import time

# ####### CONFIGURATION  ############
# ### RobotWrapper
design_conf = dict(
    urdfPath=conf.modelPath + conf.URDF_SUBPATH,
    srdfPath=conf.modelPath + conf.SRDF_SUBPATH,
    leftFootName=conf.lf_frame_name,
    rightFootName=conf.rf_frame_name,
    robotDescription="",
    controlledJointsNames=[
        "root_joint",
        "leg_left_1_joint",
        "leg_left_2_joint",
        "leg_left_3_joint",
        "leg_left_4_joint",
        "leg_left_5_joint",
        "leg_left_6_joint",
        "leg_right_1_joint",
        "leg_right_2_joint",
        "leg_right_3_joint",
        "leg_right_4_joint",
        "leg_right_5_joint",
        "leg_right_6_joint",
        "torso_1_joint",
        "torso_2_joint",
        # "arm_left_1_joint",
        # "arm_left_2_joint",
        # "arm_left_3_joint",
        # "arm_left_4_joint",
        # "arm_right_1_joint",
        # "arm_right_2_joint",
        # "arm_right_3_joint",
        # "arm_right_4_joint",
    ],
)
design = RobotDesigner()
design.initialize(design_conf)

ref = Guide(conf, design)
control = BipedControl(conf, design)

simu = SimuEstimator(conf, design)
q_current, v_current = simu.measure_state()
logger = Loggin()

# ### SIMULATION LOOP ###
for s in range(10000):
    if control.mpc.timeToSolveDDP(s):

        timming = control.get_timming()

        ref_motion = ref.generate_references(timming)
        control.set_references(ref_motion)

    command = control.iterate(s, q_current, v_current)

    q_current, v_current, real_motion = simu.iterate(
        command, control.mpc.horizon.get_contacts(0), s, control.mpc.horizon.pinData(0)
    )

    logger.save_simple_CM(real_motion, s * 0.001)
    logger.save_simple_wrench(simu.device, s * 0.001)


logger.plot_simple_CM()
logger.plot_simple_wrench()
simu.close()
