#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 13:03:35 2022

@author: nvilla
"""

import matplotlib.pyplot as plt
import numpy as np


class Guide:
    def __init__(self, conf, design):

        self.conf = conf
        self.design = design

    def foot_trajectory(self, time_to_land, translation, trajectory="sine"):
        """Functions to generate steps."""
        horizon_length = self.conf.T
        tmax = self.conf.TsingleSupport
        landing_advance = 10  # 3
        takeoff_delay = 10  # 9
        times = range(
            time_to_land - landing_advance,
            time_to_land - landing_advance - horizon_length,
            -1,
        )

        z = []
        if trajectory == "sine":
            for t in times:
                z.append(
                    0
                    if t < 0 or t > tmax - landing_advance - takeoff_delay
                    else (np.sin(t * np.pi / (tmax - landing_advance - takeoff_delay)))
                    * 0.05
                )

        else:
            for t in times:
                z.append(
                    0
                    if t < 0 or t > tmax - landing_advance - takeoff_delay
                    else (
                        1
                        - np.cos(
                            2 * t * np.pi / (tmax - landing_advance - takeoff_delay)
                        )
                    )
                    * 0.025
                )

        return [np.array([translation[0], translation[1], move_z]) for move_z in z]

    def generate_references(self, timming):

        landing_LF = timming["landing_LF"]
        landing_RF = timming["landing_RF"]

        LF_refs = self.foot_trajectory(
            landing_LF, self.design.get_LF_frame().translation, "cosine"
        )
        RF_refs = self.foot_trajectory(
            landing_RF, self.design.get_RF_frame().translation, "cosine"
        )

        ref_motion = {"LF_refs": LF_refs, "RF_refs": RF_refs}

        return ref_motion

    def print_trajectory(self, ref):
        u = [y.translation for y in ref]
        t = np.array([z[2] for z in u])
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(t)
        ax.set_ylim(0, 0.05)
