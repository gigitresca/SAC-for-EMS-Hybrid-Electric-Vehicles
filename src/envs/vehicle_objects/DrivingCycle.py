import pandas as pd
import numpy as np
from scipy.integrate import cumtrapz


class DrivingCycle:
    """
    Object that a driving cycle with the speed, acceleration, grade and travelled distance
    """
    def __init__(self, param=pd.DataFrame, name=""):
        """
        Constructor method: Initializes all the parameters of the driving cycle
        """
        self.name = name
        self.time_simu = param["Time"][1:].values.astype(float)
        self.time_simu = self.time_simu[~np.isnan(self.time_simu)]
        self.time_step = self.time_simu[1]-self.time_simu[0]
        self.veh_spd = param["Vehicle Speed"][1:].values.astype(float)
        self.veh_spd = self.veh_spd[~np.isnan(self.veh_spd)]
        if param["Vehicle Speed"][0] == "[km/h]":
            self.veh_spd = self.veh_spd / 3.6  # conversion km/h to m/s
        elif param["Vehicle Speed"][0] == "[mph]":
            self.veh_spd = self.veh_spd * (1609.34 / 3600)  # conversion mph to m/s
        self.veh_acc = np.append(np.array([0]), np.diff(self.veh_spd) / np.diff(self.time_simu))
        self.gear = param["Gear Number"][1:].values.astype(float)
        self.gear = self.gear[~np.isnan(self.gear)]
        self.gear = self.gear.astype(int)
        if 'Grade' in param.columns:
            self.grade = param["Grade"][1:].values.astype(float)
            self.grade = self.grade[~np.isnan(self.grade)]
        else:
            self.grade = np.zeros(len(self.veh_spd))
        self.veh_dist = np.append(np.array([0]), cumtrapz(self.veh_spd, self.time_simu))

    def resample(self, time_step_new=1):
        """
        Resample method to resample the driving cycle given the time step (time_step)
        """
        time_simu_new = np.arange(0, self.time_simu[-1]+self.time_step, time_step_new)
        veh_spd_new = np.interp(time_simu_new, self.time_simu, self.veh_spd)
        gear_new = np.round(np.interp(time_simu_new, self.time_simu, self.gear))
        gear_new = gear_new.astype(int)
        grade_new = np.interp(time_simu_new, self.time_simu, self.grade)
        veh_acc_new = np.append(np.array([0]), np.diff(gear_new) / np.diff(time_simu_new))
        veh_dist_new = np.append(np.array([0]), cumtrapz(veh_spd_new, time_simu_new))
        # Assignment to object properties
        self.time_simu = time_simu_new
        self.time_step = time_step_new
        self.veh_spd = veh_spd_new
        self.gear = gear_new
        self.grade = grade_new
        self.veh_acc = veh_acc_new
        self.veh_dist = veh_dist_new
