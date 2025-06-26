import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.interpolate import interp2d


class ElectricMotor:
    """
    Object that defines a generic electric motor described through performance maps
    """

    def __init__(self, params=pd.DataFrame, limits_df=pd.DataFrame, effpwr_df=pd.DataFrame, reg_df=pd.DataFrame):
        """
        Constructor method: Initializes all the parameters of the electric motor
        """
        self.mass = params["Mass"][1]
        self.inertia = params["Inertia"][1]
        self.gear_ratio = params["Gear ratio"][1]

        # Torque and power limits
        trq_max = namedtuple('trq_max', ['spd', 'data'])
        trq_min = namedtuple('trq_min', ['spd', 'data'])
        pwr_max = namedtuple('pwr_max', ['spd', 'data'])
        pwr_min = namedtuple('pwr_min', ['spd', 'data'])
        self.trq_max = trq_max(spd=limits_df["Speed"][1:].values.astype(float),
                               data=limits_df["Max Torque"][1:].values.astype(float)
                               )
        self.trq_min = trq_min(spd=limits_df["Speed"][1:].values.astype(float),
                               data=limits_df["Min Torque"][1:].values.astype(float)
                               )
        self.pwr_max = pwr_max(spd=limits_df["Speed"][1:].values.astype(float),
                               data=limits_df["Max Power"][1:].values.astype(float)
                               )
        self.pwr_min = pwr_min(spd=limits_df["Speed"][1:].values.astype(float),
                               data=limits_df["Min Power"][1:].values.astype(float)
                               )

        # Efficiency power map
        EffPwrMap = namedtuple('EffPwrMap', ['spd', 'pwr', 'data'])
        self.EffPwrMap = EffPwrMap(spd=effpwr_df.iloc[1, 2:].values.astype(float),
                                   pwr=effpwr_df.iloc[2:, 1].values.astype(float),
                                   data=effpwr_df.iloc[2:, 2:].values.astype(float)
                                   )

        # Regeneration map
        RegMap = namedtuple('RegMap', ['spd', 'acc', 'data'])
        self.RegMap = RegMap(spd=reg_df.iloc[2:, 1].values.astype(float),
                             acc=reg_df.iloc[1, 2:].values.astype(float),
                             data=reg_df.iloc[2:, 2:].values.astype(float)
                             )

    def get_trq_max(self, spd):
        if spd > max(self.trq_max.spd):
            spd = max(self.trq_max.spd)
        elif spd < min(self.trq_max.spd):
            spd = min(self.trq_max.spd)
        return np.interp(spd, self.trq_max.spd, self.trq_max.data)

    def get_trq_min(self, spd):
        if spd > max(self.trq_min.spd):
            spd = max(self.trq_min.spd)
        elif spd < min(self.trq_min.spd):
            spd = min(self.trq_min.spd)
        return np.interp(spd, self.trq_min.spd, self.trq_min.data)

    def get_pwr_max(self, spd):
        if spd > max(self.pwr_max.spd):
            spd = max(self.pwr_max.spd)
        elif spd < min(self.pwr_max.spd):
            spd = min(self.pwr_max.spd)
        return np.interp(spd, self.pwr_max.spd, self.pwr_max.data)

    def get_pwr_min(self, spd):
        if spd > max(self.pwr_min.spd):
            spd = max(self.pwr_min.spd)
        elif spd < min(self.pwr_min.spd):
            spd = min(self.pwr_min.spd)
        return np.interp(spd, self.pwr_min.spd, self.pwr_min.data)

    def get_eff_from_pwr(self, spd, pwr):
        if spd > max(self.EffPwrMap.spd):
            spd = max(self.EffPwrMap.spd)
        elif spd < min(self.EffPwrMap.spd):
            spd = min(self.EffPwrMap.spd)
        if pwr > max(self.EffPwrMap.pwr):
            pwr = max(self.EffPwrMap.pwr)
        elif pwr < min(self.EffPwrMap.pwr):
            pwr = min(self.EffPwrMap.pwr)
        f = interp2d(self.EffPwrMap.spd, self.EffPwrMap.pwr, self.EffPwrMap.data, kind='linear')
        return f(spd, pwr).item()

    def get_reg_ratio(self, spd, acc):
        if spd > max(self.RegMap.spd):
            spd = max(self.RegMap.spd)
        elif spd < min(self.RegMap.spd):
            spd = min(self.RegMap.spd)
        if acc > max(self.RegMap.acc):
            acc = max(self.RegMap.acc)
        elif acc < min(self.RegMap.acc):
            acc = min(self.RegMap.acc)
        f = interp2d(self.RegMap.spd, self.RegMap.acc, self.RegMap.data, kind='linear')
        return f(spd, acc).item()