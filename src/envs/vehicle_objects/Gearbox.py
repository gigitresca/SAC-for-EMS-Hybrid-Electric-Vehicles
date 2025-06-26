import pandas as pd
import numpy as np
from collections import namedtuple
from scipy.interpolate import LinearNDInterpolator, interp2d


class Gearbox:
    """
    Object that defines a generic gearbox defined with an efficiency map for each gear
    """

    def __init__(self, param=pd.DataFrame, eff_list=None):
        """
        Constructor method: Initializes all the parameters of the gearbox
        """
        self.inertia_in = param["Inertia in"][1]
        self.inertia_out = param["Inertia out"][1]
        self.gear_idx = param["N gear"][1:].values.astype(float)
        gear_ratio = namedtuple('gear_ratio', ['gear', 'data'])
        self.gear_ratio = gear_ratio(gear=self.gear_idx,
                                     data=param["Gear ratios"][1:].values.astype(float)
                                     )

        # Efficiency maps
        if eff_list is None:
            eff_list = list()
        self.EffTrqInMap = list()
        self.EffTrqOutMap = list()
        EffMap = namedtuple('EffMap', ['spd', 'trq', 'data'])
        eff_tuple = EffMap(spd=eff_list[0].iloc[1, 2:].values.astype(float),
                           trq=eff_list[0].iloc[2:, 1].values.astype(float),
                           data=np.ones(eff_list[0].iloc[2:, 2:].shape)
                           )
        self.EffTrqInMap.append(eff_tuple)  # Initialize a map of efficiency 1 for the idle gear
        self.EffTrqOutMap.append(eff_tuple)  # Initialize a map of efficiency 1 for the idle gear
        for idx in range(len(self.gear_idx)):
            eff_tuple = EffMap(spd=eff_list[idx].iloc[1, 2:].values.astype(float),
                               trq=eff_list[idx].iloc[2:, 1].values.astype(float),
                               data=eff_list[idx].iloc[2:, 2:].values.astype(float)
                               )
            self.EffTrqInMap.append(eff_tuple)
            spd, trq, out_map = self.from_in_to_out_map(self.EffTrqInMap[idx].spd, self.EffTrqInMap[idx].trq,
                                                        self.gear_ratio.data[idx], self.EffTrqInMap[idx].data
                                                        )
            eff_tuple = EffMap(spd=spd, trq=trq, data=out_map)
            self.EffTrqOutMap.append(eff_tuple)

        # Add 0 gear to the gear_ratio namedtuple
        self.gear_idx = np.concatenate((np.array([0]), self.gear_idx))
        self.gear_ratio = gear_ratio(gear=self.gear_idx,
                                     data=np.concatenate((np.array([0]), self.gear_ratio.data))
                                     )

    def get_gear_ratio(self, gear):
        if gear > max(self.gear_ratio.gear):
            gear = max(self.gear_ratio.gear)
        elif gear < min(self.gear_ratio.gear):
            gear = min(self.gear_ratio.gear)
        return np.interp(gear, self.gear_ratio.gear, self.gear_ratio.data)

    def get_eff_from_in(self, spd, trq, gear):
        if spd > max(self.EffTrqInMap[gear].spd):
            spd = max(self.EffTrqInMap[gear].spd)
        elif spd < min(self.EffTrqInMap[gear].spd):
            spd = min(self.EffTrqInMap[gear].spd)
        if trq > max(self.EffTrqInMap[gear].trq):
            trq = max(self.EffTrqInMap[gear].trq)
        elif trq < min(self.EffTrqInMap[gear].trq):
            trq = min(self.EffTrqInMap[gear].trq)
        f = interp2d(self.EffTrqInMap[gear].spd, self.EffTrqInMap[gear].trq, self.EffTrqInMap[gear].data, kind='linear')
        return f(spd, trq).item()

    def get_eff_from_out(self, spd, trq, gear):
        if spd > max(self.EffTrqOutMap[gear].spd):
            spd = max(self.EffTrqOutMap[gear].spd)
        elif spd < min(self.EffTrqOutMap[gear].spd):
            spd = min(self.EffTrqOutMap[gear].spd)
        if trq > max(self.EffTrqOutMap[gear].trq):
            trq = max(self.EffTrqOutMap[gear].trq)
        elif trq < min(self.EffTrqOutMap[gear].trq):
            trq = min(self.EffTrqOutMap[gear].trq)
        f = interp2d(self.EffTrqOutMap[gear].spd, self.EffTrqOutMap[gear].trq, self.EffTrqOutMap[gear].data, kind='linear')
        return f(spd, trq).item()

    @staticmethod
    def from_out_to_in_map(spd, y, gear_ratio, out_map):
        spd_out_mtx, y_out_mtx = np.meshgrid(spd, y)
        spd_in_mtx = spd_out_mtx * gear_ratio
        y_in_mtx = y_out_mtx / gear_ratio / out_map
        f = LinearNDInterpolator(list(zip(spd_in_mtx.flatten(), y_in_mtx.flatten())), out_map.flatten())
        spd = spd * gear_ratio
        y = np.linspace(0, y[-1] / gear_ratio / out_map[-1, -1], len(y))
        spd_in_mtx, y_in_mtx = np.meshgrid(spd, y)
        return spd, y, f(spd_in_mtx, y_in_mtx)

    @staticmethod
    def from_in_to_out_map(spd, y, gear_ratio, in_map):
        spd_in_mtx, y_in_mtx = np.meshgrid(spd, y)
        spd_out_mtx = spd_in_mtx / gear_ratio
        y_out_mtx = y_in_mtx * gear_ratio * in_map
        f = LinearNDInterpolator(list(zip(spd_out_mtx.flatten(), y_out_mtx.flatten())), in_map.flatten(), )
        spd = spd / gear_ratio
        y = np.linspace(0, y[-1] * gear_ratio * in_map[-1, -1], len(y))
        spd_out_mtx, y_out_mtx = np.meshgrid(spd, y)
        return spd, y, f(spd_out_mtx, y_out_mtx)

    @staticmethod
    def from_trq_to_pwr_map(spd, trq, pwr, trq_map):
        spd_mtx, trq_mtx = np.meshgrid(spd, trq)
        pwr_mtx = spd_mtx * trq_mtx
        f = LinearNDInterpolator(list(zip(spd_mtx.flatten(), pwr_mtx.flatten())), trq_map.flatten())
        spd_mtx, pwr_mtx = np.meshgrid(spd, pwr)
        return f(spd_mtx, pwr_mtx)
