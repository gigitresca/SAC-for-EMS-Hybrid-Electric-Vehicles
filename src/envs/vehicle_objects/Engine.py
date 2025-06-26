import math
import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.interpolate import interp2d, LinearNDInterpolator


class Engine:
    """
    Object that defines a generic Internal Combustion Engine described by
    fuel consumption map and bsfc map
    """

    def __init__(self, param=pd.DataFrame, limits_df=pd.DataFrame, fueltrq_df=pd.DataFrame, fuelpwr_df=pd.DataFrame,
                 bsfctrq_df=pd.DataFrame, bsfcpwr_df=pd.DataFrame, efftrq_df=pd.DataFrame):
        """
        Constructor method: Initializes all the parameters of the engine
        """
        # Main parameters
        self.fuel_heating_val = param["Fuel LHV"][1]
        self.fuel_density_val = param["Fuel density value"][1]
        self.spd_idle = param["Idle speed"][1]
        self.spd_max = param["Max speed"][1]
        self.bore = param["Bore"][1]
        self.stroke = param["Stroke"][1]
        self.n_cyl = param["N cylinder"][1]
        self.inertia = param["Inertia"][1]
        if math.isnan(param["Displecement"][1]):
            self.disp = (self.bore ** 2 * math.pi / 4 * self.stroke * self.n_cyl) / 1000
        else:
            self.disp = param["Displecement"][1]
        self.co2_mol_mass = param["CO2 molar mass"][1]
        self.fuel_mol_mass = param["Fuel molar mass"][1]
        self.cut_off_lim = param["Cut-off limit"][1]

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

        # Fuel torque map
        FuelTrqMap = namedtuple('FuelTrqMap', ['spd', 'trq', 'data'])
        self.FuelTrqMap = FuelTrqMap(spd=fueltrq_df.iloc[1, 2:].values.astype(float),
                                     trq=fueltrq_df.iloc[2:, 1].values.astype(float),
                                     data=fueltrq_df.iloc[2:, 2:].values.astype(float)
                                     )

        # Fuel power map
        FuelPwrMap = namedtuple('FuelPwrMap', ['spd', 'pwr', 'data'])
        self.FuelPwrMap = FuelPwrMap(spd=fuelpwr_df.iloc[1, 2:].values.astype(float),
                                     pwr=fuelpwr_df.iloc[2:, 1].values.astype(float),
                                     data=fuelpwr_df.iloc[2:, 2:].values.astype(float)
                                     )

        # BSFC torque map
        BsfcTrqMap = namedtuple('BsfcTrqMap', ['spd', 'trq', 'data'])
        self.BsfcTrqMap = BsfcTrqMap(spd=bsfctrq_df.iloc[1, 2:].values.astype(float),
                                     trq=bsfctrq_df.iloc[2:, 1].values.astype(float),
                                     data=bsfctrq_df.iloc[2:, 2:].values.astype(float)
                                     )

        # BSFC power map
        BsfcPwrMap = namedtuple('BsfcPwrMap', ['spd', 'pwr', 'data'])
        self.BsfcPwrMap = BsfcPwrMap(spd=bsfcpwr_df.iloc[1, 2:].values.astype(float),
                                     pwr=bsfcpwr_df.iloc[2:, 1].values.astype(float),
                                     data=bsfcpwr_df.iloc[2:, 2:].values.astype(float)
                                     )

        # Efficiency torque map
        EffTrqMap = namedtuple('EffTrqMap', ['spd', 'trq', 'data'])
        self.EffTrqMap = EffTrqMap(spd=efftrq_df.iloc[1, 2:].values.astype(float),
                                   trq=efftrq_df.iloc[2:, 1].values.astype(float),
                                   data=efftrq_df.iloc[2:, 2].values.astype(float)
                                   )

        # Optimal Operating Line
        trq_minbsfc = namedtuple('trq_minbsfc', ['spd', 'data'])
        pwr_minbsfc = namedtuple('pwr_minbsfc', ['spd', 'data'])
        spd, data = self.get_trq_ool()
        self.trq_minbsfc = trq_minbsfc(spd=spd, data=data)
        spd, data = self.get_pwr_ool()
        self.pwr_minbsfc = pwr_minbsfc(spd=spd, data=data)

    def get_trq_ool(self):
        """
        Method to compute the torque values that minimizes the BSFC on the torque map
        """
        trq_fl_idx = np.interp(self.BsfcTrqMap.spd, self.trq_max.spd, self.trq_max.data)
        spd_mtx, trq_mtx = np.meshgrid(self.BsfcTrqMap.spd, self.BsfcTrqMap.trq)
        bsfc_mtx = self.BsfcTrqMap.data
        bsfc_mtx[trq_mtx > trq_fl_idx] = 10 ** 6
        minbsfc_idx = np.argmin(bsfc_mtx, axis=0)
        spd_minbsfc = self.BsfcTrqMap.spd
        trq_minbsfc = trq_mtx[minbsfc_idx, np.arange(trq_mtx.shape[1])]
        return spd_minbsfc, trq_minbsfc

    def get_pwr_ool(self):
        """
        Method to compute the torque values that minimizes the BSFC on the torque map
        """
        pwr_fl_idx = np.interp(self.BsfcPwrMap.spd, self.pwr_max.spd, self.pwr_max.data)
        spd_mtx, pwr_mtx = np.meshgrid(self.BsfcPwrMap.spd, self.BsfcPwrMap.pwr)
        bsfc_mtx = self.BsfcPwrMap.data
        bsfc_mtx[pwr_mtx > pwr_fl_idx] = 10 ** 6
        minbsfc_idx = np.argmin(bsfc_mtx, axis=0)
        spd_minbsfc = self.BsfcPwrMap.spd
        pwr_minbsfc = pwr_mtx[minbsfc_idx, np.arange(pwr_mtx.shape[1])]
        pwr_minbsfc[0] = 0
        return spd_minbsfc, pwr_minbsfc

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
            spd = min(self.trq_min.spd)
        return np.interp(spd, self.pwr_min.spd, self.pwr_min.data)

    def get_fuel_from_trq(self, spd, trq):
        if spd > max(self.FuelTrqMap.spd):
            spd = max(self.FuelTrqMap.spd)
        elif spd < min(self.FuelTrqMap.spd):
            spd = min(self.FuelTrqMap.spd)
        if trq > max(self.FuelTrqMap.trq):
            trq = max(self.FuelTrqMap.trq)
        elif trq < min(self.FuelTrqMap.trq):
            trq = min(self.FuelTrqMap.trq)
        f = interp2d(self.FuelTrqMap.spd, self.FuelTrqMap.trq, self.FuelTrqMap.data, kind='linear')
        return f(spd, trq).item()

    def get_fuel_from_pwr(self, spd, pwr):
        if spd > max(self.FuelPwrMap.spd):
            spd = max(self.FuelPwrMap.spd)
        elif spd < min(self.FuelPwrMap.spd):
            spd = min(self.FuelPwrMap.spd)
        if pwr > max(self.FuelPwrMap.pwr):
            pwr = max(self.FuelPwrMap.pwr)
        elif pwr < min(self.FuelPwrMap.pwr):
            pwr = min(self.FuelPwrMap.pwr)
        f = interp2d(self.FuelPwrMap.spd, self.FuelPwrMap.pwr, self.FuelPwrMap.data, kind='linear')
        return f(spd, pwr).item()

    def get_bsfc_from_trq(self, spd, trq):
        if spd > max(self.BsfcTrqMap.spd):
            spd = max(self.BsfcTrqMap.spd)
        elif spd < min(self.BsfcTrqMap.spd):
            spd = min(self.BsfcTrqMap.spd)
        if trq > max(self.BsfcTrqMap.trq):
            trq = max(self.BsfcTrqMap.trq)
        elif trq < min(self.BsfcTrqMap.trq):
            trq = min(self.BsfcTrqMap.trq)
        f = interp2d(self.BsfcTrqMap.spd, self.BsfcTrqMap.trq, self.BsfcTrqMap.data, kind='linear')
        return f(spd, trq).item()

    def get_bsfc_from_pwr(self, spd, pwr):
        if spd > max(self.BsfcPwrMap.spd):
            spd = max(self.BsfcPwrMap.spd)
        elif spd < min(self.BsfcPwrMap.spd):
            spd = min(self.BsfcPwrMap.spd)
        if pwr > max(self.BsfcPwrMap.pwr):
            pwr = max(self.BsfcPwrMap.pwr)
        elif pwr < min(self.BsfcPwrMap.pwr):
            pwr = min(self.BsfcPwrMap.pwr)
        f = interp2d(self.BsfcPwrMap.spd, self.BsfcPwrMap.pwr, self.BsfcPwrMap.data, kind='linear')
        return f(spd, pwr).item()

    def get_eff_from_trq(self, spd, trq):
        if spd > max(self.EffTrqMap.spd):
            spd = max(self.EffTrqMap.spd)
        elif spd < min(self.EffTrqMap.spd):
            spd = min(self.EffTrqMap.spd)
        if trq > max(self.EffTrqMap.trq):
            trq = max(self.EffTrqMap.trq)
        elif trq < min(self.EffTrqMap.trq):
            trq = min(self.EffTrqMap.trq)
        f = interp2d(self.EffTrqMap.spd, self.EffTrqMap.trq, self.EffTrqMap.data, kind='linear')
        return f(spd, trq).item()

    def bmep_from_trq(self, trq, i=2):
        return (trq * 2 * np.pi * i) / (self.disp * 1e-6) * 1e-5

    def bmep_from_pwr(self, spd, pwr, i=2):
        return (60 * i * pwr) / ((spd * 30 / np.pi) * (self.disp * 1e-6)) * 1e-5

    def pwr_from_bmep(self, spd, bmep, i=2):
        return ((bmep * 1e5) * spd * (30 / np.pi) * (self.disp * 1e-6)) / (60 * i)

    @staticmethod
    def from_trq_to_pwr_map(spd, trq, pwr, trq_map):
        spd_mtx, trq_mtx = np.meshgrid(spd, trq)
        pwr_mtx = spd_mtx * trq_mtx
        f = LinearNDInterpolator(list(zip(spd_mtx.flatten(), pwr_mtx.flatten())), trq_map.flatten())
        spd_mtx, pwr_mtx = np.meshgrid(spd, pwr)
        return f(spd_mtx, pwr_mtx)
