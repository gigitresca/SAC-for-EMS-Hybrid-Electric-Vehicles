import pandas as pd
from collections import namedtuple
import numpy as np


class Battery:
    """
    Object that defines a generic Battery described by
    OCV and internal resistance maps
    """

    def __init__(self, param=pd.DataFrame, ocv_rint_df=pd.DataFrame, batt_limits_df=pd.DataFrame, soc_init=0.5, soc_trg=0.5):
        """
        Constructor method: Initializes all the parameters of the battery
        """
        self.num_cell_series = param["N cell series"][1]
        self.num_module_parallel = param["N module parallel"][1]
        self.num_cells = self.num_cell_series * self.num_module_parallel
        self.volt_nom = param["Nominal voltage"][1]
        self.volt_max = param["Max voltage"][1]
        self.volt_min = param["Min voltage"][1]
        self.mass = param["Mass"][1]
        self.mass_cell = param["Mass cell"][1]
        self.soc_min = param["Min SoC"][1]
        self.soc_max = param["Max SoC"][1]
        self.soc_high = param["Low SoC"][1]
        self.soc_low = param["High SoC"][1]
        self.cell_cap = param["Cell capacity"][1]
        self.cell_curr_max_chg = param["Cell max current charge"][1]
        self.cell_curr_max_dis = param["Cell max current discharge"][1]
        self.access = param["Electric accessory"][1]

        # Open Circuit Voltage
        cell_ocv = namedtuple('cell_ocv', ['soc', 'data'])
        self.cell_ocv = cell_ocv(soc=ocv_rint_df["State of charge"][1:].values.astype(float),
                                 data=ocv_rint_df["Cell open circuit voltage"][1:].values.astype(float)
                                 )

        # Internal resistance
        r_int_chg = namedtuple('r_int_chg', ['soc', 'data'])
        r_int_dis = namedtuple('r_int_dis', ['soc', 'data'])
        self.r_int_chg = r_int_chg(soc=ocv_rint_df["State of charge"][1:].values.astype(float),
                                   data=ocv_rint_df["Cell internal resistance charge"][1:].values.astype(float)
                                   )
        self.r_int_dis = r_int_dis(soc=ocv_rint_df["State of charge"][1:].values.astype(float),
                                   data=ocv_rint_df["Cell internal resistance discharge"][1:].values.astype(float)
                                   )

        # Battery peak power at 20°C
        pwr_max_dis = namedtuple('pwr_max_dis', ['soc', 'data'])
        pwr_max_chg = namedtuple('pwr_max_chg', ['soc', 'data'])
        self.pwr_max_dis = pwr_max_dis(soc=batt_limits_df["State of charge"][1:].values.astype(float),
                                       data=batt_limits_df["Battery peak discharge power"][1:].values.astype(float)
                                       )
        self.pwr_max_chg = pwr_max_chg(soc=batt_limits_df["State of charge"][1:].values.astype(float),
                                       data=batt_limits_df["Battery peak charge power"][1:].values.astype(float)
                                       )

        # Discharge strategy
        self.soc_init = soc_init
        self.soc_trg = soc_trg

    def get_ocv(self, soc):
        if soc > max(self.cell_ocv.soc):
            soc = max(self.cell_ocv.soc)
        elif soc < min(self.cell_ocv.soc):
            soc = min(self.cell_ocv.soc)
        return np.interp(soc, self.cell_ocv.soc, self.cell_ocv.data)

    def get_rint_chg(self, soc):
        if soc > max(self.r_int_chg.soc):
            soc = max(self.r_int_chg.soc)
        elif soc < min(self.r_int_chg.soc):
            soc = min(self.r_int_chg.soc)
        return np.interp(soc, self.r_int_chg.soc, self.r_int_chg.data)

    def get_rint_dis(self, soc):
        if soc > max(self.r_int_dis.soc):
            soc = max(self.r_int_dis.soc)
        elif soc < min(self.r_int_dis.soc):
            soc = min(self.r_int_dis.soc)
        return np.interp(soc, self.r_int_dis.soc, self.r_int_dis.data)

    def get_pwrmax_dis(self, soc):
        if soc > max(self.pwr_max_dis.soc):
            soc = max(self.pwr_max_dis.soc)
        elif soc < min(self.pwr_max_dis.soc):
            soc = min(self.pwr_max_dis.soc)
        return np.interp(soc, self.pwr_max_dis.soc, self.pwr_max_dis.data)

    def get_pwrmax_chg(self, soc):
        if soc > max(self.pwr_max_chg.soc):
            soc = max(self.pwr_max_chg.soc)
        elif soc < min(self.pwr_max_chg.soc):
            soc = min(self.pwr_max_chg.soc)
        return np.interp(soc, self.pwr_max_chg.soc, self.pwr_max_chg.data)
