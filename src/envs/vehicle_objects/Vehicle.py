import pandas as pd
import math


class Vehicle:
    """
    Object that defines the main parameters for the description of the longitudinal dynamic of a vehicle
    """

    def __init__(self, param=pd.DataFrame):
        """
        Constructor method: Initializes all the parameters of the vehicle
        """
        self.axal_base = param["Axal base"][1]
        self.cg_height = param["CG height"][1]
        self.cargo_mass = param["Cargo mass"][1]
        self.body_mass = param["Body mass"][1]
        self.mass = param["Mass"][1]
        if math.isnan(self.mass):
            self.mass = self.body_mass + self.cargo_mass
        self.F0 = param["F0"][1]
        self.F1 = param["F1"][1]
        self.F2 = param["F2"][1]
