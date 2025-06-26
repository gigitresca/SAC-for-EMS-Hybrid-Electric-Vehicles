import pandas as pd


class FinalDrive:
    """
    Object that defines the main parameters for the final drive
    """

    def __init__(self, param=pd.DataFrame):
        """
        Constructor method: Initializes all the parameters of the final drive
        """
        self.ratio = param["Ratio"][1]
        self.eff_max = param["Efficiency max"][1]
        self.inertia = param["Inertia"][1]
