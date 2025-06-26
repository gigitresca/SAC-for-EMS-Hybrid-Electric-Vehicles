import pandas as pd


class Wheels:
    """
    Object that defines the main parameters for the description of the wheels
    """

    def __init__(self, param=pd.DataFrame):
        """
        Constructor method: Initializes all the parameters of the wheels
        """
        self.radius = param["Radius"][1]
        self.inertia_f = param["Inertia F"][1]  # 2 front wheels considered
        self.inertia_r = param["Inertia R"][1]  # 2 rear wheels considered
