import pandas as pd
import pickle

from src.envs.vehicle_objects.Engine import Engine
from src.envs.vehicle_objects.ElectricMotor import ElectricMotor
from src.envs.vehicle_objects.Battery import Battery
from src.envs.vehicle_objects.FinalDrive import FinalDrive
from src.envs.vehicle_objects.Gearbox import Gearbox
from src.envs.vehicle_objects.Wheels import Wheels
from src.envs.vehicle_objects.Vehicle import Vehicle
from src.envs.vehicle_objects.DrivingCycle import DrivingCycle


def gen_vehicle_objects(xlsx_vehicle):
    """
    Function used to create the objects for describing the main components of a vehicle given an Excel file
    """
    # Engine parameters
    ice_param = pd.read_excel(xlsx_vehicle, sheet_name="Engine", engine='openpyxl')
    ice_limits_df = pd.read_excel(xlsx_vehicle, sheet_name="Engine-Limits", engine='openpyxl')
    ice_fueltrq_df = pd.read_excel(xlsx_vehicle, sheet_name="Engine-FuelTrqMap", engine='openpyxl')
    ice_fuelpwr_df = pd.read_excel(xlsx_vehicle, sheet_name="Engine-FuelPwrMap", engine='openpyxl')
    ice_bsfctrq_df = pd.read_excel(xlsx_vehicle, sheet_name="Engine-BsfcTrqMap", engine='openpyxl')
    ice_bsfcpwr_df = pd.read_excel(xlsx_vehicle, sheet_name="Engine-BsfcPwrMap", engine='openpyxl')
    ice_efftrq_df = pd.read_excel(xlsx_vehicle, sheet_name="Engine-EffTrqMap", engine='openpyxl')
    ice = Engine(ice_param, ice_limits_df, ice_fueltrq_df, ice_fuelpwr_df, ice_bsfctrq_df, ice_bsfcpwr_df,
                 ice_efftrq_df)

    # Electric motor parameters
    em_param = pd.read_excel(xlsx_vehicle, sheet_name="Electric motor", engine='openpyxl')
    em_limits_df = pd.read_excel(xlsx_vehicle, sheet_name="Electric motor-Limits", engine='openpyxl')
    em_eff_pwr_ds = pd.read_excel(xlsx_vehicle, sheet_name="Electric motor-EffPwrMap", engine='openpyxl')
    em_reg_ds = pd.read_excel(xlsx_vehicle, sheet_name="Electric motor-RegMap", engine='openpyxl')
    em = ElectricMotor(em_param, em_limits_df, em_eff_pwr_ds, em_reg_ds)

    # Battery parameters
    ess_param = pd.read_excel(xlsx_vehicle, sheet_name="Battery", engine='openpyxl')
    ess_ocv_rint_df = pd.read_excel(xlsx_vehicle, sheet_name="Battery-VoC-Rint", engine='openpyxl')
    ess_limits_df = pd.read_excel(xlsx_vehicle, sheet_name="Battery-Limits", engine='openpyxl')
    ess = Battery(ess_param, ess_ocv_rint_df, ess_limits_df, soc_init=0.5, soc_trg=0.5)

    # Gearbox parameters
    gb_param = pd.read_excel(xlsx_vehicle, sheet_name="Gearbox", engine='openpyxl')
    gb_eff_list = list()
    gb_eff_list.append(pd.read_excel(xlsx_vehicle, sheet_name="Gearbox-EffMap1"))
    gb_eff_list.append(pd.read_excel(xlsx_vehicle, sheet_name="Gearbox-EffMap2"))
    gb_eff_list.append(pd.read_excel(xlsx_vehicle, sheet_name="Gearbox-EffMap3"))
    gb_eff_list.append(pd.read_excel(xlsx_vehicle, sheet_name="Gearbox-EffMap4"))
    gb_eff_list.append(pd.read_excel(xlsx_vehicle, sheet_name="Gearbox-EffMap5"))
    gb_eff_list.append(pd.read_excel(xlsx_vehicle, sheet_name="Gearbox-EffMap6"))
    gb_eff_list.append(pd.read_excel(xlsx_vehicle, sheet_name="Gearbox-EffMap7"))
    gb_eff_list.append(pd.read_excel(xlsx_vehicle, sheet_name="Gearbox-EffMap8"))
    gb_eff_list.append(pd.read_excel(xlsx_vehicle, sheet_name="Gearbox-EffMap9"))
    gb = Gearbox(gb_param, gb_eff_list)

    # Final drive parameters
    fd_param = pd.read_excel(xlsx_vehicle, sheet_name="Final drive", engine='openpyxl')
    fd = FinalDrive(fd_param)

    # Wheels parameters
    wh_param = pd.read_excel(xlsx_vehicle, sheet_name="Wheels", engine='openpyxl')
    wh = Wheels(wh_param)

    # Vehicle parameters
    veh_param = pd.read_excel(xlsx_vehicle, sheet_name="Vehicle", engine='openpyxl')
    veh = Vehicle(veh_param)
    return ice, em, ess, gb, fd, wh, veh


def gen_driving_cycles_objects(xlsx_cycles):
    """
    Function used to create the objects for each driving cycles in the input Excel file
    """
    xlsx_file_cycles = pd.ExcelFile(xlsx_cycles)
    sheets = xlsx_file_cycles.sheet_names
    driving_cycles = dict()
    for name in sheets:
        cycle_df = pd.read_excel(xlsx_cycles, sheet_name=name, header=2, engine='openpyxl')
        cycle = DrivingCycle(cycle_df, name)
        if cycle.time_step != 1:
            cycle.resample(time_step_new=1)
        driving_cycles[name] = cycle
    # Saving driving_cycles object
    with open("data/driving_cycles.pkl", "wb") as file:
        pickle.dump(driving_cycles, file)
    return driving_cycles
