import numpy as np
import random
import gymnasium as gym
import pdb
import math
import random

from gymnasium import spaces
from gymnasium.envs.registration import register

from collections import namedtuple


class VehicleBackwardKinEnv(gym.Env):
    """
    Custom Environment that follows gym interface: A P2 diesel plug-in hybrid electric vehicle is modelled
    following a backward kinematic approach
    """

    def __init__(self, driving_cycles=None, veh=None, wh=None, fd=None, gb=None, ice=None, em=None, ess=None,
                 env_time_step=1):
        """
        Constructor method: Initializes all the components of the car, giving as input the objects representing the
        vehicle components
        """
        super().__init__()
        self.driving_cycles = list(driving_cycles.values())
        self.time_step = env_time_step
        self.wh = wh
        self.ess = ess
        self.ice = ice
        self.em = em
        self.fd = fd
        self.gb = gb
        self.veh = veh
        self.sim_input = dict(cycle=[],  # Input dictionary initialization
                              time=[],
                              time_perc=[],
                              veh_spd=[],
                              veh_acc=[],
                              veh_dist=[],
                              veh_dist_perc=[],
                              gear=[],
                              grade=[]
                              )
        self.sim_output = dict(veh_tot_force=np.array([]).astype(float),  # Output dictionary initialization
                               fd_spd=np.array([]).astype(float),
                               fd_pwr=np.array([]).astype(float),
                               gb_spd_in=np.array([]).astype(float),
                               gb_pwr_in=np.array([]).astype(float),
                               em_spd=np.array([]).astype(float),
                               em_pwr=np.array([]).astype(float),
                               em_pwr_ele=np.array([]).astype(float),
                               ice_state=np.array([]).astype(float),
                               ice_spd=np.array([]).astype(float),
                               ice_pwr=np.array([]).astype(float),
                               ice_fuel=np.array([]).astype(float),
                               batt_pwr=np.array([]).astype(float),
                               batt_soc=np.array([]).astype(float),
                               reward=np.array([]).astype(float),
                               unfeasible=np.array([]).astype(bool),
                               action=np.array([]).astype(float),
                               )
        self.obs_name = ['veh_spd', 'veh_acc', 'em_spd', 'batt_soc', 'veh_dist_perc']
        self.act_name = 'ice_trq_max_perc'

        '''The action is the normalized engine torque spacing continuously from 0 to 1'''
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float)
        '''The observations are 5 normalized continuous variables:
            - vehicle speed
            - powertrain power demand (at the gearbox inlet)
            - powertrain speed (at the gearbox inlet)
            - battery State of Charge
            - traveled distance over the total distance'''
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=float)

        # Reward definition
        veh_param_tuple = namedtuple('veh_param',
                                     ['fuel_heating_val', 'cell_cap', 'num_cell_series', 'volt_nom', 'soc_trg',
                                      'time_step'])
        self.reward = Reward(reward_type='dense',
                             k1=0,
                             k2=-100,
                             k3=-200,
                             unfeaspenalty=0,
                             veh_param=veh_param_tuple(fuel_heating_val=ice.fuel_heating_val,
                                                       cell_cap=ess.cell_cap,
                                                       num_cell_series=ess.num_cell_series,
                                                       volt_nom=ess.volt_nom,
                                                       soc_trg=ess.soc_trg,
                                                       time_step=self.time_step
                                                       )
                             )

        # Normalization object initialization
        # NOTE: if the states changes, also the max and min states vectors should be updated in the normalization object
        self.normalization = Normalization(up_value=1, low_value=0)

        # Driving cycle resample
        if env_time_step != 1:
            for i, cycle in enumerate(self.driving_cycles):
                cycle.resample(time_step_new=env_time_step)
                self.driving_cycles[i] = cycle

    def set_input(self, tr_step):
        """
        Set the simulation input into the sim_input dictionary
        """
        cycle = self.sim_input['cycle']
        self.sim_input['time'] = cycle.time_simu[tr_step]
        self.sim_input['time_perc'] = self.sim_input['time'] / cycle.time_simu[-1]
        self.sim_input['veh_spd'] = cycle.veh_spd[tr_step]
        self.sim_input['veh_acc'] = cycle.veh_acc[tr_step]
        self.sim_input['veh_dist'] = cycle.veh_dist[tr_step]
        self.sim_input['veh_dist_perc'] = self.sim_input['veh_dist'] / cycle.veh_dist[-1]
        self.sim_input['gear'] = cycle.gear[tr_step]
        self.sim_input['grade'] = cycle.grade[tr_step]

    def reset_output(self):
        """
        Method to reset the output vocabulary when a new episode starts
        """
        self.sim_output['veh_tot_force'] = np.array([]).astype(float)  # Output dictionary initialization
        self.sim_output['fd_spd'] = np.array([]).astype(float)
        self.sim_output['fd_pwr'] = np.array([]).astype(float)
        self.sim_output['gb_spd_in'] = np.array([]).astype(float)
        self.sim_output['gb_pwr_in'] = np.array([]).astype(float)
        self.sim_output['em_spd'] = np.array([]).astype(float)
        self.sim_output['em_pwr'] = np.array([]).astype(float)
        self.sim_output['em_pwr_ele'] = np.array([]).astype(float)
        self.sim_output['ice_state'] = np.array([]).astype(float)
        self.sim_output['ice_spd'] = np.array([]).astype(float)
        self.sim_output['ice_pwr'] = np.array([]).astype(float)
        self.sim_output['ice_fuel'] = np.array([]).astype(float)
        self.sim_output['batt_pwr'] = np.array([]).astype(float)
        self.sim_output['batt_soc'] = np.array([]).astype(float)
        self.sim_output['reward'] = np.array([]).astype(float)
        self.sim_output['unfeasible'] = np.array([]).astype(bool)
        self.sim_output['action'] = np.array([]).astype(float)

    def update_states(self):
        """
        Method to automatically update the states given the obs name
        """
        states = np.zeros(len(self.obs_name))

        # Get last output value
        last_gb_spd_in = self.sim_output['gb_spd_in'][-1]
        last_gb_pwr_in = self.sim_output['gb_pwr_in'][-1]
        last_em_spd = self.sim_output['em_spd'][-1]
        last_ice_spd = self.sim_output['ice_spd'][-1]
        last_ice_state = self.sim_output['ice_state'][-1]
        last_batt_soc = self.sim_output['batt_soc'][-1]

        # Candidate states vocabulary creation
        states_map = {'veh_spd': self.sim_input['veh_spd'],
                      'veh_acc': self.sim_input['veh_acc'],
                      'veh_dist_perc': self.sim_input['veh_dist_perc'],
                      'time_perc': self.sim_input['time_perc'],
                      'gb_in_spd': last_gb_spd_in,
                      'gb_in_pwr': last_gb_pwr_in,
                      'em_spd': last_em_spd,
                      'ice_spd': last_ice_spd,
                      'ice_state': last_ice_state,
                      'batt_soc': last_batt_soc
                      }
        # Get the states vector
        for i, name in enumerate(self.obs_name):
            states[i] = states_map[name]
        return states

    def get_road_force(self, veh_spd, veh_acc, gear, ice_state):
        """
        Road force computation
        """
        veh_eq_mass = (self.veh.mass + (2 * self.wh.inertia_f + 2 * self.wh.inertia_r) *
                       1 / (self.wh.radius ** 2) + ice_state * self.ice.inertia *
                       (self.gb.get_gear_ratio(gear) ** 2) * self.fd.ratio **
                       2 / (self.wh.radius ** 2) + self.em.inertia *
                       (self.gb.get_gear_ratio(gear) ** 2) * self.fd.ratio ** 2 *
                       self.em.gear_ratio ** 2 / (self.wh.radius ** 2)
                       )
        veh_res_force = (self.veh.F0 + self.veh.F1 * (veh_spd * 3.6) +
                         self.veh.F2 * (veh_spd * 3.6) ** 2
                         )
        veh_tot_force = veh_res_force + veh_eq_mass * veh_acc
        veh_pwr = veh_tot_force * veh_spd
        return veh_tot_force, veh_pwr

    def solve_wheels(self, veh_spd, veh_tot_force):
        """
        Compute wheels speed and power
        """
        wh_spd = veh_spd / self.wh.radius
        wh_trq = veh_tot_force * self.wh.radius
        wh_pwr = wh_trq * wh_spd
        return wh_spd, wh_trq, wh_pwr

    def solve_final_drive(self, wh_spd, wh_trq):
        """
        Compute final drive speed and power
        """
        fd_spd = wh_spd * self.fd.ratio
        fd_trq = wh_trq / self.fd.ratio
        fd_pwr = fd_trq * fd_spd
        return fd_spd, fd_trq, fd_pwr

    def solve_gearbox(self, fd_spd, fd_trq, fd_pwr, gear):
        """
        Compute gearbox speeds and power demand
        """
        gb_eff = self.gb.get_eff_from_out(fd_spd, fd_trq, gear)
        gb_pwr_in = (fd_pwr * gb_eff ** (np.sign(-fd_pwr)))
        gb_spd_in = fd_spd * self.gb.get_gear_ratio(gear)
        gb_trq_in = gb_pwr_in / max(0.0001, gb_spd_in)
        if math.isnan(gb_trq_in):
            gb_trq_in = 0
        return gb_spd_in, gb_trq_in, gb_pwr_in

    def solve_powersplit(self, action, gb_pwr_in, gb_spd_in, ice_state, veh_spd, veh_acc):
        """
        Compute all the variables related to the power split
        """
        ice_spd = max(self.ice.spd_idle, gb_spd_in) * ice_state
        em_spd = gb_spd_in * self.em.gear_ratio

        # Physical limitations
        ice_pwr_max = self.ice.get_pwr_max(ice_spd)
        ice_pwr_min = self.ice.get_pwr_min(ice_spd)
        em_pwr_max = self.em.get_pwr_max(em_spd)
        em_pwr_min = self.em.get_pwr_min(em_spd)
        ice_trq_max = self.ice.get_trq_max(ice_spd)

        """
            The power at the gearbox inlet represents the power demand to the powertrain
            Compute the power-split
        """
        pwt_pwr_dmd = gb_pwr_in
        ice_trq = action * ice_trq_max
        ice_pwr = ice_trq * ice_spd
        unfeasible = 0

        if ice_pwr > ice_pwr_max:
            ice_pwr = ice_pwr_max
            unfeasible = 1  #ice physical limit

        if gb_pwr_in > 0:
            em_pwr = pwt_pwr_dmd - ice_pwr
            brk_pwr_mech = 0
            em_pwr_reg = 0
        elif gb_pwr_in <= 0 and ice_state == 1:
            ice_pwr = ice_pwr_min
            pwt_pwr_dmd = gb_pwr_in - ice_pwr
            em_pwr = self.em.get_reg_ratio(veh_spd, veh_acc) * pwt_pwr_dmd
            em_pwr_reg = em_pwr
            brk_pwr_mech = pwt_pwr_dmd - em_pwr
            unfeasible = 1  #condition not efficient
        else:
            ice_pwr = 0
            pwt_pwr_dmd = gb_pwr_in
            em_pwr = self.em.get_reg_ratio(veh_spd, veh_acc) * pwt_pwr_dmd
            em_pwr_reg = em_pwr
            brk_pwr_mech = pwt_pwr_dmd - em_pwr

        if em_pwr > em_pwr_max:
            em_pwr = em_pwr_max
            ice_pwr = pwt_pwr_dmd - em_pwr
            unfeasible = 1  #em physical limit
        elif em_pwr < em_pwr_min:
            em_pwr = em_pwr_min
            em_pwr_reg = em_pwr
            ice_pwr = pwt_pwr_dmd - em_pwr
            unfeasible = 1  #em physical limit

        return ice_spd, ice_pwr, ice_trq, em_spd, em_pwr, em_pwr_reg, brk_pwr_mech, unfeasible




    def solve_engine(self, ice_spd, ice_pwr, ice_state):
        """
        Compute the fuel consumption and CO2 emissions
        """

        ice_fuel_rate = self.ice.get_fuel_from_pwr(ice_spd, ice_pwr)    # [kg/s]

        if ice_state == 0:
            ice_fuel_rate = 0
        elif ice_state == 1 and ice_spd > self.ice.cut_off_lim and ice_pwr < 0:
            ice_fuel_rate = 0

        ice_co2_rate = ice_fuel_rate * self.ice.co2_mol_mass / self.ice.fuel_mol_mass

        return ice_fuel_rate, ice_co2_rate

    def solve_electrical_units(self, em_spd, em_pwr, batt_soc):
        """
        Compute all the variables for the electrical energy source
        """

        em_eff = self.em.get_eff_from_pwr(em_spd, em_pwr)
        em_pwr_el = em_pwr * em_eff ** np.sign(- em_pwr)  # [W]
        batt_pwr = em_pwr_el + self.ess.access  # [W]
        unfeasible = 0

        if batt_pwr > 0 and batt_pwr > self.ess.get_pwrmax_dis(batt_soc):
            batt_pwr = self.ess.get_pwrmax_dis(batt_soc)
            unfeasible = 1
        elif batt_pwr <= 0 and abs(batt_pwr) > self.ess.get_pwrmax_chg(batt_soc):
            batt_pwr = - self.ess.get_pwrmax_chg(batt_soc)
            unfeasible = 1

        # from now on the cell is considered instead of the whole battery
        batt_cell_pwr = batt_pwr / (self.ess.num_module_parallel * self.ess.num_cell_series)  # [W]

        batt_cell_r_chg = self.ess.get_rint_chg(batt_soc)  # [Ohm]
        batt_cell_r_dis = self.ess.get_rint_dis(batt_soc)  # [Ohm]
        if em_pwr > 0:
            batt_cell_r = batt_cell_r_dis  # [Ohm]
        else:
            batt_cell_r = batt_cell_r_chg  # [Ohm]

        batt_cell_ocv = self.ess.get_ocv(batt_soc)  # [V]
        batt_cell_current = (batt_cell_ocv - (((batt_cell_ocv) ** 2 - 4 * batt_cell_r * batt_cell_pwr) ** 0.5)) / (2 * batt_cell_r)  # [A]

        # Avoiding errors with complex numbers (the current might be one of those)
        if np.imag(batt_cell_current) != 0:
            batt_cell_current = 10 ** 4  # [A]

        batt_cell_capacity = self.ess.cell_cap * 3600  # [As]
        batt_soc = batt_soc - batt_cell_current / batt_cell_capacity
        batt_energy_rate = (batt_cell_current / batt_cell_capacity) * self.ess.cell_cap * self.ess.volt_nom * self.ess.num_cell_series / 1000  # [kWh]

        return batt_soc, batt_energy_rate, batt_pwr, em_pwr_el, batt_cell_pwr, batt_cell_current, unfeasible


    def reset(self, seed=None, options=None):
        """
        Method to reset environment to initial state and output initial observation
        """
        # Random selection of the driving cycle among the driving cycles dataset
        if len(self.driving_cycles) > 1:
            cycle_num = random.randint(0, len(self.driving_cycles) - 1)
            self.sim_input['cycle'] = self.driving_cycles[cycle_num]
        else:
            self.sim_input['cycle'] = self.driving_cycles[0]

        # SoC boundary constraints calculation for soc_bc reward
        if self.reward.reward_type == 'soc_bc':
            dsoc_init = 0.05
            dsoc_end = 0.02
            time_idx = self.sim_input['cycle'].time_simu
            time_soc_decay_init = 3 / 4 * time_idx[-1]
            time_soc_decay_end = time_idx[-1] - self.time_step
            time_mask = time_idx >= time_soc_decay_init
            soc_decay = (dsoc_init - dsoc_end) / (time_soc_decay_end - time_soc_decay_init)
            soc_bc_up = np.full_like(time_idx, self.ess.soc_trg + dsoc_init)
            soc_bc_up[time_mask] = -soc_decay * (time_idx[time_mask] - time_soc_decay_init) + (
                    self.ess.soc_trg + dsoc_init)
            soc_bc_low = self.ess.soc_trg - (soc_bc_up - self.ess.soc_trg)
            soc_bc_tuple = namedtuple('soc_bc_tuple', ['time', 'up_bound', 'lo_bound'])
            soc_bc = soc_bc_tuple(time=time_idx,
                                  up_bound=soc_bc_up,
                                  lo_bound=soc_bc_low)
            self.reward.soc_bc = soc_bc

        # Set simulation input and ice state
        self.time_idx = 0
        self.set_input(self.time_idx)
        #self.set_input(tr_step=0)  #GIGI
        self.ice_state = 0

        # Get power demand
        veh_tot_force, veh_pwr = self.get_road_force(self.sim_input['veh_spd'], self.sim_input['veh_acc'],
                                                     self.sim_input['gear'], self.ice_state)
        wh_spd, wh_trq, wh_pwr = self.solve_wheels(self.sim_input['veh_spd'], veh_tot_force)
        fd_spd, fd_trq, fd_pwr = self.solve_final_drive(wh_spd, wh_trq)
        gb_spd_in, gb_trq_in, gb_pwr_in = self.solve_gearbox(fd_spd, fd_trq, fd_pwr, self.sim_input['gear'])

        # Set output
        self.reset_output()
        self.sim_output['fd_spd'] = np.append(self.sim_output['fd_spd'], fd_spd)
        self.sim_output['gb_spd_in'] = np.append(self.sim_output['gb_spd_in'], gb_spd_in)
        self.sim_output['gb_pwr_in'] = np.append(self.sim_output['gb_pwr_in'], gb_pwr_in)
        self.sim_output['ice_state'] = np.append(self.sim_output['gb_pwr_in'], self.ice_state)
        self.sim_output['batt_soc'] = np.append(self.sim_output['batt_soc'], self.ess.soc_init)

        # States initialization and normalization
        states = np.zeros(len(self.obs_name))
        states[self.obs_name.index('batt_soc')] = self.ess.soc_init
        obs_init = self.normalization.normalize(states)

        # Is done variable initialization
        self.isdone = False
        self.istruncated = False
        self.info = {}

        return obs_init, self.info

    def step(self, action):
        """
        Method that apply system dynamics and simulates the environment with the given action for one step
        """

        if action > 0.075:
            self.ice_state = 1
        else:
            self.ice_state = 0

        # Get power demand (only to go on with computations, they have already been updated in the output dict)
        veh_tot_force, veh_pwr = self.get_road_force(self.sim_input['veh_spd'], self.sim_input['veh_acc'],
                                                     self.sim_input['gear'], self.ice_state)
        wh_spd, wh_trq, wh_pwr = self.solve_wheels(self.sim_input['veh_spd'], veh_tot_force)
        fd_spd, fd_trq, fd_pwr = self.solve_final_drive(wh_spd, wh_trq)
        gb_spd_in, gb_trq_in, gb_pwr_in = self.solve_gearbox(fd_spd, fd_trq, fd_pwr, self.sim_input['gear'])

        # Get Power-split
        ice_spd, ice_pwr, ice_trq, em_spd, em_pwr, em_pwr_reg, brk_pwr_mech, unfeasible = self.solve_powersplit(action, gb_pwr_in, gb_spd_in, self.ice_state, self.sim_input['veh_spd'], self.sim_input['veh_acc'])
        # Get the energy consumption
        ice_fuel_rate, ice_co2_rate = self.solve_engine(ice_spd, ice_pwr, self.ice_state)
        batt_soc_0 = self.sim_output['batt_soc'][-1]
        batt_soc, batt_energy_rate, batt_pwr, em_pwr_el, batt_cell_pwr, batt_cell_current, unfeasible = self.solve_electrical_units(em_spd, em_pwr, batt_soc_0)

        # Reward
        reward_raw = self.reward.k1 - self.reward.get_reward(ice_fuel_rate, batt_soc, self.isdone)
        reward = self.normalization.normalize_reward(reward_raw)

        # Update remaining output
        self.sim_output['em_spd'] = np.append(self.sim_output['em_spd'], em_spd)
        self.sim_output['em_pwr'] = np.append(self.sim_output['em_pwr'], em_pwr)
        self.sim_output['em_pwr_ele'] = np.append(self.sim_output['em_pwr_ele'], em_pwr_el)
        self.sim_output['ice_state'] = np.append(self.sim_output['ice_state'], self.ice_state)
        self.sim_output['ice_spd'] = np.append(self.sim_output['ice_spd'], ice_spd)
        self.sim_output['ice_pwr'] = np.append(self.sim_output['ice_pwr'], ice_pwr)
        self.sim_output['ice_fuel'] = np.append(self.sim_output['ice_fuel'], ice_fuel_rate)
        self.sim_output['batt_pwr'] = np.append(self.sim_output['batt_pwr'], batt_pwr)
        self.sim_output['batt_soc'] = np.append(self.sim_output['batt_soc'], batt_soc)
        self.sim_output['reward'] = np.append(self.sim_output['reward'], reward)
        self.sim_output['unfeasible'] = np.append(self.sim_output['unfeasible'], unfeasible)
        self.sim_output['action'] = np.append(self.sim_output['action'], action)

        # timestep update
        if self.time_idx < self.sim_input['cycle'].time_simu[-1]:

            self.time_idx = self.time_idx + self.time_step
            self.set_input(self.time_idx)

            # Get power demand
            veh_tot_force, veh_pwr = self.get_road_force(self.sim_input['veh_spd'], self.sim_input['veh_acc'],
                                                         self.sim_input['gear'], self.ice_state)
            wh_spd, wh_trq, wh_pwr = self.solve_wheels(self.sim_input['veh_spd'], veh_tot_force)
            fd_spd, fd_trq, fd_pwr = self.solve_final_drive(wh_spd, wh_trq)
            gb_spd_in, gb_trq_in, gb_pwr_in = self.solve_gearbox(fd_spd, fd_trq, fd_pwr, self.sim_input['gear'])

            # Update outputs to call update states
            self.sim_output['veh_tot_force'] = np.append(self.sim_output['veh_tot_force'], veh_tot_force)
            self.sim_output['fd_spd'] = np.append(self.sim_output['fd_spd'], fd_spd)
            self.sim_output['fd_pwr'] = np.append(self.sim_output['fd_pwr'], fd_pwr)
            self.sim_output['gb_spd_in'] = np.append(self.sim_output['gb_spd_in'], gb_spd_in)
            self.sim_output['gb_pwr_in'] = np.append(self.sim_output['gb_pwr_in'], gb_pwr_in)

        elif self.time_idx == self.sim_input['cycle'].time_simu[-1]:
            self.isdone = True
            print(f"done - {self.time_idx}")
        elif self.SOC <= 0.3 and self.SOC >= 0.7:
            self.istruncated = True
            print(f"truncated - {self.time_idx}")

        # Update states and normalize them
        states = self.update_states()
        obs = self.normalization.normalize(states)
        self.info = {}

        return obs, reward, self.isdone, self.istruncated, self.info


class Reward:
    """
    Object that defines several reward functions to be used for training RL agents for HEV power-split management
    """

    def __init__(self, reward_type='dense', k1=0, k2=-100, k3=-200, unfeaspenalty=0, soc_bc_gain=None, soc_bc=None,
                 veh_param=None):
        """
        Constructor method: Initializes all the parameters of the reward
        """
        self.reward_type = reward_type
        if not self.check_reward_type():
            raise ValueError('The selected reward type is not valid: please use "dense", "sparse" or "soc_bc"')
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.unfeaspenalty = unfeaspenalty
        if veh_param is None:
            veh_param_tuple = namedtuple('veh_param',
                                         ['fuel_heating_val', 'cell_cap', 'num_cell_series', 'volt_nom', 'soc_trg',
                                          'time_step'])
            veh_param = veh_param_tuple(fuel_heating_val=43000000,
                                        cell_cap=28.60,
                                        num_cell_series=100,
                                        volt_nom=3.71,
                                        soc_trg=0.5,
                                        time_step=1
                                        )
        self.veh_param = veh_param
        if reward_type == 'soc_bc':
            if soc_bc_gain is None:
                soc_bc_gain = 1 / 10
            if soc_bc is None:
                soc_bc_tuple = namedtuple('soc_bc_tuple', ['time', 'up_bound', 'lo_bound'])
                soc_bc = soc_bc_tuple(time=0,
                                      up_bound=0,
                                      lo_bound=0)
            self.soc_bc = soc_bc
            self.soc_bc_gain = soc_bc_gain

    def get_eq_fuel_consumption(self, batt_soc, ):
        """
        Method to set the equations that describes a dense reward, in which the SoC penalty is continuously applied
        during the driving cycle
        """
        eff_ch = 0.95  # Sort of average chemical efficiency during charge and discharge of the battery
        soc_gain = self.veh_param.cell_cap * 3600 * self.veh_param.volt_nom * self.veh_param.num_cell_series / (
                eff_ch * self.veh_param.fuel_heating_val)
        return self.k3 * soc_gain * (batt_soc - self.veh_param.soc_trg) ** 2  # equivalent fuel consumption in kg

    def get_reward(self, ice_fuel_rate, batt_soc, isdone, out_of_range=False):
        """
        Method to compute the reward considering both the ice fuel consumption and the equivalent fuel consumtpion
        coming from the electric energy stored in the battery exploitation
        """
        fuel_consumption = self.k2 * ice_fuel_rate * self.veh_param.time_step
        eq_fuel_consumption = 0
        if self.reward_type == 'dense':
            eq_fuel_consumption = self.get_eq_fuel_consumption(batt_soc)
        elif self.reward_type == 'sparse':
            if isdone == 1:
                eq_fuel_consumption = self.get_eq_fuel_consumption(batt_soc)
            else:
                eq_fuel_consumption = 0
        elif self.reward_type == 'soc_bc':
            eq_fuel_consumption = self.get_eq_fuel_consumption(batt_soc)
            soc_up = np.interp(batt_soc, self.soc_bc.time, self.soc_bc.up_bound)
            soc_low = np.interp(batt_soc, self.soc_bc.time, self.soc_bc.low_bound)
            out_of_range = (batt_soc >= soc_up) or (batt_soc <= soc_low)
            if out_of_range:
                eq_fuel_consumption = (1 + self.soc_bc_gain) * eq_fuel_consumption
        return fuel_consumption + eq_fuel_consumption

    def check_reward_type(self):
        if self.reward_type == 'dense':
            return True
        elif self.reward_type == 'sparse':
            return True
        elif self.reward_type == 'soc_bc':
            return True
        else:
            return False


class Normalization:
    """
    Object used to normalize the states of the system given the states name
    """

    def __init__(self, up_value=1, low_value=0):
        """
        Constructor method: Initializes all the parameters of the normalization object
        states: veh_spd, veh_acc, em_spd, batt_soc, veh_dist
        """
        self.up_value = up_value
        self.low_value = low_value
        self.state_max = np.array([47.25, 4.45, 372.6, 0.7, 1])
        self.state_min = np.array([0, -10.45, 0, 0.3, 0])

    def normalize(self, state):
        """
        Method to normalize the states into observations
        """
        obs = (state - self.state_min) / (self.state_max - self.state_min) * (
                    self.up_value - self.low_value) + self.low_value
        return obs

    def normalize_reward(self, reward, max_reward=1, min_reward=0):
        """
        Method to normalize the reward
        """
        return (reward - max_reward) / (max_reward - min_reward) * (self.up_value - self.low_value) + self.low_value
