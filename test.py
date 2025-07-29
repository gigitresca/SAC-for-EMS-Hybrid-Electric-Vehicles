import gymnasium as gym
import datetime
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib
matplotlib.use('TkAgg')
import pickle
import pickletools
import os
import argparse

from src.utilities.initialization import gen_vehicle_objects, gen_driving_cycles_objects
from gymnasium.envs.registration import register
from stable_baselines3 import SAC

# Argument parser
parser = argparse.ArgumentParser(description='SAC_test')
parser.add_argument('--agent_name', type=str, default=datetime.date.today().strftime("%Y%m%d")+"_SAC", metavar='S',
                    help='agent name (default: date_of_today_SAC)')
args = parser.parse_args()

# Vehicle and driving cycles definition
ice, em, ess, gb, fd, wh, veh = gen_vehicle_objects('data/vehicle_data.xlsx')
if os.path.exists('data/driving_cycles.pkl'):
    with open("data/driving_cycles.pkl", "rb") as file:
        driving_cycles = pickle.load(file)
else:
    driving_cycles = gen_driving_cycles_objects('data/driving_cycles.xlsx')

# Register the environment
register(
    id='Mercedes300deP2-v0',
    entry_point='src.envs.VehicleBackwardKin:VehicleBackwardKinEnv',
)
env_id = 'Mercedes300deP2-v0'
cycle = {'WLTC':driving_cycles['WLTC']}
env = gym.make(env_id, driving_cycles=cycle, veh=veh, wh=wh, fd=fd, gb=gb, ice=ice, em=em, ess=ess, env_time_step=1)


model_name = f"results/saved_agents/{args.agent_name}.pth"
model = SAC.load(model_name, env=env)

episodes = 1
for ep in range(episodes):
    obs, info = env.reset()
    episode_reward = []
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward.append(reward)

        if done or truncated:
            obs, info = env.reset()
