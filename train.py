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
parser = argparse.ArgumentParser(description='SAC_training')
parser.add_argument('--agent_name', type=str, default=datetime.date.today().strftime("%Y%m%d")+"_SAC", metavar='S',
                    help='agent name (default: date_of_today_SAC)')
parser.add_argument('--critic_hidden_nodes', type=int, default=256, metavar='S',
                    help='Number of neurons of the 2 hidden layers of the critic policy (default: 256)')
parser.add_argument('--actor_hidden_nodes', type=int, default=256, metavar='S',
                    help='Number of neurons of the 2 hidden layers of the actor policy (default: 256)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='S',
                    help='Learning rate for the policy nets update (default: 0.0003)')
parser.add_argument('--batch', type=float, default=256, metavar='S',
                    help='Mini-batch size for the nets training (default: 256)')
parser.add_argument('--tau', type=float, default=0.005, metavar='S',
                    help='The soft update coefficient for the target critic (default: 0.005)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='S',
                    help='Discount factor (default: 0.99)')
parser.add_argument('--episodes', type=int, default=500, metavar='S',
                    help='Episodes to run during the training phase (default: 500)')
args = parser.parse_args()
# Saving directories
models_dir = f"results/saved_agents/{args.agent_name}"
logdir = "results/logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Vehicle and driving cycles definition
ice, em, ess, gb, fd, wh, veh = gen_vehicle_objects('data/vehicle_data.xlsx')
if os.path.exists('data/driving_cycles.pkl'):
    with open("data/driving_cycles.pkl", "rb") as file:
        driving_cycles = pickle.load(file)
else:
    driving_cycles = gen_driving_cycles_objects('data/driving_cycles.xlsx')

# Environment definition
register(
    id='Mercedes300deP2-v0',
    entry_point='src.envs.VehicleBackwardKin:VehicleBackwardKinEnv',
)
env_id = 'Mercedes300deP2-v0'
cycle = {'WLTC':driving_cycles['WLTC']}
env = gym.make(env_id, driving_cycles=cycle, veh=veh, wh=wh, fd=fd, gb=gb, ice=ice, em=em, ess=ess, env_time_step=1)

# Model definition and training
policy_kwargs = dict(
    net_arch=dict(qf=[args.critic_hidden_nodes, args.critic_hidden_nodes], pi=[args.actor_hidden_nodes, args.actor_hidden_nodes])
)
model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=logdir, learning_rate=args.lr,
            learning_starts=200, batch_size=args.batch, tau=args.tau, gamma=args.gamma, train_freq=1,
            gradient_steps=1, target_update_interval=1, policy_kwargs=policy_kwargs,
            ent_coef='auto')
model.learn(total_timesteps=args.episodes*1801, log_interval=10, reset_num_timesteps=False, tb_log_name=args.agent_name, progress_bar=True)
model_path = os.path.join(models_dir, f"{args.agent_name}")
model.save(model_path)
'''model.save(f"{models_dir}/{args.agent_name}")'''
