from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import pickle
import torch
import numpy as np
from utils import overwrite_cfg, add_params_cfg, destroy_simulator

class IsaacGymCollector:
    def __init__(self, trained_policy_model_path, rollout_size,  *args):
        
        
        self.trained_policy_model, self.rollout_size = trained_policy_model_path, rollout_size
        self.args, self.env_cfg, self.train_cfg = args
        
        self.collect_path = None
        
        # self.env_cfg.env.num_envs = min(self.env_cfg.env.num_envs, 1)
        # self.env_cfg.terrain.mesh_type = "plane"
        # self.env_cfg.terrain.num_rows = 10
        # self.env_cfg.terrain.num_cols = 10
        # self.env_cfg.terrain.curriculum = False
        # self.env_cfg.noise.add_noise = False # TODO: maybe we can change this to see the effects of noise on ground truth data (may give a more solid proof of concept).
        # self.env_cfg.domain_rand.randomize_friction = False
        # self.env_cfg.domain_rand.randomize_base_mass = False
        # self.env_cfg.domain_rand.push_robots = False
        
        # self.env_cfg.commands.ranges.lin_vel_x = [0, 0]
        # self.env_cfg.commands.ranges.lin_vel_y = [0, 0]
        # self.env_cfg.commands.ranges.ang_vel_yaw = [0, 0]
        # self.env_cfg.commands.ranges.heading = [0, 0]
        
    def reset_cfg(self, env_cfg):
        overwrite_cfg(self.env_cfg, env_cfg)
        try:
            add_params_cfg(self.env_cfg, env_cfg.additional_params)
        except:
            print('no additional params to add')

    def set_collect_path(self, collect_path):
        self.collect_path = collect_path
        
    def train_model(self):
        # self.env_cfg.env.num_envs = 4096 #min(self.env_cfg.env.num_envs, 4096)
        
        # more robust policy for better search
        # self.env_cfg.noise.add_noise = True
        # self.env_cfg.domain_rand.randomize_friction = True
        # self.env_cfg.domain_rand.randomize_base_mass = True
        # self.env_cfg.domain_rand.push_robots = True
        
        env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg) # makes all the environments given in env_cfg 
        _ = env.get_observations()
       
        self.train_cfg.runner.resume = False
        ppo_runner, self.train_cfg = task_registry.make_alg_runner(env=env, name=self.args.task, args=self.args, train_cfg=self.train_cfg, log_root=self.trained_policy_model)
        
        ppo_runner.learn(num_learning_iterations=self.train_cfg.runner.max_iterations, init_at_random_ep_len=True)
        
        destroy_simulator(env)
        
    def collect_data(self):
        
        if not self.collect_data:
            raise "Path to storing the data collected needs to be specified"
        
        ground_truth_states = []
        mean_rew = 0
        
        self.env_cfg.env.shapes = {v:k for k, v in enumerate(self.env_cfg.env.shapes)}
        self.env_cfg.env.bodies = {v:k for k, v in enumerate(self.env_cfg.env.bodies)}
        
        env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg) # makes all the environments given in env_cfg 
        obs = env.get_observations()
        
        self.train_cfg.runner.resume = True
        ppo_runner, self.train_cfg = task_registry.make_alg_runner(env=env, name=self.args.task, args=self.args, train_cfg=self.train_cfg, log_root=self.trained_policy_model)
        
        policy = ppo_runner.get_inference_policy(device=env.device)
        
        total_rew_tracker = torch.zeros((self.env_cfg.env.num_envs, 1)).to(env.device)
        total_ep_tracker = torch.ones((self.env_cfg.env.num_envs, 1)).to(env.device)
        
        for i in range(self.rollout_size):
            ground_truth_states.append(env.dof_pos - env.default_dof_pos)
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            total_rew_tracker +=  rews.view(-1, 1)
            total_ep_tracker += torch.where(dones.view(-1, 1), 1, 0)
        
        if self.args.save_gt:
            with open(self.collect_path, 'wb') as f:
                pickle.dump(ground_truth_states, f)
        print(total_rew_tracker/total_ep_tracker)
        destroy_simulator(env)
        
        
# if __name__ == '__main__':
#     args = get_args()  
#     i_dc = IsaacGymCollector(args, 'plane_default')
#     # i_dc.train_model()
#     i_dc.collect_data('ground_truth/gt.pkl', num_steps=1000)