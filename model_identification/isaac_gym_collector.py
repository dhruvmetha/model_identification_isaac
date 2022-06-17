from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import pickle
import torch
import numpy as np

class IsaacGymCollector:
    def __init__(self, args, log_root):
        self.args = args
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        
        self.env_cfg = env_cfg

        self.train_cfg = train_cfg
        self.log_root = log_root
        
        # self.env_cfg.env.num_envs = min(self.env_cfg.env.num_envs, 1)
        self.env_cfg.terrain.mesh_type = "plane"
        self.env_cfg.terrain.num_rows = 10
        self.env_cfg.terrain.num_cols = 10
        self.env_cfg.terrain.curriculum = False
        self.env_cfg.noise.add_noise = False # TODO: maybe we can change this to see the effects of noise on ground truth data (may give a more solid proof of concept).
        self.env_cfg.domain_rand.randomize_friction = False
        self.env_cfg.domain_rand.randomize_base_mass = False
        self.env_cfg.domain_rand.push_robots = False
        
        # self.env_cfg.commands.ranges.lin_vel_x = [0, 0]
        # self.env_cfg.commands.ranges.lin_vel_y = [0, 0]
        # self.env_cfg.commands.ranges.ang_vel_yaw = [0, 0]
        # self.env_cfg.commands.ranges.heading = [0, 0]
        
    def train_model(self):
        self.env_cfg.env.num_envs = 4096 #min(self.env_cfg.env.num_envs, 4096)
        
        # more robust policy for better search
        self.env_cfg.noise.add_noise = True
        self.env_cfg.domain_rand.randomize_friction = True
        self.env_cfg.domain_rand.randomize_base_mass = True
        self.env_cfg.domain_rand.push_robots = False
        env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg) # makes all the environments given in env_cfg 
        _ = env.get_observations()
       
        self.train_cfg.runner.resume = False
        ppo_runner, self.train_cfg = task_registry.make_alg_runner(env=env, name=self.args.task, args=self.args, train_cfg=self.train_cfg, log_root=self.log_root)
        
        ppo_runner.learn(num_learning_iterations=self.train_cfg.runner.max_iterations, init_at_random_ep_len=True)
        
    def collect_data(self, path, num_steps=100):
        ground_truth_states = []
        mean_rew = 0
        self.env_cfg.env.num_envs = 1
        
        self.env_cfg.noise.add_noise = False
        self.env_cfg.init_state.add_noise = False
        
        self.env_cfg.domain_rand.randomize_friction = True
        self.env_cfg.domain_rand.friction_range = [2., 4.]
        self.env_cfg.domain_rand.randomize_base_mass = True
        self.env_cfg.domain_rand.added_mass_range = [1., 1.]
        
        self.env_cfg.commands.ranges.lin_vel_x = [1, 1]
        self.env_cfg.commands.ranges.lin_vel_y = [0, 0]
        self.env_cfg.commands.ranges.ang_vel_yaw = [-1, 1]
        self.env_cfg.commands.ranges.heading = [0, 0]
        
        shapes = [0] # list(range(13)) # all the colliding shapes we care about (friction)
        bodies = [0] # list(range(17)) # all the rigid_bodies we care about (mass)
        
        self.env_cfg.env.shapes = {v:k for k, v in enumerate(shapes)}
        self.env_cfg.env.bodies = {v:k for k, v in enumerate(bodies)}
        
        print(self.env_cfg.env.shapes, self.env_cfg.env.bodies)
        
        env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg) # makes all the environments given in env_cfg 
        obs = env.get_observations()
        
        
        # for i in env.gym.get_actor_rigid_shape_properties(env.envs[0], env.actor_handles[0]):
        #     print(i.friction)
        # print(len(env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])))       
        # ground_truth_states.append(env.dof_pos - env.default_dof_pos)
        
        self.train_cfg.runner.resume = True
        ppo_runner, self.train_cfg = task_registry.make_alg_runner(env=env, name=self.args.task, args=self.args, train_cfg=self.train_cfg, log_root=self.log_root)
        
        policy = ppo_runner.get_inference_policy(device=env.device)
        
        total_rew_tracker = torch.zeros((self.env_cfg.env.num_envs, 1)).to(env.device)
        total_ep_tracker = torch.ones((self.env_cfg.env.num_envs, 1)).to(env.device)
        
        for i in range(num_steps):
            ground_truth_states.append(env.dof_pos - env.default_dof_pos)
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            total_rew_tracker +=  rews.view(-1, 1)
            total_ep_tracker += torch.where(dones.view(-1, 1), 1, 0)
        
        if self.args.save_gt:
            with open(path, 'wb') as f:
                pickle.dump(ground_truth_states, f)
        print(total_rew_tracker/total_ep_tracker)
        
        
if __name__ == '__main__':
    args = get_args()   
    i_dc = IsaacGymCollector(args, 'plane_default')
    i_dc.train_model()
    # i_dc.collect_data('ground_truth/gt.pkl', num_steps=1000)