import isaacgym
import numpy as np
import torch
from legged_gym.envs import *
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from legged_gym.utils import get_args, task_registry
from legged_gym.envs import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from utils import torch_random_tensor

class A1RoughCfgCollection(A1RoughCfg):
    class init_state( A1RoughCfg.init_state ):
        add_noise = False
        
class LeggedRobotForCollection(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.query_model = getattr(self.cfg.env, "query_model", None)
        # self.partition_point = None
        # if self.query_model is not None:
        #     self.partition_point = len(self.cfg.env.shapes)
        # self.cfg.env.num_envs = 1
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
    
    # def set_query_points(self, query_points):
    #     print('here')
    #     self.query_points = query_points
        
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_random_tensor(friction_range[0], friction_range[1], (num_buckets,1))
                self.friction_coeffs = friction_buckets[bucket_ids]

        for s in range(len(props)):
            if getattr(self.cfg.env, "shapes", None) is not None and s in self.cfg.env.shapes.keys():
                props[s].friction = self.query_model.shapes[self.cfg.env.shapes[s]]
            else:
                props[s].friction = self.friction_coeffs[env_id]
        return props
    
    
    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        for s in range(len(props)):
            if getattr(self.cfg.env, "bodies", None) is not None and s in self.cfg.env.bodies.keys():
                props[s].mass += self.query_model.bodies[self.cfg.env.bodies[s]]
            else:
                if self.cfg.domain_rand.randomize_base_mass and s == 0:
                    rng = self.cfg.domain_rand.added_mass_range
                    props[s].mass += np.random.uniform(rng[0], rng[1])
            # else:
            #     props[s].mass = self.mass_coeffs[env_id]
        # props[0].mass += self.query_model[env_id][-1]
        return props
    
    
    # addtional reward functions
    
    def _reward_ang_vel_pitch(self):
        # rewarding back flip while punishing sideways rotation
        
        return (- self.base_ang_vel[:, 1] - torch.abs(self.base_ang_vel[:, 0]) - torch.abs(self.base_ang_vel[:, 2])) * (self.episode_length_buf < (0.6  * self.max_episode_length)) 
        
        # rewarding front flip while punishing sideways rotation
        return self.base_ang_vel[:, 1] - torch.abs(self.base_ang_vel[:, 0])
    
    def _reward_pre_termination(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :]), dim=1) * (self.episode_length_buf > (0.95  * self.max_episode_length)) 
    
    def _reward_lin_vel(self):
        # penalizing any linear velocity
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    
    def _reward_termination(self):
        # Terminal reward / penalty
        r = torch.sum(torch.square(self.base_ang_vel[:, :]), dim=1) * self.time_out_buf + torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * self.time_out_buf
        return r
        return torch.sum(torch.square(self.root_states[:] - self.base_init_state), dim=1) * self.time_out_buf
        

task_registry.register('a1_collect', LeggedRobotForCollection, A1RoughCfgCollection(), A1RoughCfgPPO())

