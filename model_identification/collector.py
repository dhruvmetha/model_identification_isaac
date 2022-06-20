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
        self.query_model = self.cfg.env.query_model # num_envs x 13 x 1
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
            if s in self.cfg.env.shapes.keys():
                props[s].friction = self.query_points[env_id][self.cfg.env.shapes[s]]
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
            if s in self.cfg.env.bodies.keys():
                props[s].mass += self.query_model[env_id][self.cfg.env.bodies[s]]
            # else:
            #     props[s].mass = self.mass_coeffs[env_id]
        # props[0].mass += self.query_model[env_id][-1]
        return props

task_registry.register('a1_collect', LeggedRobotForCollection, A1RoughCfgCollection(), A1RoughCfgPPO())

