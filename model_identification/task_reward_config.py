from legged_gym.envs.base.base_config import BaseConfig

class TaskConfig(BaseConfig):
    class env:
        num_envs = 4096
     
    class init_state:   
        pos = [0.0, 0.0, 0.42]
    
class WalkConfig(TaskConfig):
    class domain_rand:
        randomize_friction = True
        randomize_base_mass = True
        friction_range = [1., 1.]
        added_mass_range = [0., 0.]
        push_robots = True
    
    class noise:
        add_noise = True
    
    class commands:
        class ranges:
            lin_vel_x = [0., 1.0] # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
                
class BackFlipConfig(TaskConfig):
    
    class env(TaskConfig.env):
        episode_length_s = 1
    
    class domain_rand:
        randomize_friction = True
        randomize_base_mass = False
        friction_range = [0.5, 1.25]
        added_mass_range = [-1., 1.]
        push_robots = False
    
    class noise:
        add_noise = False 
    
    class commands:
        class ranges:
            lin_vel_x = [0., 0.] # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0., 0.]

    class rewards:
        class scales:
            termination = -2.0
            
            tracking_lin_vel = -0.0
            tracking_ang_vel = -0.0
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            # orientation = -1.
            # torques = -0.0
            # dof_vel = -0.
            # dof_acc = -0.
            base_height = -0.0
            feet_air_time =  0.0
            collision = -2.0
            # feet_stumble = -0.0 
            action_rate = -0.01
            # stand_still = -0.
        
        only_positive_rewards = False
        
    class additional_params:
        class rewards:
            class scales:
                ang_vel_pitch = 2.0
                pre_termination = -0.0
                # lin_vel = -0.2