from legged_gym.envs.base.base_config import BaseConfig
from search_strategies_config import *

class RunConfig(BaseConfig):
    train_base: bool = True
    collect_base: bool = True
    search_model: bool = False
    train_best_model: bool = False
    collect_best_model: bool = False
    record_error: bool = False
    
    trained_model: str = "plane_default_flip" # location of trained model to be used
    results_folder = "results_flip"
    root_save_folder: str = f"{results_folder}/test"
    ground_truth_path: str = f"{root_save_folder}/ground_truth/gt.pkl"
    best_model_path: str = f"{root_save_folder}/best_model/model.pkl"
    iteration_error = f'{root_save_folder}/errors/iteration.pkl'
    trained_best_model_path = "plane_using_best_model"
    observation_path: str = f"{root_save_folder}/observation/obs.pkl"

class TrainConfig(BaseConfig):

    class env:
        num_envs = 4096
    
    class domain_rand:
        randomize_friction = True
        # randomize_base_mass = True # walk
        randomize_base_mass = False
        friction_range = [0.5, 1.25]
        added_mass_range = [-1., 1.]
        # push_robots = True  # walk
        push_robots = False
    
    class noise:
        add_noise = True 
    
    class commands:
        class ranges:
            # lin_vel_x = [0., 1.0] # min max [m/s]
            # lin_vel_y = [0., 0.]   # min max [m/s]
            # ang_vel_yaw = [-1, 1]    # min max [rad/s]
            # heading = [-3.14, 3.14]

            lin_vel_x = [0., 0.] # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [-3.14., 3.14]    # min max [rad/s]
            heading = [0., 0.]

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = -0.0
            tracking_ang_vel = 0.0
            lin_vel_z = 1.0
            ang_vel_xy = -1.0
            orientation = -0.
            torques = -0.0
            dof_vel = -0.
            dof_acc = -0.
            base_height = -0.0 
            feet_air_time =  2.0
            collision = -1.0
            feet_stumble = -0.0 
            action_rate = -0.0
            stand_still = -0.
        
        only_positive_rewards = False

class SearchConfig(BaseConfig):
    strategies = dict(exhaustive=exhaustive(), random=random(), adaptive=adaptive(), cross_entropy=cross_entropy()) 
    class noise:
        add_noise = False
        
    class init_state:
        add_noise = False


    class commands:
        class ranges:
            lin_vel_x = [1., 1.] # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [0., 0.]    # min max [rad/s]
            heading = [0, 0]
            
    class domain_range:
        randomize_friction = True
        randomize_base_mass = True
        friction_range = [1., 1.]
        added_mass_range = [-0.5, -0.5]
    
    class strategy:
        name = "cross_entropy"
        

class CollectConfig(BaseConfig):
    collect_data = False
    collection: str = "isaac" # "isaac", "real"
    rollout_size: int = 1000
    class env:
        num_envs = 1
    
    class domain_rand:
        randomize_friction = True
        randomize_base_mass = True
        friction_range = [1., 1.]
        added_mass_range = [-0.5, -0.5]

    class noise:
        add_noise = False
    class init_state:
        add_noise = False
        
    class commands:
        class ranges:
            lin_vel_x = [1., 1.] # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [0., 0.]    # min max [rad/s]
            heading = [0, 0]
    
    class additional_params:
        class env:
            shapes = list(range(13))# [0] 
            bodies = [0, *list(range(10, 17))] # [0, 1]
            class query_model:
                ## ground truth definition
                shapes = [1.4,] * 13 # friction values -> same as len(env.shapes)
                bodies = [0.4, *([0.002] * 5), *([-0.001] * 2)]  # mass values -> same as len(env.bodies)
                


