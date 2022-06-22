from legged_gym.envs.base.base_config import BaseConfig
from search_strategies_config import *

class RunConfig(BaseConfig):
    train_base: bool = False
    collect_base: bool = True
    search_model: bool = True
    train_best_model: bool = False
    collect_best_model: bool = False
    record_error: bool = False
    
    trained_model: str = "plane_default_test" # location of trained model to be used
    root_save_folder: str = "exhaustive_1dim"
    ground_truth_path: str = f"{root_save_folder}/ground_truth/gt.pkl"
    best_model_path: str = f"{root_save_folder}/best_model/model.pkl"
    iteration_error = f'{root_save_folder}/errors/iteration.pkl'
    trained_best_model_path = "plane_using_best_model"
    observation_path: str = f"{root_save_folder}/observation/obs.pkl"

class TrainConfig(BaseConfig):
    
    class domain_rand:
        randomize_friction = True
        randomize_base_mass = True
        friction_range = [0.5, 1.25]
        added_mass_range = [-1., 1.]
        push_robots = True
    
    class noise:
        add_noise = True
    
    class commands:
        class ranges:
            lin_vel_x = [0., 1.0] # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

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
        name = "exhaustive"
        

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
            shapes = [0]
            bodies = [0]
            class query_model:
                ## ground truth definition
                shapes = [1.4,] # friction values -> same as len(env.shapes)
                bodies = [-0.4,] # mass values -> same as len(env.bodies)
                


