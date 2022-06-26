from legged_gym.envs.base.base_config import BaseConfig
from search_strategies_config import *
from task_reward_config import *

class RunConfig(BaseConfig):
    train_base: bool = False
    collect_base: bool = True
    search_model: bool = True 
    train_best_model: bool = False 
    collect_best_model: bool = False
    record_error: bool = False
    
    all_trained_models_folder = "models"
    trained_model: str = f"{all_trained_models_folder}/plane_walk" # location of trained model to be used
    trained_best_model_path = f"{all_trained_models_folder}/plane_walk_best_model"
    
    results_folder = "walk"
    search_technique = f"{results_folder}/adaptive_base"
    ground_truth_path: str = f"{search_technique}/ground_truth/gt.pkl"
    best_model_path: str = f"{search_technique}/best_model/model.pkl"
    iteration_error = f'{search_technique}/errors/iteration.pkl'
    observation_path: str = f"{search_technique}/observation/obs.pkl"

class TrainConfig(BaseConfig):
    task_config = dict(walk=WalkConfig(), back_flip=BackFlipConfig())
    
    base_task = "walk"
    test_task = "walk"
    

class SearchConfig(BaseConfig):
    strategies = dict(exhaustive=exhaustive(), random=random(), adaptive=adaptive(), cross_entropy=cross_entropy()) 
    class noise:
        add_noise = False
        
    class init_state:
        add_noise = False


    class commands:
        class ranges:
            lin_vel_x = [0., 0.] # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [0., 0.]    # min max [rad/s]
            heading = [0, 0]
            
    class domain_range:
        randomize_friction = True
        randomize_base_mass = True
        friction_range = [1., 1.]
        added_mass_range = [-0.5, -0.5]
    
    class strategy:
        name = "adaptive"
        

class CollectConfig(BaseConfig):
    collect_data = False
    collection: str = "isaac" # "isaac", "real"
    rollout_size: int = 1000
    class env:
        num_envs = 1
        # episode_length_s = 2
    
    class domain_rand:
        randomize_friction = True
        randomize_base_mass = True
        friction_range = [1., 1.]
        added_mass_range = [0., 0.]

    class noise:
        add_noise = False
    class init_state:
        add_noise = False
        
    class commands:
        class ranges:
            lin_vel_x = [0., 0.] # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [0., 0.]    # min max [rad/s]
            heading = [0, 0]
    
    class additional_params:
        class env:
            shapes = [0]  # list(range(13)) 
            bodies =  [0] # [0, *list(range(10, 17))] # [0, 1]
            class query_model:
                ## ground truth definition
                shapes = [1.0,] # friction values -> same as len(env.shapes)
                bodies = [-0.4]  # mass values -> same as len(env.bodies)
                


