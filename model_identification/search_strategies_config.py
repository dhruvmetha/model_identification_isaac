from legged_gym.envs.base.base_config import BaseConfig
import numpy as np

class random(BaseConfig):
    rollout_size = 100
    name = "random"
    batch_size = 4096
    ranges = dict(
        friction = [dict(start=-2., end=2., step=11)],
        mass = [dict(start=-6., end=6., step=50)])
    
    class additional_params:
        class env:
            shapes = [0]  # list(range(13)) -> number of colliding shapes
            bodies = [0] # list(range(17)) -> number of rigid bodies with mass

class exhaustive(BaseConfig):
    rollout_size = 100
    name = "exhaustive"
    batch_size = 4096
    ranges = dict(
        friction = [dict(start=0., end=3., step=100)],
        mass = [dict(start=-2., end=2., step=100)])
    
    class additional_params:
        class env:
            shapes = [0]  # list(range(13)) -> number of colliding shapes
            bodies = [0] # list(range(17)) -> number of rigid bodies with mass    
            
class adaptive(BaseConfig):
    rollout_size = 100
    name = "adaptive"
    batch_size = 4096
    num_iterations = 10
    
    ranges = dict(
        friction = [dict(start=0., end=2., step=5)],
        mass = [dict(start=-1., end=1., step=5)] * 2)
    
    interval_zoom = 0.75
    adaptive_step = 5
    
    class additional_params:
        class env:
            shapes = [0]  # list(range(13)) -> number of colliding shapes
            bodies = [0, 1] # list(range(17)) -> number of rigid bodies with mass 
            
            
class cross_entropy(BaseConfig):
    rollout_size = 100
    name = "cross_entropy"
    batch_size = 4096
    num_iterations = 50

    start_mean = 2 * np.random.rand(21) - 1 # same size as all shapes + all bodies that need to be adjusted
    start_var = np.diag([1] * 21) # same size as all shapes + all bodies that need to be adjusted
    elite_frac = 0.01
    
    # ranges = dict(
    #     friction = [dict(start=0., end=10., step=64)],
    #     mass = [dict(start=-3., end=3., step=64)])
    
    class additional_params:
        class env:
            shapes = list(range(13)) # [0]  # list(range(13)) -> number of colliding shapes
            bodies = [0, *list(range(10, 17))] # [0] # list(range(17)) -> number of rigid bodies with mass 
