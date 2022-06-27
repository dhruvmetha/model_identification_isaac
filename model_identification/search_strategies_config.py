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
        friction = [dict(start=0., end=3., step=2)] * 1,
        mass = [dict(start=0., end=2.5, step=10)] * 5) 
    
    class additional_params:
        class env:
            shapes = [0]  # list(range(13)) -> number of colliding shapes
            bodies = [0, 10, 11, 14, 15] # list(range(17)) -> number of rigid bodies with mass    
            
class adaptive(BaseConfig):
    rollout_size = 100
    name = "adaptive"
    batch_size = 4096
    num_iterations = 15
    
    ranges = dict(
        friction = [dict(start=0., end=3., step=2)] * 1,
        mass = [dict(start=0., end=2., step=8)] * 5)
    
    interval_zoom = 0.75
    adaptive_step = [*[2], *([5]*5)]
    
    class additional_params:
        class env:
            shapes = [0]  # list(range(13)) -> number of colliding shapes
            bodies = [0, 10, 11, 14, 15] # list(range(17)) -> number of rigid bodies with mass 
            
            
class cross_entropy(BaseConfig):
    rollout_size = 100
    name = "cross_entropy"
    batch_size = 4096
    num_iterations = 15

    start_mean = np.random.rand(6) # same size as all shapes + all bodies that need to be adjusted
    start_var = np.diag([1] * 6) # same size as all shapes + all bodies that need to be adjusted
    elite_frac = 0.1
    
    # ranges = dict(
    #     friction = [dict(start=0., end=10., step=64)],
    #     mass = [dict(start=-3., end=3., step=64)])
    
    
    class additional_params:
        class env:
            shapes = [0] # [0]  # list(range(13)) -> number of colliding shapes
            bodies = [0, 10, 11, 14, 15] # [0] # list(range(17)) -> number of rigid bodies with mass 
