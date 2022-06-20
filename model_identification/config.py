from legged_gym.envs.base.base_config import BaseConfig


class RunConfig(BaseConfig):
    train: bool = False
    trained_model: str = None # location of trained model to be used
    
    class domain_rand:
            randomize_friction = True
            randomize_base_mass = True


class SearchConfig(RunConfig):
    class env:
        class domain_rand(RunConfig.domain_rand):
            friction_range = [0.5, 1.25]
            added_mass_range = [1., 1.]

    class search:
        strategy = ""


class CollectConfig(RunConfig):
    class env:
        class domain_rand(RunConfig.domain_rand):
            friction_range = [0.5, 1.25]
            added_mass_range = [1., 1.]
