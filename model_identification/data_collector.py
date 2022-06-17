from abc import ABC


class DataStrategy(ABC):
    pass

class DataCollector:
    """Data Collector Interface
    To collect data using any policy. This collection can happen either using a real robot (using ROS or others) or can be collected in simulation (Isaac Sim/Gym).
    """
    def __init__(self, store, name, strategy, env, policy):
        pass