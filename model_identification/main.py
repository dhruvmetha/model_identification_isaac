import inspect
import pickle
from pathlib import Path
import numpy as np
import identifier
from isaac_gym_collector import IsaacGymCollector
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from search_strategies import *
import time
from config import *
import collector
from utils import *
from matplotlib import pyplot as plt

search_strategies = {
    'exhaustive': BatchExhaustiveSearch,
    'random': BatchRandomSearch,
    'adaptive': AdaptiveSearch, 
    'cross_entropy': CrossEntropySearch
}

def error_function(a, b):
    return torch.sum((a-b) ** 2, axis=1)

def run_for_query_points(args, env_cfg, train_cfg, query_points, base_policy, rollout_size=100):
    env_cfg.env.query_points = query_points
    env_cfg.env.num_envs = len(query_points)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg) # makes all the environments given in env_cfg 
    obs = env.get_observations()
    qp_obs = []
    # qp_obs.append(env.dof_pos - env.default_dof_pos)
    
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=base_policy)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # start = time.time()
    for i in range(rollout_size):
        qp_obs.append(env.dof_pos - env.default_dof_pos)
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
    destroy_simulator(env)
    return qp_obs       

def main(args):
    
    run_config, env_train_config, env_search_config, env_collect_config= RunConfig(), TrainConfig(), SearchConfig(), CollectConfig()
    
    Path(run_config.root_save_folder).mkdir(parents=True, exist_ok=True)
    
    if run_config.collect_base:
    # collect ground truth data for base policy
        if env_collect_config.collection == "isaac":
            # setup configs and start sim
            mode = "gt"
            env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
            collector = IsaacGymCollector(run_config.trained_model, env_collect_config.rollout_size, args, env_cfg, train_cfg)
            # if trained base policy does not exist -> run trainer
            if run_config.train_base:
                collector.reset_cfg(env_train_config)
                collector.train_model()
            # run collector and save ground truth
            collector.reset_cfg(env_collect_config)
            Path(run_config.ground_truth_path).parent.mkdir(parents=True, exist_ok=True)
            collector.set_collect_path(run_config.ground_truth_path)
            collector.collect_data()
            # destroy simulator
            
            
    ## read ground truth
    gt_obs = None
    with open(run_config.ground_truth_path, 'rb') as f:
        gt_obs = pickle.load(f)
        
    if run_config.search_model:
        
    # setup search using the strategy and ranges (friction of shapes, mass of bodies) given in search config
        args.task = "a1_search"
        
        name = env_search_config.strategy.name
        strategy = env_search_config.strategies[name]
        
        Search = search_strategies[name](strategy)
        
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        overwrite_cfg(env_cfg, env_search_config)
        add_params_cfg(env_cfg, strategy.additional_params)
        env_cfg.env.shapes = {v:k for k, v in enumerate(env_cfg.env.shapes)}
        env_cfg.env.bodies = {v:k for k, v in enumerate(env_cfg.env.bodies)}

        iteration_error = []
        # do search and return best model
        qps_error = (None, None)
        best_qps = (None, 999999) 
        while True:
            error_tracker = None
            query_points, done = Search.get_query_points(best_qps[0], qps_error)
            if len(query_points) == 0:
                raise 'query points is 0 -> something is wrong'
                break
            obs = run_for_query_points(args, env_cfg, train_cfg, query_points, run_config.trained_model, strategy.rollout_size)
            if (error_tracker is None) and len(obs) > 0:
                error_tracker = torch.zeros(len(query_points)).to(obs[0].device)
            for i in range(strategy.rollout_size):
                error_tracker += error_function(obs[i], gt_obs[i])
            qps_error = (query_points, error_tracker)
            # for k, v in zip(query_points, error_tracker):
            #     print(k, v)    
            min_error_qp, min_error = query_points[torch.argmin(error_tracker)], torch.min(error_tracker)
            if min_error < best_qps[1]:
                print(min_error_qp, min_error)
                best_qps = (min_error_qp, min_error)
            iteration_error.append(best_qps[1])
            if done:
                break
            
        Path(run_config.iteration_error).parent.mkdir(parents=True, exist_ok=True)
        with open(run_config.iteration_error, 'wb') as f:
            pickle.dump(iteration_error, f)
        
        best_model = {
            "shapes" : env_cfg.env.shapes,
            "bodies" : env_cfg.env.bodies,
            "shapes_model" : best_qps[0][:len(env_cfg.env.shapes)],
            "bodies_model" : best_qps[0][-len(env_cfg.env.bodies):],
            "others": env_collect_config.domain_rand
        }
        Path(run_config.best_model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(run_config.best_model_path, 'wb') as f:
            pickle.dump(best_model, f)

    best_model = None
    with open(run_config.best_model_path, 'rb') as f:    
        best_model = pickle.load(f)
    
    shapes_, bodies_ = best_model['shapes'], best_model['bodies']
    shapes_model, bodies_model, others = best_model['shapes_model'], best_model['bodies_model'], best_model['others']
    print('best model', shapes_model, bodies_model)
    
    class additional_params_:
        class env:
            shapes = shapes_
            bodies = bodies_
            class query_model:
                shapes = shapes_model
                bodies = bodies_model
    
    class new_train_config(BaseConfig):
        domain_rand = others
        additional_params = additional_params_
    
    class new_collect_config(BaseConfig):
        domain_rand = others
        additional_params = additional_params_
    
    if run_config.collect_best_model:    
        args.task = "a1_collect"
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        collector = IsaacGymCollector(run_config.trained_best_model_path, env_collect_config.rollout_size, args, env_cfg, train_cfg)
        # train policy using simulator and the best model
        if run_config.train_best_model:
            collector.reset_cfg(env_train_config)
            collector.reset_cfg(new_train_config())
            collector.train_model()
        # run policy in simulation and collect data
        collector.reset_cfg(env_collect_config)
        collector.reset_cfg(new_collect_config())
        Path(run_config.observation_path).parent.mkdir(parents=True, exist_ok=True)
        collector.set_collect_path(run_config.observation_path)
        collector.collect_data()
    
    # measure error against the ground truth
    if run_config.record_error:
        best_model_obs = None
        with open(run_config.observation_path, 'rb') as f:
            best_model_obs = pickle.load(f)
    
        error = []
        rollout_for_error = min(len(gt_obs), len(best_model_obs))
        for gt, bm in zip(gt_obs[:rollout_for_error], best_model_obs[:rollout_for_error]):
            error.append(error_function(gt, bm).cpu().numpy())
        
        plt.plot(error)
        plt.show()
    
    # env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # gt_obs = None
    # with open('ground_truth/gt.pkl', 'rb') as f:
    #     gt_obs = pickle.load(f)
    # env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # # env_cfg.env.num_envs = 2
    # env_cfg.terrain.mesh_type = "plane"
    # env_cfg.terrain.num_rows = 100
    # env_cfg.terrain.num_cols = 100
    # env_cfg.terrain.curriculum = False
    # env_cfg.noise.add_noise = False # TODO: maybe we can change this to see the effects of noise on ground truth data (may give a more solid proof of concept).
    # env_cfg.domain_rand.randomize_friction = True
    # env_cfg.domain_rand.friction_range = [1., 1.]
    # env_cfg.domain_rand.randomize_base_mass = True
    # env_cfg.domain_rand.added_mass_range = [2., 2.]
    # env_cfg.domain_rand.push_robots = False
    
    # env_cfg.commands.ranges.lin_vel_x = [1, 1]
    # env_cfg.commands.ranges.lin_vel_y = [0, 0]
    # env_cfg.commands.ranges.ang_vel_yaw = [-1, 1]
    # env_cfg.commands.ranges.heading = [0, 0]
    
    # # run a search strategy here and get the query points.
    # env_cfg.env.friction_joints = {0: 0, 1: 1, 4: 2, 7: 3, 10: 4}
    # env_cfg.env.mass_bodies = {0: 0}
    
    
    # search_ranges = {'friction': {
    #             'start': 0.8,
    #             'end': 1.2,
    #             'step': 4, 
    #             'repeat': len(env_cfg.env.friction_joints)
    #         },
    #         'mass': {
    #             'start': -2,
    #             'end': 3,
    #             'step': 10, 
    #             'repeat': len(env_cfg.env.mass_bodies)
    #         }
    # }
    
    # num_iterations = 20
    # rollout_size = 5
    # batch_size = 4096
    # # search = BatchRandomSearch(search_ranges, batch_size=batch_size, num_iterations=num_iterations)
    # search = BatchExhaustiveSearch(search_ranges, batch_size=batch_size)
    # # error_tracker = None #torch.zeros(batch_size)
    # best_qps = (None, 999999)
    # while True:
    #     error_tracker = None
    #     qps, done = search.get_query_points()
    #     obs = run_for_query_points(args, env_cfg, train_cfg, qps, rollout_size=rollout_size)
    #     if (error_tracker is None) and len(obs) > 0:
    #         error_tracker = torch.zeros(len(qps)).to(obs[0].device)
    #     for i in range(rollout_size):
    #         if False:
    #             print(obs[i].shape)
    #             print((obs[i] - gt_obs[i]))
    #         print(gt_obs[i].device, obs[i].device)
    #         error_tracker += error_function(obs[i], gt_obs[i])
    #     # errors = error_function(obs, gt_obs)
    #     for k, v in zip(qps, error_tracker):
    #         print(k, v)
    #     # print(error_tracker)
    #     min_error_qp, min_error = qps[torch.argmin(error_tracker)], torch.min(error_tracker)
    #     if min_error < best_qps[1]:
    #         best_qps = (min_error_qp, min_error)
    #         print(min_error_qp)
    #     if done:
    #         break
    # print(best_qps)
    
    
    
    
    
    
    # query_points = get_query_points('none', size=(env_cfg.env.num_envs * 2, 13+1), batch_size=batch_size)
    # finished_points = 0
    # friction_envs = []
    # mass_envs = []
    # while finished_points < len(query_points):
    #     env_cfg.env.query_points = query_points[finished_points : finished_points + batch_size]
    #     env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg) # makes all the environments given in env_cfg 
    #     obs = env.get_observations()
        
    #     train_cfg.runner.resume = True
    #     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root='plane_default')
    #     mean_rew = 0
    #     policy = ppo_runner.get_inference_policy(device=env.device)
        
    #     err_tracker = torch.zeros(env_cfg.env.num_envs).to(env.device)
        
    #     for i in range(1):
    #         err_tracker += torch.sum(((env.dof_pos - env.default_dof_pos) - gt_obs[i])**2, axis=1)
    #         actions = policy(obs.detach())
    #         obs, _, rews, dones, infos = env.step(actions.detach())
        
    #     for created_env in env.envs:
    #         env.gym.destroy_env(created_env)
    #     env.gym.destroy_sim(env.sim)
    #     env.gym.destroy_viewer(env.viewer)
        
    #     # for i in err
        
    #     # friction_envs.extend([[i.friction] for i in env.gym.get_actor_rigid_shape_properties(env.envs[torch.argmin(err)], env.actor_handles[torch.argmin(err)])])
    #     # env = None
    #     finished_points += batch_size
    #     # gt_states = None
    # # with open('ground_truth/gt.pkl', 'rb') as f:
    # #     gt_states = pickle.load(f)
        
    # # print(len(gt_states[:100]))
    # # print(gt_states[0].shape)
    # return
    
    
    
if __name__ == "__main__":
    main(get_args())
    
    
