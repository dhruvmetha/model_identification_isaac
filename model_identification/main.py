import pickle
import identifier
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from search_strategies import *
import time

# search_strategies = {
#     'exhaustive': BatchExhaustiveSearch,
# }

def error_function(a, b):
    return torch.sum((a-b) ** 2, axis=1)

def run_for_query_points(args, env_cfg, train_cfg, query_points, rollout_size=100):
    env_cfg.env.query_points = query_points
    env_cfg.env.num_envs = len(query_points)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg) # makes all the environments given in env_cfg 
    obs = env.get_observations()
    qp_obs = []
    # qp_obs.append(env.dof_pos - env.default_dof_pos)
    
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root='plane_default')
    policy = ppo_runner.get_inference_policy(device=env.device)

    # start = time.time()
    for i in range(rollout_size):
        qp_obs.append(env.dof_pos - env.default_dof_pos)
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
    
    for created_env in env.envs:
        env.gym.destroy_env(created_env)
    # end = time.time()
    # print(end - start)

    env.gym.destroy_sim(env.sim)
    env.gym.destroy_viewer(env.viewer)
    
    return qp_obs

def main(args):
    gt_obs = None
    with open('ground_truth/gt.pkl', 'rb') as f:
        gt_obs = pickle.load(f)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # env_cfg.env.num_envs = 2
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.num_rows = 100
    env_cfg.terrain.num_cols = 100
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False # TODO: maybe we can change this to see the effects of noise on ground truth data (may give a more solid proof of concept).
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.friction_range = [1., 1.]
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [2., 2.]
    env_cfg.domain_rand.push_robots = False
    
    env_cfg.commands.ranges.lin_vel_x = [1, 1]
    env_cfg.commands.ranges.lin_vel_y = [0, 0]
    env_cfg.commands.ranges.ang_vel_yaw = [-1, 1]
    env_cfg.commands.ranges.heading = [0, 0]
    
    # run a search strategy here and get the query points.
    env_cfg.env.friction_joints = {0: 0, 1: 1, 4: 2, 7: 3, 10: 4}
    env_cfg.env.mass_bodies = {0: 0}
    
    
    search_ranges = {'friction': {
                'start': 0.8,
                'end': 1.2,
                'step': 4, 
                'repeat': len(env_cfg.env.friction_joints)
            },
            'mass': {
                'start': -2,
                'end': 3,
                'step': 10, 
                'repeat': len(env_cfg.env.mass_bodies)
            }
    }
    
    num_iterations = 20
    rollout_size = 5
    batch_size = 4096
    # search = BatchRandomSearch(search_ranges, batch_size=batch_size, num_iterations=num_iterations)
    search = BatchExhaustiveSearch(search_ranges, batch_size=batch_size)
    # error_tracker = None #torch.zeros(batch_size)
    best_qps = (None, 999999)
    while True:
        error_tracker = None
        qps, done = search.get_query_points()
        obs = run_for_query_points(args, env_cfg, train_cfg, qps, rollout_size=rollout_size)
        if (error_tracker is None) and len(obs) > 0:
            error_tracker = torch.zeros(len(qps)).to(obs[0].device)
        for i in range(rollout_size):
            if False:
                print(obs[i].shape)
                print((obs[i] - gt_obs[i]))
            error_tracker += error_function(obs[i], gt_obs[i])
        # errors = error_function(obs, gt_obs)
        for k, v in zip(qps, error_tracker):
            print(k, v)
        # print(error_tracker)
        min_error_qp, min_error = qps[torch.argmin(error_tracker)], torch.min(error_tracker)
        if min_error < best_qps[1]:
            best_qps = (min_error_qp, min_error)
            print(min_error_qp)
        if done:
            break
    print(best_qps)
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
    
    
