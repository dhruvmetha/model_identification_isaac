import torch
import inspect

def torch_random_tensor(min, max, size):
    return (max-min) * torch.rand(size=size) + min


def is_builtin_class_instance(obj):
    return obj.__class__.__module__ == 'builtins'

def add_params_cfg(cfg, cfg_add):
    for key in dir(cfg_add):
        if key.startswith('__') or key == "init_member_classes":
            continue
        var_new = getattr(cfg_add, key)
        var = getattr(cfg, key, None)
        if inspect.isclass(type(var_new)) and not is_builtin_class_instance(var_new):
            if var is None:
                setattr(cfg, key, var_new)
                print('adding', key)
                var = getattr(cfg, key)
            add_params_cfg(var, var_new)
        else:
            setattr(cfg, key, var_new)
            print('adding', key, var_new)

def overwrite_cfg(cfg, cfg_ov):
    for key in dir(cfg):
        c_keys = [c for c in dir(cfg_ov) if not c.startswith('__')]
        if key.startswith('__'):
            continue
        if key in c_keys:
            if key in ["init_member_classes"]:
                continue
            var = getattr(cfg, key)
            var_ov = getattr(cfg_ov, key)
            if inspect.isclass(type(var)) and not is_builtin_class_instance(var):
                overwrite_cfg(var, var_ov)
            else:
                setattr(cfg, key, var_ov)
                print('setting', key, 'to', var_ov)
                
def destroy_simulator(env):
    for created_env in env.envs:
        env.gym.destroy_env(created_env)
    env.gym.destroy_sim(env.sim)
    env.gym.destroy_viewer(env.viewer)
    