import gymnasium as gym
import h5py
import numpy as np


class VizDoomDataCollection:
    def __init__(self):
        self.data =  {
            'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
        }
    
    def on_step(self, obs, act, done, rew):
        self.data['observations'].append(obs)
        self.data['actions'].append(act)
        self.data['terminals'].append(done)
        self.data['rewards'].append(rew)
    
    def save_dataset(self, fname):
        dataset = h5py.File(fname, "w")
        for k in self.data:
            if k == 'terminals':
                dtype = np.bool_
            else:
                dtype = np.float32

            data = np.array(self.data[k], dtype=dtype)
            # TODO(aty): make sure the obs space is trivial!
            dataset.create_dataset(k, data=data, compression='gzip')
        return dataset


class OfflineVizDoomGymnasiumWrapper(gym.Wrapper):
    def __init__(self, env, dataset_path, **kwargs):
        self.env = env
        self.dataset_path = dataset_path
        self.dc = VizDoomDataCollection()
        gym.Wrapper.__init__(self, env, **kwargs)
    
    def step(self, act, **kwargs):
        obs, rew, term, info, done = self.env.step(act, **kwargs)
        self.dc.on_step(obs, act, done, rew)
        return obs, rew, term, info, done
    
    def save_dataset(self):
        return self.dc.save_dataset(self.dataset_path)
    
    def close(self, **kwargs):
        self.save_dataset()
        return self.env.close(**kwargs)


def find_dc_wrapper(env: gym.Wrapper):
    while isinstance(env, gym.Wrapper):
        if isinstance(env, OfflineVizDoomGymnasiumWrapper):
            return env
        else:
            env = env.unwrapped
    raise RuntimeError("OfflineVizDoomGymnasiumWrapper is not found")