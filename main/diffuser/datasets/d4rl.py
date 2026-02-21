import os
import collections
import numpy as np
import pdb
import pickle as pkl
import re

_backend = os.environ.get("CONTRADIFF_EVAL_BACKEND", "auto").strip().lower()
if _backend in ("gymnasium", "gymnasium_mujoco", "mujoco"):
    gym = None
else:
    try:
        import gym
    except Exception:
        gym = None

try:
    import gymnasium
except Exception:
    gymnasium = None

# from contextlib import (
#     contextmanager,
#     redirect_stderr,
#     redirect_stdout,
# )

# @contextmanager
# def suppress_output():
#     """
#         A context manager that redirects stdout and stderr to devnull
#         https://stackoverflow.com/a/52442331
#     """
#     with open(os.devnull, 'w') as fnull:
#         with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
#             yield (err, out)

# with suppress_output():
#     ## d4rl prints out a variety of warnings
d4rl = None
if _backend not in ("gymnasium", "gymnasium_mujoco", "mujoco"):
    try:
        import d4rl  # noqa: F401
    except Exception as e:
        d4rl = None
        print(f"[ diffuser/datasets/d4rl ] WARNING: d4rl import failed ({type(e).__name__}: {e})")


class _OfflineDummyEnv:
    def __init__(self, name, max_episode_steps):
        self.name = name
        self.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps

    def seed(self, seed=None):
        return

    def get_dataset(self):
        raise RuntimeError(
            f"Dummy env for {self.name} has no dataset. "
            "Use precomputed dataset_infos/cluster_infos, or install full d4rl+mujoco runtime."
        )


def _fallback_max_episode_steps(name: str):
    lower = name.lower()
    if lower.startswith(("halfcheetah-", "hopper-", "walker2d-")):
        return 1000
    return None


_GYMNASIUM_MUJOCO_ENV_MAP = {
    "halfcheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "walker2d": "Walker2d-v4",
    "ant": "Ant-v4",
}

_D4RL_REF_MIN_SCORE = {
    "halfcheetah-random-v0": -280.178953,
    "halfcheetah-medium-v0": -280.178953,
    "halfcheetah-expert-v0": -280.178953,
    "halfcheetah-medium-replay-v0": -280.178953,
    "halfcheetah-medium-expert-v0": -280.178953,
    "hopper-random-v0": -20.272305,
    "hopper-medium-v0": -20.272305,
    "hopper-expert-v0": -20.272305,
    "hopper-medium-replay-v0": -20.272305,
    "hopper-medium-expert-v0": -20.272305,
    "walker2d-random-v0": 1.629008,
    "walker2d-medium-v0": 1.629008,
    "walker2d-expert-v0": 1.629008,
    "walker2d-medium-replay-v0": 1.629008,
    "walker2d-medium-expert-v0": 1.629008,
    "ant-random-v0": -325.6,
    "ant-medium-v0": -325.6,
    "ant-expert-v0": -325.6,
    "ant-medium-replay-v0": -325.6,
    "ant-medium-expert-v0": -325.6,
}

_D4RL_REF_MAX_SCORE = {
    "halfcheetah-random-v0": 12135.0,
    "halfcheetah-medium-v0": 12135.0,
    "halfcheetah-expert-v0": 12135.0,
    "halfcheetah-medium-replay-v0": 12135.0,
    "halfcheetah-medium-expert-v0": 12135.0,
    "hopper-random-v0": 3234.3,
    "hopper-medium-v0": 3234.3,
    "hopper-expert-v0": 3234.3,
    "hopper-medium-replay-v0": 3234.3,
    "hopper-medium-expert-v0": 3234.3,
    "walker2d-random-v0": 4592.3,
    "walker2d-medium-v0": 4592.3,
    "walker2d-expert-v0": 4592.3,
    "walker2d-medium-replay-v0": 4592.3,
    "walker2d-medium-expert-v0": 4592.3,
    "ant-random-v0": 3879.7,
    "ant-medium-v0": 3879.7,
    "ant-expert-v0": 3879.7,
    "ant-medium-replay-v0": 3879.7,
    "ant-medium-expert-v0": 3879.7,
}


def _backend_name():
    return os.environ.get("CONTRADIFF_EVAL_BACKEND", "auto").strip().lower()


def _to_v0_name(name):
    match = re.match(r"^(.*)-v\d+$", name)
    if match is None:
        return name
    return f"{match.group(1)}-v0"


def _gymnasium_env_id(name):
    key = name.split("-")[0].lower()
    return _GYMNASIUM_MUJOCO_ENV_MAP.get(key)


class _GymnasiumMujocoCompatEnv:
    def __init__(self, d4rl_name):
        env_id = _gymnasium_env_id(d4rl_name)
        if env_id is None:
            raise ValueError(
                f"Unsupported dataset for gymnasium backend: {d4rl_name}. "
                f"Supported prefixes: {sorted(_GYMNASIUM_MUJOCO_ENV_MAP.keys())}"
            )
        if gymnasium is None:
            raise ImportError(
                "gymnasium is not available. Install with: pip install 'gymnasium[mujoco]' mujoco"
            )

        self.name = d4rl_name
        self.gymnasium_env_id = env_id
        self._env = gymnasium.make(env_id)

        max_steps = getattr(self._env, "_max_episode_steps", None)
        if max_steps is None and getattr(self._env, "spec", None) is not None:
            max_steps = getattr(self._env.spec, "max_episode_steps", None)
        if max_steps is None:
            max_steps = _fallback_max_episode_steps(d4rl_name) or 1000
        self.max_episode_steps = int(max_steps)
        self._max_episode_steps = self.max_episode_steps

        self._seed = None
        ref_key = _to_v0_name(d4rl_name)
        self._ref_min = _D4RL_REF_MIN_SCORE.get(ref_key)
        self._ref_max = _D4RL_REF_MAX_SCORE.get(ref_key)

    def seed(self, seed=None):
        if seed is None:
            self._seed = None
            return []
        self._seed = int(seed)
        if hasattr(self._env.action_space, "seed"):
            self._env.action_space.seed(self._seed)
        return [self._seed]

    def reset(self):
        kwargs = {}
        if self._seed is not None:
            kwargs["seed"] = self._seed
        observation, _ = self._env.reset(**kwargs)
        self._seed = None
        return np.asarray(observation, dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.ndim > 1:
            action = action.reshape(-1)
        observation, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        return np.asarray(observation, dtype=np.float32), float(reward), done, info

    def get_normalized_score(self, score):
        score = float(score)
        if (
            self._ref_min is None
            or self._ref_max is None
            or self._ref_max <= self._ref_min
        ):
            return score
        return (score - self._ref_min) / (self._ref_max - self._ref_min)

    def get_dataset(self):
        # Keep compatibility with existing loading path: when evaluation loads
        # saved diffusion/value experiments, dataset construction still expects
        # an offline dataset provider.
        return _load_dataset_from_dataset_info(self.name)

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()


#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name

    if _backend_name() in ("gymnasium", "gymnasium_mujoco", "mujoco"):
        env = _GymnasiumMujocoCompatEnv(name)
        print(
            f"[ diffuser/datasets/d4rl ] INFO: using gymnasium backend for {name} "
            f"-> {env.gymnasium_env_id}"
        )
        return env

    # with suppress_output():
    #     wrapped_env = gym.make(name)
    last_error = None
    if gym is not None:
        try:
            wrapped_env = gym.make(name)
            env = wrapped_env.unwrapped
            env.max_episode_steps = wrapped_env._max_episode_steps
            env.name = name
            return env
        except Exception as e:
            last_error = e
    else:
        last_error = ImportError("gym is unavailable")

    max_steps = _fallback_max_episode_steps(name)
    if max_steps is not None:
        print(
            f"[ diffuser/datasets/d4rl ] WARNING: gym.make({name}) failed "
            f"({type(last_error).__name__}: {last_error}); using dummy env (max_episode_steps={max_steps})."
        )
        return _OfflineDummyEnv(name, max_steps)
    raise last_error

def _dataset_info_candidates(env_name):
    roots = []
    env_path = os.environ.get('DATASET_INFOS_PATH')
    if env_path:
        roots.append(env_path)
    roots.append('./dataset_infos')
    roots.append(os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_infos'))

    for root in roots:
        path = os.path.abspath(os.path.join(root, f'dataset_info_{env_name}.pkl'))
        if os.path.exists(path):
            yield path


def _load_dataset_from_dataset_info(env_name):
    for p in _dataset_info_candidates(env_name):
        with open(p, 'rb') as f:
            info = pkl.load(f)
        if isinstance(info, dict) and 'dataset' in info:
            print(f"[ diffuser/datasets/d4rl ] INFO: loaded dataset for {env_name} from {p}")
            return info['dataset']

    raise FileNotFoundError(
        f"dataset_info_{env_name}.pkl not found under DATASET_INFOS_PATH/./dataset_infos"
    )


def get_dataset(env):
    try:
        dataset = env.get_dataset()
    except Exception as e:
        if isinstance(env, _OfflineDummyEnv):
            dataset = _load_dataset_from_dataset_info(env.name)
        else:
            raise

    # if 'antmaze' in str(env).lower():
    #     ## the antmaze-v0 environments have a variety of bugs
    #     ## involving trajectory segmentation, so manually reset
    #     ## the terminal and timeout fields
    #     dataset = antmaze_fix_timeouts(dataset)
    #     dataset = antmaze_scale_rewards(dataset)
    #     get_max_delta(dataset)

    return dataset

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1



def sequence_dataset_plain(env, preprocess_fn):

    dataset = get_dataset(env)
    # dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset
    all_data = []
    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            all_data.append(episode_data)
            data_ = collections.defaultdict(list)

        episode_step += 1

    return all_data, dataset

# 返回值： 1. 所有的切分的trajectory  2. 原始的dataset
# 注意： 没有做padding
def sequence_dataset_mix(env):

    dataset = get_dataset(env)
    # dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset
    all_data = []
    episode_step = 0
    start = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            # if done_bool:
            #     print("what the hell")
            end = i
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            episode_data['start'] = start
            episode_data['end'] = end
            episode_data['accumulated_reward'] = np.sum(episode_data['rewards'])
            start = end + 1
            all_data.append(episode_data)
            data_ = collections.defaultdict(list)

        episode_step += 1

    return all_data, dataset


def sequence_dataset_maze2d(env):

    dataset = get_dataset(env)
    # dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    all_data = []
    episode_step = 0
    start = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])
        if done_bool:
            end = i
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            episode_data['start'] = start
            episode_data['end'] = end
            episode_data['accumulated_reward'] = np.sum(episode_data['rewards'])
            start = end + 1
            all_data.append(episode_data)
            data_ = collections.defaultdict(list)

        episode_step += 1

    return all_data, dataset


from tqdm import tqdm

def sequence_dataset_mix_kitchen(env):

    dataset = get_dataset(env)
    # dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    use_timeouts = False
    all_data = []
    episode_step = 0
    start = 0
    for i in tqdm(range(N)):

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])
        
        if dataset['terminals'][i]:
            # if done_bool:
            #     print("what the hell")
            end = i
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            episode_data['start'] = start
            episode_data['end'] = end
            episode_data['accumulated_reward'] = None
            start = end + 1
            all_data.append(episode_data)
            data_ = collections.defaultdict(list)

        episode_step += 1

    return all_data, dataset


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
