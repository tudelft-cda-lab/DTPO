import gymnax

from .gymnax_wrapper import GymnaxToGymnasiumWrapper, GymnaxToVectorGymnasiumWrapper

import gymnasium


class Leaf:
    def __init__(self, value, prune=False):
        self.value = value
        self.prune = prune
        self.id = None


class Node:
    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.id = None


def make_env_from_name(
    env_name, seed=None, return_gym_env=False, return_gym_vec_env=False, num_envs_vec=1
):
    """
    Returns a gymnax or gymnasium environment and env_params with the given name.
    For gymnasium environments the env_params returned is None.
    """
    if env_name in gymnax.registered_envs:
        env, env_params = gymnax.make(env_name)
    else:
        if env_name == "CartPoleSwingup":
            from .environments.cartpoleswingup import CartPoleSwingUp

            env = CartPoleSwingUp()
        elif env_name == "Navigation3D":
            from .environments.navigation_3d import Navigation3D

            env = Navigation3D()
        elif env_name == "Blackjack":
            from .environments.blackjack import Blackjack

            env = Blackjack()
        elif env_name == "Frozenlake4x4":
            from .environments.frozenlake import Frozenlake4x4

            env = Frozenlake4x4()
        elif env_name == "Frozenlake8x8":
            from .environments.frozenlake import Frozenlake8x8

            env = Frozenlake8x8()
        elif env_name == "Frozenlake12x12":
            from .environments.frozenlake import Frozenlake12x12

            env = Frozenlake12x12()
        elif env_name == "InventoryManagement":
            from .environments.inventory_management import InventoryManagement

            env = InventoryManagement()
        elif env_name == "PendulumBangBang":
            from .environments.pendulum_bangbang import PendulumBangBang

            env = PendulumBangBang()
        elif env_name == "SystemAdministrator1":
            from .environments.system_administrator import SystemAdministrator1

            env = SystemAdministrator1()
        elif env_name == "SystemAdministrator2":
            from .environments.system_administrator import SystemAdministrator2

            env = SystemAdministrator2()
        elif env_name == "SystemAdministratorTree":
            from .environments.system_administrator import SystemAdministratorTree

            env = SystemAdministratorTree()
        elif env_name == "TictactoeVsRandom":
            from .environments.tictactoe_vs_random import TictactoeVsRandom

            env = TictactoeVsRandom()
        elif env_name == "TigerVsAntelope":
            from .environments.tiger_vs_antelope import TigerVsAntelope

            env = TigerVsAntelope()
        elif env_name == "TrafficIntersection":
            from .environments.traffic_intersection import TrafficIntersection

            env = TrafficIntersection()
        elif env_name == "Xor":
            from .environments.xor import Xor

            env = Xor()
        else:
            raise ValueError(f"Unkown env_name {env_name}")

        env_params = env.default_params

    gym_env = None
    gym_vec_env = None

    if return_gym_env:
        # If env_params is None then this is already a gymnasium environment
        if env_params is None:
            gym_env = env
        else:
            gym_env = GymnaxToGymnasiumWrapper(env=env, params=env_params, seed=seed)

    if return_gym_vec_env:
        # If env_params is None then this is already a gymnasium environment
        if env_params is None:
            gym_vec_env = gymnasium.vector.SyncVectorEnv(
                [lambda: env for _ in range(num_envs_vec)]
            )
        else:
            gym_vec_env = GymnaxToVectorGymnasiumWrapper(
                env=env, num_envs=num_envs_vec, params=env_params, seed=seed
            )

    if not return_gym_env and not return_gym_vec_env:
        return env, env_params
    elif return_gym_env and not return_gym_vec_env:
        return env, env_params, gym_env
    elif not return_gym_env and return_gym_vec_env:
        return env, env_params, gym_vec_env
    elif return_gym_env and return_gym_vec_env:
        return env, env_params, gym_env, gym_vec_env
