#!/usr/bin/env python3

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.cmd_util import make_robotics_env, robotics_arg_parser
import mujoco_py
import baselines.common.tf_util as U
from kukaCamGymEnv import KukaCamGymEnv
from cnnpolicy import MlpPolicy
# from cnn_policy import CnnPolicy
import pposgd_simple

def train(env, num_timesteps, seed):


    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    # mujoco_py.ignore_mujoco_warnings().__enter__()
    workerseed = seed + 10000 * rank
    set_global_seeds(workerseed)
    # env = make_robotics_env(env_id, workerseed, rank=rank)
    # def policy_fn(name, ob_space, ac_space):
    #     return CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    #         hid_size=256, num_hid_layers=3)

    # def policy_fn(name, ob_space, ac_space):
    #     return CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)

    def policy_fn(name,ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_dims_p=[64, 64], hid_dims_v=[64, 64])

    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.01,
            optim_epochs=5, optim_stepsize=3e-4, optim_batchsize=256,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()


def main():
    # args = robotics_arg_parser().parse_args()
    env = KukaCamGymEnv(renders=False)
    num_timesteps = 50000
    train(env, num_timesteps=num_timesteps, seed=1000)


if __name__ == '__main__':
    main()
