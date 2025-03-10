import yaml

from envs import T1Hist
from learning.runners import OnPolicyRunner, ROARunner

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from easydict import EasyDict


# class RslRlPpoActorCriticCfg(EasyDict):
#     """Configuration for the PPO actor-critic networks."""

#     class_name: str = "ActorCritic"
#     """The policy class name. Default is ActorCritic."""

#     init_noise_std: float = MISSING
#     """The initial noise standard deviation for the policy."""

#     actor_hidden_dims: list[int] = MISSING
#     """The hidden dimensions of the actor network."""

#     critic_hidden_dims: list[int] = MISSING
#     """The hidden dimensions of the critic network."""

#     activation: str = MISSING
#     """The activation function for the actor and critic networks."""


# class RslRlPpoAlgorithmCfg(EasyDict):
#     """Configuration for the PPO algorithm."""

#     class_name: str = "PPO"
#     """The algorithm class name. Default is PPO."""

#     value_loss_coef: float = MISSING
#     """The coefficient for the value loss."""

#     use_clipped_value_loss: bool = MISSING
#     """Whether to use clipped value loss."""

#     clip_param: float = MISSING
#     """The clipping parameter for the policy."""

#     entropy_coef: float = MISSING
#     """The coefficient for the entropy loss."""

#     num_learning_epochs: int = MISSING
#     """The number of learning epochs per update."""

#     num_mini_batches: int = MISSING
#     """The number of mini-batches per update."""

#     learning_rate: float = MISSING
#     """The learning rate for the policy."""

#     schedule: str = MISSING
#     """The learning rate schedule."""

#     gamma: float = MISSING
#     """The discount factor."""

#     lam: float = MISSING
#     """The lambda parameter for Generalized Advantage Estimation (GAE)."""

#     desired_kl: float = MISSING
#     """The desired KL divergence."""

#     max_grad_norm: float = MISSING
#     """The maximum gradient norm."""


# class RslRlOnPolicyRunnerCfg(EasyDict):
#     """Configuration of the runner for on-policy algorithms."""

#     seed: int = 42
#     """The seed for the experiment. Default is 42."""

#     device: str = "cuda:0"
#     """The device for the rl-agent. Default is cuda:0."""

#     num_steps_per_env: int = MISSING
#     """The number of steps per environment per update."""

#     max_iterations: int = MISSING
#     """The maximum number of iterations."""

#     empirical_normalization: bool = MISSING
#     """Whether to use empirical normalization."""

#     policy: RslRlPpoActorCriticCfg = MISSING
#     """The policy configuration."""

#     algorithm: RslRlPpoAlgorithmCfg = MISSING
#     """The algorithm configuration."""

#     ##
#     # Checkpointing parameters
#     ##

#     save_interval: int = MISSING
#     """The number of iterations between saves."""

#     experiment_name: str = MISSING
#     """The experiment name."""

#     run_name: str = ""
#     """The run name. Default is empty string.

#     The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
#     then it is appended to the run directory's name, i.e. the logging directory's name will become
#     ``{time-stamp}_{run_name}``.
#     """

#     ##
#     # Logging parameters
#     ##

#     logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
#     """The logger to use. Default is tensorboard."""

#     neptune_project: str = "isaaclab"
#     """The neptune project name. Default is "isaaclab"."""

#     wandb_project: str = "isaaclab"
#     """The wandb project name. Default is "isaaclab"."""

#     ##
#     # Loading parameters
#     ##

#     resume: bool = False
#     """Whether to resume. Default is False."""

#     load_run: str = ".*"
#     """The run directory to load. Default is ".*" (all).

#     If regex expression, the latest (alphabetical order) matching run will be loaded.
#     """

#     load_checkpoint: str = "model_.*.pt"
#     """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

#     If regex expression, the latest (alphabetical order) matching file will be loaded.
#     """

class RslRlPpoActorCriticCfg(EasyDict):
    init_noise_std=1.0
    actor_hidden_dims=[256, 128, 128]
    critic_hidden_dims=[256, 256, 128]
    activation="elu"
    class_name="ActorCriticHistory"

class RslRlPpoAlgorithmCfg(EasyDict):
    value_loss_coef=1.0
    use_clipped_value_loss=True
    clip_param=0.2
    entropy_coef=0.01
    num_learning_epochs=5
    num_mini_batches=1
    learning_rate=1.0e-3
    schedule="adaptive"
    gamma=0.99
    lam=0.95
    desired_kl=0.01
    max_grad_norm=1.0
    priv_reg_coef_schedual = [0, 0.1, 5000, 10000]
    class_name="ROAPPO"

class T1PPORunnerCfg(EasyDict):
    num_steps_per_env = 24
    max_iterations = 40000
    save_interval = 500
    experiment_name = "t1_human_reward"
    logger = "wandb"
    wandb_project = "RL-Booster"
    empirical_normalization = False
    device = "cuda:3"

    # policy = RslRlPpoActorCriticCfg()
    # algorithm = RslRlPpoAlgorithmCfg()

    dagger_update_freq = 20


if __name__ == "__main__":
    with open("envs/T1Hist.yaml", "r") as f:
        env_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    env = T1Hist(env_cfg)

    ppo_cfg = T1PPORunnerCfg()
    ppo_cfg.policy = RslRlPpoActorCriticCfg()
    ppo_cfg.algorithm = RslRlPpoAlgorithmCfg()

    runner = ROARunner(
        env,
        ppo_cfg,
        log_dir="logs",
        device="cuda:3",
    )

    runner.learn(40000)