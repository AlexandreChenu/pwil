# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Imitation loop for PWIL."""

import time

import acme
from acme.utils import counting
from acme.utils import loggers
import dm_env

import matplotlib.pyplot as plt

import seaborn
seaborn.set()
seaborn.set_style("whitegrid")


class TrainEnvironmentLoop(acme.core.Worker):
  """PWIL environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. This can be used as:

    loop = TrainEnvironmentLoop(environment, actor, rewarder)
    loop.run(num_steps)

  The `Rewarder` overwrites the timestep from the environment to define
  a custom reward.

  The runner stores episode rewards and a series of statistics in the provided
  `Logger`.
  """

  def __init__(
      self,
      environment,
      actor,
      rewarder,
      counter=None,
      logger=None,
      workdir=None
  ):
    self._environment = environment
    self._actor = actor
    self._rewarder = rewarder
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger()
    self._workdir = workdir
    self._logdir = self._logger._file.name.split('/logs/')[0]
    self._zone_logfile_train = open(self._logdir + "/train_max_zone.txt", "w")

  def run(self, num_steps, it):
    """Perform the run loop.

    Args:
      num_steps: number of steps to run the loop for.
    """
    current_steps = 0

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # self._environment.draw(ax, paths=False)

    max_zone = 1

    # trajectories = []

    cnt_steps = 0

    while current_steps < num_steps:

      # Reset any counts and start the environment.
      start_time = time.time()
      self._rewarder.reset()

      episode_steps = 0
      episode_return = 0
      episode_imitation_return = 0
      timestep = self._environment.reset()

      self._actor.observe_first(timestep)

      # trajectory = []

      # Run an episode.
      while not timestep.last():
        action = self._actor.select_action(timestep.observation)
        obs_act = {'observation': timestep.observation, 'action': action}
        # print("obs_act = ",obs_act)
        # trajectory.append(obs_act)
        imitation_reward = self._rewarder.compute_reward(obs_act)
        timestep = self._environment.step(action)
        imitation_timestep = dm_env.TimeStep(step_type=timestep.step_type,
                                             reward=imitation_reward,
                                             discount=timestep.discount,
                                             observation=timestep.observation)

        new_zone = self._eval_zone(timestep.observation)
        if new_zone > max_zone:
            max_zone = new_zone

        # print("imitation_timestep = ", imitation_timestep)
        self._actor.observe(action, next_timestep=imitation_timestep)
        self._actor.update()

        # Book-keeping.
        episode_steps += 1
        episode_return += timestep.reward
        episode_imitation_return += imitation_reward
        cnt_steps += 1
      # print("max_zone = ", max_zone)

      # trajectories.append(trajectory)
      # ## save visual logs
      # if cnt_steps > 1000 :
      #     ## save current max zone
      #     self._zone_logfile_train.write(str(max_zone) + "\n")
      #     for traj in trajectories:
      #         X = [obs_act["observation"][0] for obs_act in traj]
      #         Y = [obs_act["observation"][1] for obs_act in traj]
      #         ax.plot(X,Y,color="pink",alpha=0.6)
      #     trajectories = []
      #     plt.savefig(self._logdir + "/train_" + str(it) + "_" + str(current_steps) + ".png")
      #     plt.close(fig)
      #
      #     cnt_steps = 0
      #     fig = plt.figure()
      #     ax = fig.add_subplot()
      #     self._environment.draw(ax, paths=False)

      # Collect the results and combine with counts.
      counts = self._counter.increment(episodes=1, steps=episode_steps)
      steps_per_second = episode_steps / (time.time() - start_time)
      result = {
          'episode_length': episode_steps,
          'episode_return': episode_return,
          'episode_return_imitation': episode_imitation_return,
          'steps_per_second': steps_per_second,
      }
      result.update(counts)

      self._logger.write(result)
      current_steps += episode_steps

    # plt.savefig(self._logdir + "/train_" + str(it) + "_" + str(current_steps) + ".png")
    # plt.close(fig)

  def _eval_zone(self, state):
    x = state[0]
    y = state[1]
    if y < 1.:
      if x < 1.:
        return 1
      elif  x < 2.:
        return 2
      elif  x < 3.:
        return 3
      elif  x < 4.:
        return 4
      else:
        return 5
    elif y < 2.:
      if  x > 4.:
        return 6
      elif  x > 3.:
        return 7
      elif x > 2.:
        return 8
      else:
        return 11
    elif y < 3.:
      if x < 1.:
        return 11
      elif x < 2.:
        return 10
      elif x < 3.:
        return 9
      elif x < 4.:
        return 20
      else :
        return 21

    elif y < 4.:
      if x < 1.:
        return 12
      elif x < 2.:
        return 15
      elif x < 3.:
        return 16
      elif x < 4:
        return 19
      else :
        return 22
    else:
      if x < 1.:
        return 13
      elif x < 2.:
        return 14
      elif x < 3.:
        return 17
      elif x < 4:
        return 18
      else :
        return 23


class EvalEnvironmentLoop(acme.core.Worker):
  """PWIL evaluation environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. This can be used as:

    loop = EvalEnvironmentLoop(environment, actor, rewarder)
    loop.run(num_episodes)

  The `Rewarder` overwrites the timestep from the environment to define
  a custom reward. The evaluation environment loop does not update the agent,
  and computes the wasserstein distance with expert demonstrations.

  The runner stores episode rewards and a series of statistics in the provided
  `Logger`.
  """

  def __init__(
      self,
      environment,
      actor,
      rewarder,
      counter=None,
      logger=None,
      workdir=None
  ):
    self._environment = environment
    self._actor = actor
    self._rewarder = rewarder
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger()
    self._workdir = workdir
    self._logdir = self._logger._file.name.split('/logs/')[0]
    self._zone_logfile_eval = open(self._logdir + "/eval_max_zone.txt", "w")

  def run(self, num_episodes,it):
    """Perform the run loop.

    Args:
      num_episodes: number of episodes to run the loop for.
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    self._environment.draw(ax, paths=False)

    max_zone = 1

    for i_episode in range(num_episodes):
      # Reset any counts and start the environment.
      start_time = time.time()
      self._rewarder.reset()

      episode_steps = 0
      episode_return = 0
      episode_imitation_return = 0
      timestep = self._environment.reset()

      max_zone = 1

      # Run an episode.
      trajectory = []
      while not timestep.last():
        action = self._actor.select_action(timestep.observation)
        obs_act = {'observation': timestep.observation, 'action': action}
        trajectory.append(obs_act)
        imitation_reward = self._rewarder.compute_reward(obs_act)

        timestep = self._environment.step(action)

        new_zone = self._eval_zone(timestep.observation)
        if new_zone > max_zone:
            max_zone = new_zone

        # Book-keeping.
        episode_steps += 1
        episode_return += timestep.reward
        episode_imitation_return += imitation_reward

      self._zone_logfile_eval.write(str(max_zone) + "\n")

      ## save visual logs
      X = [obs_act["observation"][0] for obs_act in trajectory]
      Y = [obs_act["observation"][1] for obs_act in trajectory]
      ax.plot(X,Y,color="royalblue",alpha=0.7)

      for obs_act in trajectory:
          self._environment.plot_car(obs_act["observation"], ax, alpha = 0.7, cabcolor="royalblue", truckcolor="royalblue")

      counts = self._counter.increment(episodes=1, steps=episode_steps)
      w2_dist = self._rewarder.compute_w2_dist_to_expert(trajectory)

      # Collect the results and combine with counts.
      steps_per_second = episode_steps / (time.time() - start_time)
      result = {
          'episode_length': episode_steps,
          'episode_return': episode_return,
          'episode_wasserstein_distance': w2_dist,
          'episode_return_imitation': episode_imitation_return,
          'steps_per_second': steps_per_second,
      }
      result.update(counts)

      self._logger.write(result)

    plt.savefig(self._logdir + "/eval_" + str(it) +  ".png")
    plt.close(fig)


  def _eval_zone(self, state):
    x = state[0]
    y = state[1]
    if y < 1.:
      if x < 1.:
        return 1
      elif  x < 2.:
        return 2
      elif  x < 3.:
        return 3
      elif  x < 4.:
        return 4
      else:
        return 5
    elif y < 2.:
      if  x > 4.:
        return 6
      elif  x > 3.:
        return 7
      elif x > 2.:
        return 8
      else:
        return 11
    elif y < 3.:
      if x < 1.:
        return 11
      elif x < 2.:
        return 10
      elif x < 3.:
        return 9
      elif x < 4.:
        return 20
      else :
        return 21

    elif y < 4.:
      if x < 1.:
        return 12
      elif x < 2.:
        return 15
      elif x < 3.:
        return 16
      elif x < 4:
        return 19
      else :
        return 22
    else:
      if x < 1.:
        return 13
      elif x < 2.:
        return 14
      elif x < 3.:
        return 17
      elif x < 4:
        return 18
      else :
        return 23
