import sys
import os
sys.path.append(os.getcwd())

from .fetch_env import MyComplexFetchEnv
from .task_manager_fetchenv import TasksManager

from matplotlib import collections  as mc
from matplotlib.patches import Circle
import gym
from gym import error, spaces
from gym.utils import seeding

gym._gym_disable_underscore_compat = True

import numpy as np
import math
import random
import copy
from collections import OrderedDict
import torch

import  pdb


class ComplexFetchEnvGCPHERSB3(gym.Env):


	def __init__(self, L_full_demonstration, L_full_inner_demonstration, L_states, starting_states, starting_inner_states, L_actions, L_full_observations, L_goals, L_inner_states, L_budgets, m_goals = None, std_goals = None, env_option = "", do_overshoot=True):

		self.env = MyComplexFetchEnv()

		## tasks
		self.tasks = TasksManager(L_full_demonstration, L_full_inner_demonstration, L_states, starting_states, starting_inner_states, L_actions, L_full_observations, L_goals, L_inner_states, L_budgets, m_goals, std_goals , env_option)

		self.max_steps = 50
		# Counter of steps per episode
		self.rollout_steps = 0

		self.action_space = self.env.env.action_space

		self.env_option = env_option

		self.incl_extra_full_state = 1

		self.m_goals = m_goals
		self.std_goals = std_goals

		if "full" in self.env_option :
			self.observation_space = spaces.Dict(
					{
						"observation": gym.spaces.Box(-3., 3., shape=(268 + 336 * self.incl_extra_full_state,), dtype='float32'),
						"achieved_goal": spaces.Box(
							low = np.array([-3,-3,-3,-3,-3,-3]),
							high = np.array([3,3,3,3,3,3])), # gripper_pos + object pos
						"desired_goal": spaces.Box(
							low = np.array([-3,-3,-3,-3,-3,-3]),
							high = np.array([3,3,3,3,3,3])),
					}
				)
		## add time to observation (Time-Aware agent)
		# elif "grasping" in self.env_option:
		# 	self.observation_space = spaces.Dict(
		# 			{
		# 				"observation": gym.spaces.Box(-3., 3., shape=(268 + 336 * self.incl_extra_full_state + 1,), dtype='float32'),
		# 				"achieved_goal": spaces.Box(
		# 					low = np.array([-3,-3,-3,0.]),
		# 					high = np.array([3,3,3,1.])), # gripper_pos + object pos
		# 				"desired_goal": spaces.Box(
		# 					low = np.array([-3,-3,-3,0.]),
		# 					high = np.array([3,3,3,1.])),
		# 			}
		# 		)

		elif "grasping" in self.env_option:
			self.observation_space = spaces.Dict(
					{
						"observation": gym.spaces.Box(-3., 3., shape=(268 + 336 * self.incl_extra_full_state,), dtype='float32'),
						"achieved_goal": spaces.Box(
							low = np.array([-3,-3,-3,0.]),
							high = np.array([3,3,3,1.])), # gripper_pos + object pos
						"desired_goal": spaces.Box(
							low = np.array([-3,-3,-3,0.]),
							high = np.array([3,3,3,1.])),
					}
				)

		else:
			self.observation_space = spaces.Dict(
					{
						"observation": gym.spaces.Box(-3., 3., shape=(268 + 336 * self.incl_extra_full_state,), dtype='float32'),

						"achieved_goal": spaces.Box(
							low = np.array([-3,-3,-3]),
							high = np.array([3,3,3])), # gripper_pos
						"desired_goal": spaces.Box(
							low = np.array([-3,-3,-3]),
							high = np.array([3,3,3])),
					}
				)


		# self.width_reward = width_reward ## hyper-parametre assez difficile à ajuster en dim 6
		self.width_success = 0.05


		self.total_steps = sum(self.tasks.L_budgets)

		self.traj_gripper = []
		self.traj_object = []

		self.testing = False
		self.expanded = False

		self.buffer_transitions = []

		self.bonus = True
		self.weighted_selection = True

		self.target_selection = False
		self.target_ratio = 0.3

		self.frame_skip = 1
		# self.frame_skip = 3

		self.target_reached = False
		self.overshoot = False
		self.do_overshoot = do_overshoot

		self.relabelling_shift_lookup_table = self._init_relabelling_lookup_table()

	def _init_relabelling_lookup_table(self,
		):
		"""
        create a table to associate a goal to its corresponding next goal for efficient
        computation of value function in bonus reward for relabelled transition made by
        HER.

		IMPORTANT: use np.around to get correspondance after normalize + unnormalize
        """

		lookup_table = {}

		for i in range(0, len(self.tasks.L_states)):
			# lookup_table[tuple(np.around(self.project_to_goal_space(self.tasks.L_states[i]),5).astype(np.float32))] = tuple(self.project_to_goal_space(self.tasks.L_states[i]).astype(np.float32))
			# lookup_table[tuple([round(el,5) for el in self.project_to_goal_space(self.tasks.L_states[i]).astype(np.float32).tolist()])] = tuple(self.project_to_goal_space(self.tasks.L_states[i]).astype(np.float32))
			lookup_table[tuple(self.project_to_goal_space(self.tasks.L_states[i]).astype(np.float32))] = tuple(self.project_to_goal_space(self.tasks.L_states[i]).astype(np.float32))
		return lookup_table

	def _add_to_lookup_table(self,
		state,
		# achieved_goal,
		desired_goal,
		):
		"""
        associate state to next goal
        """
		# self.relabelling_shift_lookup_table[tuple(np.around(self.project_to_goal_space(state),5).astype(np.float32))] = tuple(desired_goal.astype(np.float32))
		# self.relabelling_shift_lookup_table[tuple([round(el, 5) for el in self.project_to_goal_space(state).astype(np.float32).tolist()])] = tuple(desired_goal.astype(np.float32))
		self.relabelling_shift_lookup_table[tuple(self.project_to_goal_space(state).astype(np.float32))] = tuple(desired_goal.astype(np.float32))


		return

	def divide_task(self,
		new_subgoal):

		self.relabelling_shift_lookup_table[tuple(self.project_to_goal_space(new_subgoal).astype(np.float32))] = tuple(self.project_to_goal_space(new_subgoal).astype(np.float32))

		return

	def skip_task(self,
		goal_indx):
		"""
		update next goals associated to states after goal goal_indx was removed
		"""

		for key in self.relabelling_shift_lookup_table.keys():
			if (np.array(list(self.relabelling_shift_lookup_table[key])).astype(np.float32) == self.tasks.L_goals[goal_indx].astype(np.float32)).all():
				self.relabelling_shift_lookup_table[key] = tuple(self.tasks.L_goals[goal_indx+1].astype(np.float32))

		return

	def compute_distance_in_goal_space(self, in_goal1, in_goal2):
		"""
		goal1 = achieved_goal
		goal2 = desired_goal
		"""

		goal1 = copy.deepcopy(in_goal1)
		goal2 = copy.deepcopy(in_goal2)

		if "grasping" in self.env_option:

			## single goal
			if len(goal1.shape) ==  1:
				euclidian_goal1 = goal1[:3]
				euclidian_goal2 = goal2[:3]

				if goal1[3] == goal2[3]: ## grasping boolean
					return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1)

				else:
					return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1) + 1000000 ## if no grasping, artificially push goals far away
					# return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1)
			## tensor of goals
			else:
				euclidian_goal1 = goal1[:,:3]
				euclidian_goal2 = goal2[:,:3]

				goal1_bool = goal1[:,3]
				goal2_bool = goal2[:,3]

				grasping_penalty = ((goal1_bool == goal2_bool).astype(np.float32)-1)*(-1000000)

				assert np.linalg.norm(euclidian_goal1[:,:] - euclidian_goal2[:,:], axis=-1).size == grasping_penalty.size

				return np.linalg.norm(euclidian_goal1[:,:] - euclidian_goal2[:,:], axis=-1) + grasping_penalty

		else:
			if len(goal1.shape) ==  1:
				return np.linalg.norm(goal1 - goal2, axis=-1)
			else:
				return np.linalg.norm(goal1[:,:] - goal2[:,:], axis=-1)


	def compute_reward(self, achieved_goal, desired_goal, info):
		"""
        compute the reward according to distance in goal space
        R \in {0,1}
        """
        ### single goal
		if len(achieved_goal.shape) ==  1:
			dst = self.compute_distance_in_goal_space(achieved_goal, desired_goal)

			_info = {'reward_boolean': dst<= self.width_success}

			if _info['reward_boolean']:
				return 10.#1.#0.
			else:
				return 0.#-1.

		### tensor of goals
		else:
			distances = self.compute_distance_in_goal_space(achieved_goal, desired_goal)
			distances_mask = (distances <= self.width_success).astype(np.float32)

			# rewards = distances_mask - 1. #- distances_mask * 0.1 # {-1, -0.1}
			rewards = distances_mask *10.

			return rewards

	def step(self, action) :
		"""
        step of the environment

        3 cases:
            - target reached
            - time limit
            - else
        """
		state = self.env.get_state()

		for i in range(self.frame_skip):
			new_state, reward, done, info = self.env.step(action)

			new_inner_state = self.env.get_restore()

			gripper_pos = self.get_gripper_pos(new_state)
			object_pos = self.get_object_pos(new_state) # best way to access object position so far

			self.traj_gripper.append(gripper_pos)
			self.traj_object.append(object_pos)

			if self.tasks.subgoal_adaptation and not self.tasks.skipping:
				self.tasks.add_new_starting_state(self.tasks.indx_goal, new_inner_state, new_state)

		# self._add_to_lookup_table(new_state, self.goal)

		self.rollout_steps += 1

		dst = self.compute_distance_in_goal_space(self.project_to_goal_space(new_state),  self.goal)
		info = {'target_reached': dst<= self.width_success}

		info['goal_indx'] = copy.deepcopy(self.tasks.indx_goal)
		info['goal'] = copy.deepcopy(self.goal)

		if info['target_reached']: # achieved goal

			self.target_reached = True

			self.tasks.add_success(self.tasks.indx_goal)

			if self.tasks.skipping: ## stop skipping if overshooting
				self.tasks.skipping = False

			done = True
			reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, info)

			prev_goal = self.goal.copy()
			info['done'] = done
			info['goal'] = self.goal.copy()
			info['traj'] = [self.traj_gripper, self.traj_object]

			## update subgoal trial as success if successful overshoot
			if self.tasks.subgoal_adaptation and self.overshoot and not self.tasks.skipping:
				self.tasks.update_overshoot_result(self.tasks.indx_goal - self.tasks.delta_step, self.subgoal, True)

			## add time to observation
			return OrderedDict([
					("observation", new_state.copy()), ## TODO: what's the actual state?
					("achieved_goal", self.project_to_goal_space(new_state).copy()),
					("desired_goal", prev_goal)]), reward, done, info

			# TA_new_state = np.concatenate((new_state, np.array([(self.max_steps - self.rollout_steps)/self.max_steps])))
			# return OrderedDict([
			# 		("observation", TA_new_state.copy()), ## TODO: what's the actual state?
			# 		("achieved_goal", self.project_to_goal_space(new_state).copy()),
			# 		("desired_goal", prev_goal)]), reward, done, info

		elif self.rollout_steps >= self.max_steps:
			### failed task
			self.target_reached = False

			## add failure to task results
			self.tasks.add_failure(self.tasks.indx_goal)

			reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, info)

			done = True ## no done signal if timeout (otherwise non-markovian process)

			prev_goal = self.goal.copy()
			info['done'] = done
			info['goal'] = self.goal.copy()
			info['traj'] = [self.traj_gripper, self.traj_object]

			## time limit for SB3s
			# info["TimeLimit.truncated"] = True

			## add failure to overshoot result
			if self.tasks.subgoal_adaptation and self.overshoot and not self.tasks.skipping:
				# print("Failed task")
				# print("info['subgoal_indx'] = ", info["subgoal_indx"])
				self.tasks.update_overshoot_result(self.tasks.indx_goal - self.tasks.delta_step, self.subgoal, False)

			return OrderedDict([
					("observation", new_state.copy()),
					("achieved_goal", self.project_to_goal_space(new_state).copy()),
					("desired_goal", prev_goal)]), reward, done, info

			## add time to observation
			# TA_new_state = np.concatenate((new_state, np.array([(self.max_steps - self.rollout_steps)/self.max_steps])))
			# return OrderedDict([
			# 		("observation", TA_new_state.copy()),
			# 		("achieved_goal", self.project_to_goal_space(new_state).copy()),
			# 		("desired_goal", prev_goal)]), reward, done, info
		else:

			done = False

			self.target_reached = False

			reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, info)

			info['done'] = done

			return OrderedDict([
					("observation", new_state.copy()),
					("achieved_goal", self.project_to_goal_space(new_state).copy()),
					("desired_goal", self.goal.copy()),]), reward, done, info

			## add time to observation
			# TA_new_state = np.concatenate((new_state, np.array([(self.max_steps - self.rollout_steps)/self.max_steps])))
			# return OrderedDict([
			# 		("observation", TA_new_state.copy()),
			# 		("achieved_goal", self.project_to_goal_space(new_state).copy()),
			# 		("desired_goal", self.goal.copy()),]), reward, done, info
			#

	def step_test(self, action) :
		"""
        step method for evaluation -> no reward computed, no time limit etc.
        """
		for i in range(self.frame_skip):
			new_state, reward, done, info = self.env.step(action)

			gripper_pos = self.get_gripper_pos(new_state)
			object_pos = self.get_object_pos(new_state) # best way to access object position so far

			self.traj_gripper.append(gripper_pos)
			self.traj_object.append(object_pos)

		self.rollout_steps += 1

		dst = np.linalg.norm(self.project_to_goal_space(new_state) - self.goal)
		info = {'target_reached': dst<= self.width_success}

		#reward = 0.

		return OrderedDict([
				("observation", new_state.copy()),
				("achieved_goal", self.project_to_goal_space(new_state).copy()),
				("desired_goal", self.goal.copy()),]), reward, done, info

		## add time to observation
		# TA_new_state = np.concatenate((new_state, np.array([(self.max_steps - self.rollout_steps)/self.max_steps])))
		# return OrderedDict([
		# 		("observation", TA_new_state.copy()),
		# 		("achieved_goal", self.project_to_goal_space(new_state).copy()),
		# 		("desired_goal", self.goal.copy()),]), reward, done, info

	def _get_obs(self):

		state = self.env.get_state()
		achieved_goal = self.project_to_goal_space(state)

		return OrderedDict(
			[
				("observation", state.copy()),
				("achieved_goal", achieved_goal.copy()),
				("desired_goal", self.goal.copy()),
			]
		)

		## add time to observation
		# TA_state = np.concatenate((state, np.array([(self.max_steps - self.rollout_steps)/self.max_steps])))
		# return OrderedDict(
		# 	[
		# 		("observation", TA_state.copy()),
		# 		("achieved_goal", achieved_goal.copy()),
		# 		("desired_goal", self.goal.copy()),
		# 	]
		# )

	def goal_vector(self):
		return self.goal

	def set_state(self, inner_state):
		# print("inner_state = ", inner_state)
		# self.env.restore(inner_state)

		self.env.env.set_inner_state(inner_state)

	def set_goal_state(self, goal_state):
		self.goal_state = goal_state
		# print("goal_state = ", goal_state)
		##TODO: should we normalize here?
		# norm_goal_state = self.norm_wrapper.normalize_obs(goal_state)

		self.goal = self.project_to_goal_space(goal_state)
		# print("self.goal = ", self.goal)
		## force grasping if starting state validate grasping condition
		# if self.check_grasping(self.env.get_state()):
		# 	self.goal[-1] = 1
			# print("self.goal = ", self.goal)
		return 0

	def check_grasping(self, state):

		collision_l_gripper_link_obj = state[216 + 167]
		collision_r_gripper_link_obj = state[216 + 193]
		collision_object_table = state[216 + 67] ## add collision between object and table to improve grasing check

		## if quaternion angles
		# collision_l_gripper_link_obj = state[234 + 167]
		# collision_r_gripper_link_obj = state[234 + 193]
		# collision_object_table = state[234 + 67]

		if collision_l_gripper_link_obj and collision_r_gripper_link_obj and not collision_object_table :
			grasping = 1
		else:
			grasping = 0

		return grasping

	def project_to_goal_space(self, state):
		"""
        Env-dependent projection of a state in the goal space.
        In a fetchenv -> keep (x,y,z) coordinates of the gripper + 0,1 boolean
		if the object is grasped or not.
        """

		gripper_pos = self.get_gripper_pos(state)
		object_pos = self.get_object_pos(state)
		gripper_velp = self.get_gripper_velp(state)
		gripper_quat = self.get_gripper_quat(state)
		gripper_euler = self.get_gripper_euler(state)

		norm_gripper_velp = np.linalg.norm(gripper_velp)

		if "full" in self.env_option:
			return np.concatenate((np.array(gripper_pos), np.array(object_pos)))
		elif "grasping" in self.env_option:
			bool_grasping = self.check_grasping(state)
			return np.concatenate((np.array(gripper_pos), np.array([int(bool_grasping)])))
		else:
			return np.array(gripper_pos)

	def get_gripper_pos(self, state):
		"""
		get gripper position from full state
		"""
		# print("state = ", state)
		assert len(list(state))== 268 + 336 * self.incl_extra_full_state or len(list(state))== 268
		# assert len(list(state)) == 268 + 336 * self.incl_extra_full_state +1  or len(list(state))== 268

		gripper_pos = state[84:87]
		# gripper_pos = state[102:105]

		assert len(list(gripper_pos)) == 3

		return gripper_pos

	def get_object_pos(self, state):
		"""
		get object position from full state
		"""
		assert len(list(state)) == 268 + 336 * self.incl_extra_full_state or len(list(state))== 268

		object_pos = state[105:108]
		# object_pos = state[123:126]
		assert len(list(object_pos))==3

		# print("indx object pos = ", indx)

		return object_pos

	def get_gripper_velp(self, state):
		"""
		get object position from full state
		"""
		assert len(list(state)) == 268 + 336 * self.incl_extra_full_state or len(list(state))== 268

		gripper_velp = state[138:141]
		# gripper_velp = state[156:159]
		assert len(list(gripper_velp))==3

		return gripper_velp

	def get_gripper_quat(self, state):
		"""
		get object orientation from full state
		"""
		assert len(list(state)) == 268 + 336 * self.incl_extra_full_state or len(list(state))== 268

		gripper_quat = state[36:40]
		assert len(list(gripper_quat))==4

		return gripper_quat

	def get_gripper_euler(self, state):
		"""
		get object orientation from full state
		"""
		assert len(list(state)) == 268 + 336 * self.incl_extra_full_state or len(list(state))== 268

		gripper_quat = state[27:30] #TODO
		assert len(list(gripper_quat))==3

		return gripper_quat


	def select_task(self):
		"""
		Sample task for low-level policy training.

		"""
		return self.tasks.select_task()

	def reset_task_by_nb(self, task_nb):

		self.env.reset()

		starting_state, length_task, goal_state = self.tasks.get_task(task_nb)

		self.set_goal_state(goal_state)
		self.set_state(starting_state)
		self.max_steps = length_task
		return

	def advance_task(self):
		goal_state, length_task, advance_bool = self.tasks.advance_task()

		if advance_bool:

			self.set_goal_state(goal_state)
			self.max_steps = length_task
			self.rollout_steps  = 0

		return advance_bool


	def reset(self, eval = False):
		"""
		Reset environment.

		2 cases:
			- reset after success -> try to overshoot
					if a following task exists -> overshoot i.e. update goal, step counter
					and budget but not the current state
					else -> reset to a new task
			- reset to new task
		"""
		## Case 1 - success -> automatically try to overshoot
		if self.target_reached and self.do_overshoot: ## automatically overshoot
			self.subgoal = self.goal.copy()
			advance_bool = self.advance_task()
			self.target_reached = False

            ## shift to a next task is possible (last task not reached)
			if advance_bool:
				# pdb.set_trace()
				state = copy.deepcopy(self.env.get_state())
				self.overshoot = True

				return OrderedDict([
						("observation", state.copy()),
						("achieved_goal", self.project_to_goal_space(state).copy()),
						("desired_goal", self.goal.copy()),])

				## add time to observation
				# TA_state = np.concatenate((state, np.array([(self.max_steps - self.rollout_steps)/self.max_steps])))
				# return OrderedDict([
				# 		("observation", TA_state.copy()),
				# 		("achieved_goal", self.project_to_goal_space(state).copy()),
				# 		("desired_goal", self.goal.copy()),])

			## shift impossible (current task is the last one)
			else:
				#pdb.set_trace()
				self.overshoot = False
				self.target_reached = False
				out_state = self.reset()
				return out_state

		## Case 2 - no success: reset to new task
		else:
			# print("true reset")
			self.env.reset()

			self.testing = False
			self.skipping = False
			self.tasks.skipping = False

			self.overshoot = False

			starting_state, length_task, goal_state = self.select_task()

			self.set_state(starting_state)
			self.set_goal_state(goal_state)

			self.max_steps = length_task

			self.rollout_steps = 0
			self.traj_gripper = []
			self.traj_object = []

			state = copy.deepcopy(self.env.get_state())
			gripper_pos = self.get_gripper_pos(state)
			object_pos = self.get_object_pos(state) # best way to access object position so far

			self.traj_gripper.append(gripper_pos)
			self.traj_object.append(object_pos)

			return OrderedDict([
					("observation", state.copy()),
					("achieved_goal", self.project_to_goal_space(state).copy()),
					("desired_goal", self.goal.copy()),])

			## add time to observation
			# TA_state = np.concatenate((state, np.array([(self.max_steps - self.rollout_steps)/self.max_steps])))
			# return OrderedDict([
			# 		("observation", TA_state.copy()),
			# 		("achieved_goal", self.project_to_goal_space(state).copy()),
			# 		("desired_goal", self.goal.copy()),])
