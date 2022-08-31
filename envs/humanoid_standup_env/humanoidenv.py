import numpy as np

from gym import utils
# from gym.envs.mujoco import MuJocoPyEnv
from gym.spaces import Box

from gym.envs.mujoco import mujoco_env
import os


def mass_center(model, sim):
	mass = np.expand_dims(model.body_mass, 1)
	xpos = sim.data.xipos
	return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidStandupEnv(mujoco_env.MujocoEnv, utils.EzPickle):
	metadata = {
		"render_modes": [
			"human",
			"rgb_array",
			"depth_array",
			"single_rgb_array",
			"single_depth_array",
		],
		"render_fps": 67,
	}

	def __init__(self, **kwargs):
		observation_space = Box(
			low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
		)
		mujoco_env.MujocoEnv.__init__(
			self,
			"/assets/humanoidstandup.xml",
			5
		)
		utils.EzPickle.__init__(self)

	def _get_obs(self):
		data = self.sim.data
		return np.concatenate(
			[
				data.qpos.flat,
				data.qvel.flat,
				data.cinert.flat,
				data.cvel.flat,
				data.qfrc_actuator.flat,
				data.cfrc_ext.flat,
			]
		)

	def _get_state(self):
		return self._get_obs()

	def step(self, a):
		self.do_simulation(a, self.frame_skip)
		pos_after = self.sim.data.qpos[2]
		data = self.sim.data
		uph_cost = (pos_after - 0) / self.model.opt.timestep

		quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
		quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
		quad_impact_cost = min(quad_impact_cost, 10)
		reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1

		# self.renderer.render_step()

		done = bool(False)
		return (
			self._get_obs(),
			reward,
			done,
			dict(
				reward_linup=uph_cost,
				reward_quadctrl=-quad_ctrl_cost,
				reward_impact=-quad_impact_cost,
			),
		)

	def reset_model(self):
		c = 0.01
		self.set_state(
			self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
			self.init_qvel
			+ self.np_random.uniform(
				low=-c,
				high=c,
				size=self.model.nv,
			),
		)
		return self._get_obs()

	def reset(self):
		self.set_state(
			self.init_qpos,
			self.init_qvel,
		)
		return self._get_state()

	# get simulation state
	def get_inner_state(self):
		return self.sim.get_state()

	# set simulation state (more than just the position and velocity) to saved one
	def set_inner_state(self, saved_state):
		self.sim.set_state(saved_state)
		self.sim.forward()
		#return self.sim.get_state()
		return self._get_state()

	def viewer_setup(self):
		self.viewer.cam.trackbodyid = 1
		self.viewer.cam.distance = self.model.stat.extent * 1.0
		self.viewer.cam.lookat[2] = 2.0
		self.viewer.cam.elevation = -20
