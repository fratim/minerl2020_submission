# Copyright (c) 2020 All Rights Reserved
# Author: William H. Guss, Brandon Houghton

import abc
import copy
from collections import OrderedDict

from minerl.herobraine.env_spec import EnvSpec
import minerl


class EnvWrapper(EnvSpec):

    def __init__(self, env_to_wrap: EnvSpec):
        self.env_to_wrap = env_to_wrap
        self._wrap_act_fn, self._wrap_obs_fn, self._unwrap_act_fn, self._unwrap_obs_fn = None, None, None, None
        if isinstance(self.env_to_wrap, EnvWrapper):
            self._wrap_act_fn = self.env_to_wrap.wrap_action
            self._wrap_obs_fn = self.env_to_wrap.wrap_observation
            self._unwrap_act_fn = self.env_to_wrap.unwrap_action
            self._unwrap_obs_fn = self.env_to_wrap.unwrap_observation

        super().__init__(self._update_name(env_to_wrap.name),
                         max_episode_steps=env_to_wrap.max_episode_steps,
                         reward_threshold=env_to_wrap.reward_threshold,
                         agent_count=env_to_wrap.agent_count)

    @abc.abstractmethod
    def _update_name(self, name: str) -> str:
        pass

    @abc.abstractmethod
    def _wrap_observation(self, obs: OrderedDict, agent) -> OrderedDict:
        pass

    def wrap_observation(self, obs: OrderedDict, agent):
        # self = obfuscated
        # env_to_wrap = vector
        # obs is just a treechop ob
        obs = copy.deepcopy(obs)
        if self._wrap_obs_fn is not None:
            obs = self._wrap_obs_fn(obs, agent)

        if minerl.utils.test.SHOULD_ASSERT: assert obs in self.env_to_wrap.observation_space

        # wrapped_obs = spaces.Dict({agent: self._wrap_observation(obs, agent)} for agent in self.agent_names)
        wrapped_obs = self._wrap_observation(obs, agent)

        if minerl.utils.test.SHOULD_ASSERT: assert wrapped_obs in self.observation_space
        return wrapped_obs

    @abc.abstractmethod
    def _wrap_action(self, act: OrderedDict, agent) -> OrderedDict:
        pass

    def wrap_action(self, act: OrderedDict, agent):
        act = copy.deepcopy(act)
        if self._wrap_act_fn is not None:
            act = self._wrap_act_fn(act, agent)

        if minerl.utils.test.SHOULD_ASSERT: assert act in self.env_to_wrap.action_space

        #wrapped_act = spaces.Dict({agent: self._wrap_action(act, agent)} for agent in self.agent_names)
        wrapped_act = self._wrap_action(act, agent)

        if minerl.utils.test.SHOULD_ASSERT: assert wrapped_act in self.action_space
        return wrapped_act

    @abc.abstractmethod
    def _unwrap_observation(self, obs: OrderedDict, agent) -> OrderedDict:
        pass

    def unwrap_observation(self, obs: OrderedDict, agent) -> OrderedDict:
        obs = copy.deepcopy(obs)
        if minerl.utils.test.SHOULD_ASSERT: assert obs in self.observation_space

        #obs = spaces.Dict({agent: self._unwrap_observation(obs, agent)} for agent in self.agent_names)
        obs = self._unwrap_observation(obs, agent)

        if minerl.utils.test.SHOULD_ASSERT: assert obs in self.env_to_wrap.observation_space

        if self._unwrap_obs_fn is not None:
            obs = self._unwrap_obs_fn(obs, agent)

        return obs

    @abc.abstractmethod
    def _unwrap_action(self, act: OrderedDict, agent) -> OrderedDict:
        pass


    def unwrap_action(self, act: OrderedDict, agent) -> OrderedDict:
        act = copy.deepcopy(act)
        if minerl.utils.test.SHOULD_ASSERT: assert act in self.action_space

        # act = spaces.Dict({agent: self._unwrap_action(act)} for agent in self.agent_names)
        act = self._unwrap_action(act, agent)

        if minerl.utils.test.SHOULD_ASSERT: assert act in self.env_to_wrap.action_space

        if self._unwrap_act_fn is not None:
            act = self._unwrap_act_fn(act, agent)

        return act

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return self.env_to_wrap.determine_success_from_rewards(rewards)

    def create_observation_space(self):
        return self.env_to_wrap.observation_space

    def create_action_space(self):
        return self.env_to_wrap.action_space

    def get_docstring(self):
        return self.env_to_wrap.get_docstring()

    def is_from_folder(self, folder: str) -> bool:
        return self.env_to_wrap.is_from_folder(folder)

    # TODO: SEE IF THIS SHOULD BE CALLED OR NOT.
    def create_actionables(self):
        return self.env_to_wrap.create_actionables()

    def create_observables(self):
        return self.env_to_wrap.create_observables()

    def create_rewardables(self):
        return self.env_to_wrap.create_rewardables()

    def create_agent_start(self):
        return self.env_to_wrap.create_agent_start()

    def create_agent_handlers(self):
        return self.env_to_wrap.create_agent_handlers()

    def create_server_world_generators(self):
        return self.env_to_wrap.create_server_world_generators()

    def create_server_quit_producers(self):
        return self.env_to_wrap.create_server_quit_producers()

    def create_server_decorators(self):
        return self.env_to_wrap.create_server_decorators()

    def create_server_initial_conditions(self):
        return self.env_to_wrap.create_server_initial_conditions()

    def create_monitors(self):
        return self.env_to_wrap.create_monitors()
