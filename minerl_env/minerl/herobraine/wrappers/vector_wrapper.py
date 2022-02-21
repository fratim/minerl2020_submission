# Copyright (c) 2020 All Rights Reserved
# Author: William H. Guss, Brandon Houghton

import numpy as np
from functools import reduce
from collections import OrderedDict

from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.hero import spaces
from minerl.herobraine.wrappers.util import union_spaces, flatten_spaces, intersect_space
from minerl.herobraine.wrapper import EnvWrapper

import copy

AGENT_0_VECTORIZED = True

class Vectorized(EnvWrapper):
    """
    Normalizes and flattens a typical env space for obfuscation.
    common_envs : specified
    """

    def _update_name(self, name: str) -> str:
        return name.split('-')[0] + 'Vector-' + name.split('-')[-1]

    def __init__(self, env_to_wrap: EnvSpec, common_envs=None):
        self.env_to_wrap = env_to_wrap
        self.common_envs = [env_to_wrap] if common_envs is None or len(common_envs) == 0 else common_envs

        # Compute Action Spaces
        common_actions_all_agents = reduce(union_spaces, [env.actionables for env in self.common_envs])
        self.common_actions = OrderedDict({agent: copy.deepcopy(common_actions_all_agents) for agent in self.env_to_wrap.agent_names})

        # Remove Chat action for agent 1 (lead agent) from action space
        if self.env_to_wrap.agent_count > 1:
            for elem in self.common_actions["agent_1"]:
                if elem.to_string() == 'chat':
                    self.common_actions["agent_1"].remove(elem)

        self.flat_actions = OrderedDict({agent: flatten_spaces(self.common_actions[agent])[0] for agent in self.env_to_wrap.agent_names})
        self.remaining_action_space = OrderedDict({agent: flatten_spaces(self.common_actions[agent])[1] for agent in self.env_to_wrap.agent_names})
        self.action_vector_len = OrderedDict({agent: sum(space.shape[0] for space in self.flat_actions[agent]) for agent in self.env_to_wrap.agent_names})
        self.common_action_space = spaces.Dict(
            {agent: spaces.Dict({hdl.to_string(): hdl.space for hdl in self.common_actions[agent]}) for agent in self.env_to_wrap.agent_names})

        # Compute observation spaces
        common_observations_all_agent = reduce(union_spaces, [env.observables for env in self.common_envs])
        self.common_observations = OrderedDict({agent: copy.deepcopy(common_observations_all_agent) for agent in self.env_to_wrap.agent_names})

        # Remove Chat action for agent 1 (lead agent) from action space
        if self.env_to_wrap.agent_count > 1:
            for elem in self.common_observations["agent_1"]:
                if elem.to_string() == 'location_stats':
                    self.common_observations["agent_1"].remove(elem)

        self.flat_observations = OrderedDict({agent: flatten_spaces(self.common_observations[agent])[0] for agent in self.env_to_wrap.agent_names})
        self.remaining_observation_space = OrderedDict({agent: flatten_spaces(self.common_observations[agent])[1] for agent in self.env_to_wrap.agent_names})
        self.observation_vector_len = OrderedDict({agent: sum(space.shape[0] for space in self.flat_observations[agent]) for agent in self.env_to_wrap.agent_names})
        self.common_observation_space = spaces.Dict(
            {agent: spaces.Dict({hdl.to_string(): hdl.space for hdl in self.common_observations[agent]}) for agent in self.env_to_wrap.agent_names})

        # if self.env_to_wrap.agent_count == 1:
        #     for item in [self.common_actions, self.flat_actions, self.remaining_action_space, self.action_vector_len, self.common_action_space,
        #                  self.common_observations, self.flat_observations, self.remaining_observation_space, self.observation_vector_len, self.common_observation_space]:
        #         item = item["agent_0"]

        super().__init__(env_to_wrap)

    def _wrap_observation(self, obs: OrderedDict, agent) -> OrderedDict:
        if agent == "agent_1" or AGENT_0_VECTORIZED:
            flat_obs_part = self.common_observation_space[agent].flat_map(obs)
            wrapped_obs = self.common_observation_space[agent].unflattenable_map(obs)
            wrapped_obs['vector'] = flat_obs_part
            wrapped_obs['raw'] = obs
            return wrapped_obs
        else:
            return obs

    def _wrap_action(self, act: OrderedDict, agent) -> OrderedDict:
        if agent == "agent_1" or AGENT_0_VECTORIZED:
            flat_act_part = self.common_action_space[agent].flat_map(act)
            wrapped_act = self.common_action_space[agent].unflattenable_map(act)
            wrapped_act['vector'] = flat_act_part

            return wrapped_act
        else:
            return act

    def _unwrap_observation(self, obs: OrderedDict, agent) -> OrderedDict:
        if agent == "agent_1" or AGENT_0_VECTORIZED:

            assert np.max(obs['vector']) <= 1
            assert np.min(obs['vector']) >= 0

            full_obs = self.common_observation_space[agent].unmap_mixed(obs['vector'], obs)
            return intersect_space(self.env_to_wrap.observation_space[agent], full_obs)
        else:
            return obs

    def _unwrap_action(self, act: OrderedDict, agent) -> OrderedDict:
        if agent == "agent_1" or AGENT_0_VECTORIZED:

            assert np.max(act['vector']) <= 1
            assert np.min(act['vector']) >= 0

            full_act = self.common_action_space[agent].unmap_mixed(act['vector'], act)
            return intersect_space(self.env_to_wrap.action_space[agent], full_act)
        else:
            return act

    def create_observation_space(self):

        def get_observation_space_agent(agent):
            if agent == "agent_1" or AGENT_0_VECTORIZED:
                obs_list = self.remaining_observation_space[agent]
                obs_list.append(('vector', spaces.Box(low=0.0, high=1.0, shape=[self.observation_vector_len[agent]], dtype=np.float32)))
                return spaces.Dict(sorted(obs_list))
            else:
                return copy.deepcopy(self.env_to_wrap.observation_space[agent])

        ospace = spaces.Dict({aname: get_observation_space_agent(aname) for aname in self.agent_names})
        return ospace

        # for aname in self.agent_names:
        #     obs_list = self.remaining_observation_space
        #     # Todo: add maximum.
        #     obs_list.append(('vector', spaces.Box(low=0.0, high=1.0, shape=[self.observation_vector_len], dtype=np.float32)))
        #     ospace[aname] = spaces.Dict(sorted(obs_list))


    def create_action_space(self):
        def get_action_space_agent(agent):
            if agent == "agent_1" or AGENT_0_VECTORIZED:
                act_list = self.remaining_action_space[agent]
                act_list.append(('vector', spaces.Box(low=0.0, high=1.0, shape=[self.action_vector_len[agent]], dtype=np.float32)))
                return spaces.Dict(sorted(act_list))
            else:
                return copy.deepcopy(self.env_to_wrap.action_space[agent])

        aspace = spaces.Dict({aname: get_action_space_agent(aname) for aname in self.agent_names})
        return aspace


    def get_docstring(self):
        return self.env_to_wrap.get_docstring()
